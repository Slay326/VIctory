//! Diagnostics of the BVH builder's results (local test branch, not for
//! upstream): does a builder stage compute wrong structures, or do right
//! ones get lost on their way into title memory?
//!
//! `VICTORY_BVH_GT=<shader hash>` (a prefix such as `5552dac6`): after each
//! submission with a list that bound a pipeline of that shader, the pages
//! the list's shaders marked as written through the page table
//! (`crate::memory`) are compared three ways:
//! * G, the host copy as the shaders left it, right after the submission's
//!   fence (of a copy in the GPU's memory fetched from there, its CPU side
//!   left as it was), before anything goes back;
//! * T, title memory at the same addresses after the copy-backs, and again
//!   after the bottom-of-pipe writes;
//! * the list's misses (pages and 256 MiB regions with no copy).
//!
//! A validator ([`check`]) looks for structure headers in G (256-byte
//! aligned, version 6) and checks each: root pointer, bounds, child boxes
//! inside their parent's, empty boxes with a child, the parent table; it
//! counts empty box16 boxes and +-FLT_MAX pairs in G and T. One compact
//! log line per part, the first three findings with their addresses.
//!
//! A second validator (senior 96, `BVH GT s.. v2` lines next to the old
//! ones, whose numbers stay comparable with earlier runs): bottom levels
//! apart by their marks (fresh 0x8004, rehydrated 0x8002, still packed
//! 0x8102 with their own check [`check_packed`], other), node boxes of a
//! structure with a scale word in its stored units, rehydrated ones
//! counted without and with the bit-30 rule (centre cb0..2 of the
//! structure's 5e687674 dispatch, noted at its bind, else the header
//! one-sided rule), and top levels by [`check_top`] (every instance reached
//! once, no bad slots, instances by the kind of their bottom level, only
//! those on rehydrated ones must have a finite box), following instances to
//! bottom levels outside the compared pages
//! (`VICTORY_BVH_GT_TOP_PAGES`, 256 pages of 64 KiB per submission).
//!
//! A third validator (senior 100, `BVH GT s.. v3` lines, Fable Aufruf 7
//! point 4b): every structure valid, invalid or not judged ([`Verdict`]);
//! not judged (header, box nodes, instances or a bottom level not read,
//! pages past the budget, a header that is no plausible top level) counts
//! neither way and is shown apart. Bottom levels end exactly at D1 * 64, a
//! top level's instance list must lie within D1 * 64 (its box nodes may end
//! at D1 * 64 + 64, measured on the harness dumps), and rehydrated ones with
//! the header one-sided fallback are counted apart from the bit-30 rule's.
//! v3c (senior 109): box nodes of every structure may end by D1 * 64 + 64
//! (a box32 root in the last 64 bytes); 0x8002 / 0x8004 nodes still start
//! inside D1 * 64.
//! v3d (senior 112, Fable Aufruf 8 points 2, 7, 8): every reached instance
//! of a top level falls in one class: (a) a bottom-level header at its
//! address A; (b) none there, but one at A + k * 64 whose parent table
//! (dword 14) starts at A; else by a complete map of the builder's output
//! addresses (every bind of the stages in [`BUILD_TARGETS`]) (c) built only
//! after the top level, (d) built before it and still no header, (e) never a
//! build target so far. Only (d) is a defect; (c) and (e) leave the top
//! level not judged. Bottom levels with dword 15 bits 24-29 set are not
//! judged (the bit-30 rule is read from data, not from the shader source).
//!
//! `VICTORY_BVH_GT_PAGES` (64): the most pages a submission compares.
//! `VICTORY_BVH_GT_MAX` (200): the most submissions compared in a run.
//! `VICTORY_BVH_UAV_BARRIER=1`: a global UAV barrier before every dispatch
//! of the builder (`crate::draw`), against races between its stages'
//! atomics and later reads in one list.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::shadow::Shadow;

/// The shader hash `VICTORY_BVH_GT` names, lower case.
pub(crate) fn target() -> Option<&'static str> {
    static TARGET: OnceLock<Option<String>> = OnceLock::new();
    TARGET
        .get_or_init(|| {
            std::env::var("VICTORY_BVH_GT")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .filter(|v| !v.is_empty())
        })
        .as_deref()
}

/// Whether every builder dispatch gets a global UAV barrier before it.
pub(crate) fn uav_barrier() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_UAV_BARRIER").is_ok_and(|v| v.trim() == "1"))
}

fn number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn most_pages() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_GT_PAGES", 64) as usize)
}

fn most_submissions() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_GT_MAX", 200))
}

/// `VICTORY_BVH_GT_READER=<shader hash>` (a ray tracing program, such as
/// f607780a): after each submission with a list that bound it, the headers
/// the last builder comparison found are checked again in what the GPU
/// holds now, the structures as the readers saw them (the builder's later
/// stages done, or what withholding left).
pub(crate) fn reader_target() -> Option<&'static str> {
    static TARGET: OnceLock<Option<String>> = OnceLock::new();
    TARGET
        .get_or_init(|| {
            std::env::var("VICTORY_BVH_GT_READER")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .filter(|v| !v.is_empty())
        })
        .as_deref()
}

#[derive(Default)]
struct Last {
    submission: u64,
    headers: Vec<u64>,
    /// Each header's validity in G and the class of its first reason
    /// ([`why_class`]), for the two-point check in [`reader_check`].
    g: Vec<(bool, String)>,
    /// Copy (none: an arena page), page, where its bytes start.
    pages: Vec<(Option<Arc<Shadow>>, usize, usize)>,
}

static LAST: std::sync::Mutex<Last> = std::sync::Mutex::new(Last { submission: 0, headers: Vec::new(), g: Vec::new(), pages: Vec::new() });
static READER_CHECKS: AtomicU64 = AtomicU64::new(0);

/// What kind of structure a header's dword 0 names: still packed, a top
/// level one whose bounds are still the builder's start values (being
/// built), or one to check.
fn kind(d0: u32, bounds_start: bool) -> &'static str {
    if d0 >> 16 & 0x100 != 0 {
        "packed"
    } else if d0 >> 16 & 0x8000 == 0 && bounds_start {
        "top,building"
    } else if d0 >> 16 & 0x8000 == 0 {
        "top"
    } else {
        "bottom"
    }
}

/// The kind of the structure whose header is at `at` ([`kind`]).
pub(crate) fn header_kind(read: &dyn Fn(u64) -> Option<u32>, at: u64) -> &'static str {
    let start = (4..10).any(|i| read(at + 4 * i).is_some_and(|w| f32::from_bits(w).abs() == f32::MAX));
    kind(read(at).unwrap_or(0), start)
}

/// After submission `submission`, whose lists bound the reader program: the
/// last builder comparison's headers checked in the GPU's bytes now.
pub(crate) fn reader_check(submission: u64) {
    if READER_CHECKS.fetch_add(1, Ordering::Relaxed) >= most_submissions() {
        return;
    }
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.headers.is_empty() {
        return;
    }
    let wanted: Vec<u64> = last.pages.iter().filter(|p| p.0.is_none()).map(|p| p.1 as u64).collect();
    let mut now: Vec<(usize, Vec<u8>)> = crate::arena::fetch(&wanted).into_iter().map(|(at, g)| (at as usize, g)).collect();
    now.extend(last.pages.iter().filter_map(|(copy, page, _)| copy.as_ref()?.gpu_bytes(*page)));
    let read = reader(&now);
    let mut by_kind: std::collections::BTreeMap<&str, (u32, u32)> = std::collections::BTreeMap::new();
    let mut first = Vec::new();
    let (mut outside, mut origin) = (0u32, 0u32);
    // Two-point check of bottom-level structures (by kind in the readers'
    // bytes): valid or not in G and for the readers, and whether an invalid
    // one fails for the same class of reason at both points.
    let mut pair = [0u32; 5];
    for (n, &at) in last.headers.iter().enumerate() {
        let (Some(d0), Some(d1)) = (read(at), read(at + 4)) else { continue };
        if !is_header(d0, d1) {
            let e = by_kind.entry("gone").or_default();
            e.0 += 1;
            continue;
        }
        let start = (4..10).any(|i| read(at + 4 * i).is_some_and(|w| f32::from_bits(w).abs() == f32::MAX));
        let k = kind(d0, start);
        let c = check(&read, at);
        if k == "bottom" {
            outside += c.outside;
            origin += c.origin;
            if let Some((g_valid, g_class)) = last.g.get(n) {
                let i = usize::from(!*g_valid) * 2 + usize::from(!c.valid);
                pair[i] += 1;
                if !*g_valid && !c.valid && *g_class == why_class(c.why.as_deref()) {
                    pair[4] += 1;
                }
            }
        }
        let e = by_kind.entry(k).or_default();
        e.0 += 1;
        e.1 += u32::from(c.valid);
        if !c.valid && k == "bottom" && first.len() < 3 {
            let why = c.why.as_deref().unwrap_or("?");
            first.push(format!("{at:#x} {why}"));
            // The first one raw: header, root node and the node the reason
            // names last (its parent for a child box).
            if first.len() == 1 {
                let words = |a: u64, n: u64| -> String {
                    (0..n).map(|i| read(a + 4 * i).map_or_else(|| "-".to_string(), |w| format!("{w:08x}"))).collect::<Vec<_>>().join(" ")
                };
                let node = |p: u32| u64::from(p & !7) << 3;
                let root = read(at + 48).unwrap_or(0);
                let named = why
                    .rsplit("0x")
                    .next()
                    .and_then(|t| u32::from_str_radix(&t.chars().take_while(char::is_ascii_hexdigit).collect::<String>(), 16).ok())
                    .unwrap_or(0);
                tracing::info!(
                    target: "victory_gpu::bvh_gt",
                    "BVH GT reader s{submission} raw {at:#x}: header [{}]; root {root:#x} [{}]; named {named:#x} at +{:#x} [{}]",
                    words(at, 16),
                    words(at + node(root), 32),
                    node(named),
                    words(at + node(named), 32),
                );
            }
        }
    }
    let kinds: Vec<String> = by_kind.iter().map(|(k, (n, v))| format!("{k} {v}/{n}")).collect();
    tracing::info!(
        target: "victory_gpu::bvh_gt",
        "BVH GT reader s{submission} (headers of builder s{}): valid by kind {}; bottom child boxes outside {outside}, of them only by the origin {origin}; first invalid bottom: {}",
        last.submission,
        kinds.join(", "),
        first.join(", "),
    );
    // The second validator on the readers' bytes (top levels only with
    // what these pages hold: nothing is read besides).
    let built = |top: u64, blas: u64| built_of(&builds(), top, blas);
    let next = |at: u64| builds().map.range(at + 1..).next().map(|(a, _)| *a);
    let (v2, v3) = validate(
        &now.iter().map(|(f, b)| (*f, &b[..])).collect::<Vec<_>>(),
        &last.headers,
        false,
        &Targets { built: &built, next: &next },
    );
    tracing::info!(
        target: "victory_gpu::bvh_gt",
        "BVH GT reader s{submission} v2 (headers of builder s{}): {}; first invalid: {}",
        last.submission,
        v2.text(),
        v2.first.join(", "),
    );
    tracing::info!(
        target: "victory_gpu::bvh_gt",
        "BVH GT reader s{submission} v3 (headers of builder s{}): {}; first invalid: {}; first not judged: {}",
        last.submission,
        v3.text(),
        v3.first.join(", "),
        v3.first_not_judged.join(", "),
    );
    let mut total = TWO_POINT.lock().unwrap_or_else(|e| e.into_inner());
    for (t, p) in total.iter_mut().zip(pair) {
        *t += u64::from(p);
    }
    tracing::info!(
        target: "victory_gpu::bvh_gt",
        "BVH GT two-point s{submission} (builder s{}): bottom G valid -> reader valid {}, G valid -> reader invalid {}, G invalid -> reader valid {}, G invalid -> reader invalid {} (same class {}); run so far {} / {} / {} / {} (same class {})",
        last.submission,
        pair[0], pair[1], pair[2], pair[3], pair[4],
        total[0], total[1], total[2], total[3], total[4],
    );
}

/// Two-point totals of the run ([`reader_check`]): G valid and reader
/// valid, G valid and reader invalid, G invalid and reader valid, both
/// invalid, both invalid for the same class of reason.
static TWO_POINT: std::sync::Mutex<[u64; 5]> = std::sync::Mutex::new([0; 5]);

/// The class of a validator reason: its words without the numbers and
/// addresses in it ("child box of outside 's", "root (size", ...).
pub(crate) fn why_class(why: Option<&str>) -> String {
    why.unwrap_or("")
        .split_whitespace()
        .filter(|w| !w.chars().any(|c| c.is_ascii_digit()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `VICTORY_BVH_GT_DUMPS` (3): invalid headers per submission whose raw
/// dwords are logged.
fn dumps() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_GT_DUMPS", 3) as usize)
}

/// Submissions compared so far.
static COMPARED: AtomicU64 = AtomicU64::new(0);

/// What a list's bitmap marked, read after its run and before the
/// copy-backs clear it.
#[derive(Default)]
pub(crate) struct Marks {
    /// Pages a copy holds: the copy and the page's title address.
    pub(crate) owned: Vec<(Arc<Shadow>, usize)>,
    /// Pages in the BVH arena (`crate::arena`): their title addresses.
    pub(crate) arena: Vec<usize>,
    pub(crate) missed_pages: u64,
    pub(crate) missed_regions: u64,
    pub(crate) first_missed: Vec<u64>,
}

/// A page compared: where its bytes start, its copy (none for an arena
/// page) and page, G.
struct Page {
    from: usize,
    page: usize,
    copy: Option<Arc<Shadow>>,
    g: Vec<u8>,
}

/// One submission's comparison, from the fence to the log lines.
pub(crate) struct Gt {
    submission: u64,
    lists: usize,
    marked: usize,
    /// Arena pages written, and of them compared.
    arena_marked: usize,
    arena_compared: usize,
    pages: Vec<Page>,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
    list_missed: bool,
    withholding: bool,
    g: Scan,
    /// The second and third validator's counts of G's headers
    /// ([`validate`]).
    v2: V2,
    v3: V3,
    t: [Option<Compare>; 2],
}

impl Gt {
    /// Right after the fence of submission `submission`: G of the pages
    /// the target lists (`lists` of them) marked. None if no list ran the
    /// target or enough submissions were compared.
    pub(crate) fn take(submission: u64, lists: usize, marks: Vec<Marks>) -> Option<Gt> {
        if lists == 0 || COMPARED.fetch_add(1, Ordering::Relaxed) >= most_submissions() {
            return None;
        }
        let (mut owned, mut missed_pages, mut missed_regions, mut first_missed) = (Vec::new(), 0, 0, Vec::new());
        let mut arena = Vec::new();
        for m in marks {
            owned.extend(m.owned);
            arena.extend(m.arena);
            missed_pages += m.missed_pages;
            missed_regions += m.missed_regions;
            first_missed.extend(m.first_missed.into_iter().take(3usize.saturating_sub(first_missed.len())));
        }
        owned.sort_by_key(|(c, page)| (*page, Arc::as_ptr(c) as usize));
        owned.dedup_by_key(|(c, page)| (*page, Arc::as_ptr(c) as usize));
        let marked = owned.len();
        arena.sort_unstable();
        arena.dedup();
        let arena_marked = arena.len();
        // Arena pages first: with the arena on, the builder's structures
        // are there.
        let wanted: Vec<u64> = arena.iter().take(most_pages()).map(|&p| p as u64).collect();
        let mut pages: Vec<Page> = crate::arena::fetch(&wanted)
            .into_iter()
            .map(|(at, g)| Page { from: at as usize, page: at as usize, copy: None, g })
            .collect();
        let arena_compared = pages.len();
        pages.extend(owned.into_iter().take(most_pages().saturating_sub(arena_compared)).filter_map(|(copy, page)| {
            let (from, g) = copy.gpu_bytes(page)?;
            Some(Page { from, page, copy: Some(copy), g })
        }));
        let g = scan(&pages.iter().map(|p| (p.from, &p.g[..])).collect::<Vec<_>>());
        // v3d: (e) bottom levels of earlier submissions built since.
        resolve_pending(&mut builds());
        let built = |top: u64, blas: u64| built_of(&builds(), top, blas);
        let next = |at: u64| builds().map.range(at + 1..).next().map(|(a, _)| *a);
        let (v2, v3) = validate(
            &pages.iter().map(|p| (p.from, &p.g[..])).collect::<Vec<_>>(),
            &g.headers.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
            true,
            &Targets { built: &built, next: &next },
        );
        {
            let mut b = builds();
            b.taken = submission;
            let seq = b.seq;
            for &blas in &v3.never_built {
                if b.pending.len() < 1 << 16 {
                    b.pending.push((blas, seq));
                } else {
                    b.dropped += 1;
                }
            }
        }
        V2_TOTAL.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(V2::default).add(&v2);
        V3_TOTAL.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(V3::default).add(&v3);
        // Remembered for the readers' check ([`reader_check`]): the pages
        // with headers, where they live.
        if reader_target().is_some() && !g.headers.is_empty() {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            last.headers = g.headers.iter().map(|(at, _)| *at).collect();
            last.g = g.headers.iter().map(|(_, c)| (c.valid, why_class(c.why.as_deref()))).collect();
            last.pages = pages.iter().map(|p| (p.copy.clone(), p.page, p.from)).collect();
            last.submission = submission;
        }
        Some(Gt {
            submission,
            lists,
            marked,
            arena_marked,
            arena_compared,
            pages,
            missed_pages,
            missed_regions,
            first_missed,
            list_missed: false,
            withholding: false,
            g,
            v2,
            v3,
            t: [None, None],
        })
    }

    /// Whether a target list missed memory, and whether writes were held
    /// back in this submission (`crate::memory::withholding`).
    pub(crate) fn decided(&mut self, list_missed: bool, withholding: bool) {
        self.list_missed = list_missed;
        self.withholding = withholding;
    }

    /// T number `k` (0 after the copy-backs, 1 after the bottom-of-pipe
    /// writes): title memory at the compared pages.
    pub(crate) fn read_title(&mut self, k: usize) {
        let mut c = Compare::default();
        let mut t_pages: Vec<(usize, Vec<u8>)> = Vec::with_capacity(self.pages.len());
        for p in &self.pages {
            // An arena page goes nowhere by design: only its headers count.
            if p.copy.is_none() {
                c.arena += 1;
            }
            let mut t = vec![0u8; p.g.len()];
            if !crate::host::read_into(p.from as *const u8, &mut t) {
                c.unreadable += 1;
                continue;
            }
            let mut differs = false;
            if p.copy.is_none() {
                t_pages.push((p.from, t));
                continue;
            }
            for (i, (g, t)) in p.g.as_chunks::<4>().0.iter().zip(t.as_chunks::<4>().0).enumerate() {
                if g != t {
                    differs = true;
                    c.dwords += 1;
                    if c.first.len() < 3 {
                        let word = |b: &[u8; 4]| u32::from_le_bytes(*b);
                        c.first.push((p.from as u64 + 4 * i as u64, word(g), word(t)));
                    }
                }
            }
            if differs {
                c.differing += 1;
                if p.copy.as_ref().and_then(|c| c.title_changed(p.page)) == Some(true) {
                    c.title_changed += 1;
                }
            } else {
                c.equal += 1;
            }
            t_pages.push((p.from, t));
        }
        let scanned = scan(&t_pages.iter().map(|(f, t)| (*f, &t[..])).collect::<Vec<_>>());
        // The headers G has, checked in T.
        let read = reader(&t_pages);
        c.headers_valid = self.g.headers.iter().filter(|(at, _)| check(&read, *at).valid).count();
        c.scan = Some(scanned);
        self.t[k] = Some(c);
    }

    /// The case the plan names (bvh-test-plan (2), (3)) from what was seen.
    fn case(&self) -> &'static str {
        let invalid = self.g.headers.iter().filter(|(_, c)| !c.valid).count();
        let missed = self.missed_pages + self.missed_regions > 0;
        let held = self.list_missed || self.withholding;
        let t_differs = self.t[0].as_ref().is_some_and(|t| t.differing > 0);
        if self.g.headers.is_empty() {
            return match (t_differs, held) {
                (true, true) => "no-header:T!=G,held",
                (true, false) => "no-header:T!=G,not-held",
                (false, _) => "no-header:T=G",
            };
        }
        match (invalid > 0, missed, t_differs, held) {
            (true, true, _, _) => "A1",
            (true, false, _, _) => "A2",
            (false, _, true, true) => "B",
            (false, _, true, false) => "T!=G,not-held",
            (false, _, false, _) if self.arena_compared > 0 => "G valid,arena",
            (false, _, false, _) => "G=T,valid",
        }
    }

    /// The comparison's log lines.
    pub(crate) fn log(&self) {
        let hash = target().unwrap_or("");
        let s = self.submission;
        let addrs = |v: &[u64]| v.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" ");
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} {hash}: case {}; lists {}, pages written with a copy {} (compared {}), in the arena {} (compared {}), missed {} pages + {} regions (first: {}); list missed {}, withholding {}",
            self.case(),
            self.lists,
            self.marked,
            self.pages.len() - self.arena_compared,
            self.arena_marked,
            self.arena_compared,
            self.missed_pages,
            self.missed_regions,
            addrs(&self.first_missed),
            self.list_missed,
            self.withholding,
        );
        let invalid: Vec<String> = self
            .g
            .headers
            .iter()
            .filter(|(_, c)| !c.valid)
            .take(3)
            .map(|(at, c)| format!("{at:#x} {}", c.why.as_deref().unwrap_or("?")))
            .collect();
        // Raw dwords of the first invalid headers, to check the validator's
        // reading of the format against what the builder wrote: the header,
        // the node the root pointer names, the node one before it, and the
        // node at dword 10 (offset of the box nodes).
        let g_pages: Vec<(usize, &[u8])> = self.pages.iter().map(|p| (p.from, &p.g[..])).collect();
        let read = reader(&g_pages);
        // Headers whose offsets (dwords 10, 13, 14) look like a structure's:
        // 64-byte multiples of box nodes and elements inside the size.
        let plausible = |at: u64| {
            let w = |i: u64| read(at + 4 * i).unwrap_or(u32::MAX);
            let bytes = u64::from(w(1)) * 64;
            let (d10, d13) = (u64::from(w(10)), u64::from(w(13)));
            d10 >= 64 && d10.is_multiple_of(64) && d10 <= bytes + 64 && d13.is_multiple_of(64) && d13 <= bytes + 64
        };
        let plausible_n = self.g.headers.iter().filter(|(at, _)| plausible(*at)).count();
        let plausible_valid = self.g.headers.iter().filter(|(at, c)| c.valid && plausible(*at)).count();
        let valid_past_header = self.g.headers.iter().filter(|(at, _)| check_with(&read, *at, 64).valid).count();
        let mut by_kind: std::collections::BTreeMap<&str, (u32, u32)> = std::collections::BTreeMap::new();
        for (at, c) in &self.g.headers {
            let start = (4..10).any(|i| read(at + 4 * i).is_some_and(|w| f32::from_bits(w).abs() == f32::MAX));
            let e = by_kind.entry(kind(read(*at).unwrap_or(0), start)).or_default();
            e.0 += 1;
            e.1 += u32::from(c.valid);
        }
        let kinds: Vec<String> = by_kind.iter().map(|(k, (n, v))| format!("{k} {v}/{n}")).collect();
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} format: headers {} plausible (dwords 10/13 64-byte multiples inside the size) {plausible_n}, of them valid {plausible_valid}; valid if the size counts the nodes past the header {valid_past_header}; valid by kind {}",
            self.g.headers.len(),
            kinds.join(", "),
        );
        if self.g.headers.iter().any(|(_, c)| !c.valid) {
            let words = |at: u64, n: u64| -> String {
                (0..n)
                    .map(|i| read(at + 4 * i).map_or_else(|| "-".to_string(), |w| format!("{w:08x}")))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            // Plausible ones first: the others are mostly other data that
            // happens to start like a header.
            let mut bad: Vec<&(u64, Check)> = self.g.headers.iter().filter(|(_, c)| !c.valid).collect();
            bad.sort_by_key(|(at, _)| !plausible(*at));
            for (at, _) in bad.into_iter().take(dumps()) {
                let at = *at;
                let root = read(at + 48).unwrap_or(0);
                let node = u64::from(root & !7) << 3;
                let d10 = u64::from(read(at + 40).unwrap_or(0));
                tracing::info!(
                    target: "victory_gpu::bvh_gt",
                    "BVH GT s{s} raw {at:#x}: header [{}]; root {root:#x} at +{node:#x} [{}]; at +{:#x} [{}]; dword 10 +{d10:#x} [{}]",
                    words(at, 16),
                    words(at + node, 32),
                    node.saturating_sub(64),
                    words(at + node.saturating_sub(64), 16),
                    words(at + d10, 16),
                );
            }
        }
        let valid = self.g.headers.iter().filter(|(_, c)| c.valid).count();
        let (boxes, outside, missing, parents) = self.g.headers.iter().fold((0, 0, 0, (0, 0, 0)), |a, (_, c)| {
            (a.0 + c.boxes, a.1 + c.outside, a.2 + c.missing, (a.3.0 + c.parents.0, a.3.1 + c.parents.1, a.3.2 + c.parents.2))
        });
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} G: headers {} valid {valid} (first invalid: {}); box nodes {boxes}, child boxes outside {outside}, nodes not compared {missing}, parent table checked {} fwd {} bwd {}; {}",
            self.g.headers.len(),
            invalid.join(", "),
            parents.0,
            parents.1,
            parents.2,
            self.g.patterns(),
        );
        let total = V2_TOTAL.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} v2: {}; first invalid: {}; cb dispatches with other values for a bottom level seen before {}",
            self.v2.text(),
            self.v2.first.join(", "),
            CB_DIFFER.load(Ordering::Relaxed),
        );
        tracing::info!(target: "victory_gpu::bvh_gt", "BVH GT s{s} v2 run so far: {}", total.text());
        // v3 (senior 100): "not judged" apart from valid and invalid. Per
        // submission is the number to read; "run so far" sums a structure
        // again in every submission that compares it.
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} v3: {}; first invalid: {}; first not judged: {}; v3d first: {}",
            self.v3.text(),
            self.v3.first.join(", "),
            self.v3.first_not_judged.join(", "),
            self.v3.first_v3d.join(", "),
        );
        let total3 = V3_TOTAL.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
        tracing::info!(target: "victory_gpu::bvh_gt", "BVH GT s{s} v3 run so far: {}", total3.text());
        let b = builds();
        tracing::info!(
            target: "victory_gpu::bvh_gt",
            "BVH GT s{s} v3d run so far: (e) instance checks whose bottom level was built later {}, still never built {} (not kept {}); build targets known {} (map full, binds not noted {}), binds noted {}",
            b.later,
            b.pending.len(),
            b.dropped,
            b.map.len(),
            b.full,
            b.seq,
        );
        for (k, t) in self.t.iter().enumerate() {
            let Some(t) = t else { continue };
            let first: Vec<String> = t.first.iter().map(|(a, g, t)| format!("{a:#x} G={g:#010x} T={t:#010x}")).collect();
            tracing::info!(
                target: "victory_gpu::bvh_gt",
                "BVH GT s{s} T{}: pages T=G {} T!=G {} (title wrote/retaken {}), arena pages (not compared, stay on the card) {}, unreadable {}, dwords differing {} (first: {}); G headers valid in T {}/{}; {}",
                k + 1,
                t.equal,
                t.differing,
                t.title_changed,
                t.arena,
                t.unreadable,
                t.dwords,
                first.join(", "),
                t.headers_valid,
                self.g.headers.len(),
                t.scan.as_ref().map(Scan::patterns).unwrap_or_default(),
            );
        }
    }
}

/// T against G.
#[derive(Default)]
struct Compare {
    equal: usize,
    differing: usize,
    title_changed: usize,
    arena: usize,
    unreadable: usize,
    dwords: u64,
    first: Vec<(u64, u32, u32)>,
    headers_valid: usize,
    scan: Option<Scan>,
}

/// Headers found in pages, and values the title must never read as
/// pointers: empty box16 boxes and +-FLT_MAX pairs (counts, first three).
#[derive(Default)]
pub(crate) struct Scan {
    pub(crate) headers: Vec<(u64, Check)>,
    pub(crate) empty16: (u64, Vec<u64>),
    pub(crate) flt_max: (u64, Vec<u64>),
}

impl Scan {
    fn patterns(&self) -> String {
        let addrs = |v: &[u64]| v.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" ");
        format!(
            "empty box16 boxes {} (first: {}), +-FLT_MAX pairs {} (first: {})",
            self.empty16.0,
            addrs(&self.empty16.1),
            self.flt_max.0,
            addrs(&self.flt_max.1)
        )
    }
}

/// Dwords at title addresses from `pages` (start, bytes).
pub(crate) fn reader<'a, B: AsRef<[u8]>>(pages: &'a [(usize, B)]) -> impl Fn(u64) -> Option<u32> + 'a {
    move |at: u64| {
        let at = at as usize;
        pages.iter().find_map(|(from, b)| {
            let b = b.as_ref();
            let i = at.checked_sub(*from)?;
            let w = b.get(i..i + 4)?;
            Some(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        })
    }
}

/// Scans `pages` (start, bytes): headers at 256-byte aligned addresses,
/// each checked, and the patterns.
pub(crate) fn scan(pages: &[(usize, &[u8])]) -> Scan {
    let mut s = Scan::default();
    let read = reader(pages);
    let note = |n: &mut (u64, Vec<u64>), at: u64| {
        n.0 += 1;
        if n.1.len() < 3 {
            n.1.push(at);
        }
    };
    for &(from, bytes) in pages {
        let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        for (i, pair) in words.windows(2).enumerate() {
            let at = (from + 4 * i) as u64;
            if pair == [0x7c00_7c00, 0xfc00_7c00] {
                note(&mut s.empty16, at);
            }
            let max = |w: u32| w == 0x7f7f_ffff || w == 0xff7f_ffff;
            if at.is_multiple_of(8) && max(pair[0]) && max(pair[1]) {
                note(&mut s.flt_max, at);
            }
            if at.is_multiple_of(256) && is_header(pair[0], pair[1]) {
                s.headers.push((at, check(&read, at)));
            }
        }
    }
    s
}

/// Marks in a header's dword 0 above the version: bottom level, compacted,
/// rehydrated, still packed.
const MARKS: u32 = 0x8000 | 0x4 | 0x2 | 0x100;

/// Whether dwords 0 and 1 look like a structure's header: version 6 with
/// known marks in the high half, a size of 1 to 2^22 nodes of 64 bytes.
pub(crate) fn is_header(d0: u32, d1: u32) -> bool {
    d0 & 0xffff == 6 && (d0 >> 16) & !MARKS == 0 && (1..=1 << 22).contains(&d1)
}

/// A structure's check.
#[derive(Default, Debug)]
pub(crate) struct Check {
    pub(crate) valid: bool,
    /// The first reason it is not.
    pub(crate) why: Option<String>,
    pub(crate) boxes: u32,
    /// Child boxes outside their parent's box, children past the size, and
    /// empty boxes with a child.
    pub(crate) outside: u32,
    /// Of those outside their parent's box, the ones inside it once the
    /// parent's box takes in the origin.
    pub(crate) origin: u32,
    /// Nodes not in what was read.
    pub(crate) missing: u32,
    /// Parent table: entries checked, matching as `table[node]` (forward)
    /// and as `table[-1 - node]` (backward); its form is not known.
    pub(crate) parents: (u32, u32, u32),
}

type Aabb = ([f32; 3], [f32; 3]);

fn half(h: u32) -> f32 {
    let h = h & 0xffff;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 if man == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

/// Whether `child` lies in `parent`, with room for f16 rounding: a box16
/// bound is rounded outwards by up to an f16 step at its magnitude (2^-10
/// of it), which far from the origin is much more than a share of a small
/// parent's extent.
fn inside(child: &Aabb, parent: &Aabb) -> bool {
    (0..3).all(|a| {
        let magnitude = parent.0[a].abs().max(parent.1[a].abs());
        let tol = (parent.1[a] - parent.0[a]).abs() / 256.0 + magnitude / 512.0 + 1e-3;
        child.0[a] >= parent.0[a] - tol && child.1[a] <= parent.1[a] + tol
    })
}

fn empty(b: &Aabb) -> bool {
    // NaN counts as empty.
    (0..3).any(|a| b.0[a].partial_cmp(&b.1[a]).is_none_or(|o| o == std::cmp::Ordering::Greater))
}

/// Checks the structure whose header is at `base` (bvh-test-plan (1)): the
/// version and size, the root pointer at +48 (type and address inside the
/// size), the header's bounds (finite, min <= max, not the builder's
/// +-FLT_MAX start values), and from the root down every box node: child
/// boxes in the parent's box, children inside the size, no empty box with
/// a child, the root not empty while the header counts elements; the
/// parent table (dword 14) is checked both ways and only counted.
pub(crate) fn check(read: &dyn Fn(u64) -> Option<u32>, base: u64) -> Check {
    // Built bottom levels put the root at exactly dword 1 x 64 with a box
    // node there that fits the header's bounds (runs 164431, 171123: e.g.
    // root 0x3064 at +0x18300, size 0x60c nodes): the size leaves out one
    // node (the header's or the root's), so one more node counts as inside.
    check_with(read, base, 64)
}

/// [`check`] with `extra` bytes added to the size dword 1 gives: 64 reads
/// the size as the nodes past the header (an open question of the format).
pub(crate) fn check_with(read: &dyn Fn(u64) -> Option<u32>, base: u64, extra: u64) -> Check {
    check_core(read, base, extra, None, false)
}

/// The second validator (senior 96, `BVH GT s.. v2` lines; [`check`] stays
/// as it was so that its numbers compare with earlier runs): still packed
/// bottom levels get their own check ([`check_packed`], the logic of
/// rt/packed-validator e56de5c), and node boxes of a structure with a scale
/// word (rehydrated ones) are compared in their stored units, around
/// `centre` ([`Centre`]).
pub(crate) fn check_v2(read: &dyn Fn(u64) -> Option<u32>, base: u64, centre: Centre) -> Check {
    check_core(read, base, 64, Some(centre), false)
}

/// [`check_with`] (`v2` none: as before) or [`check_v2`]; with `node_end`
/// (v3b, senior 105) a box node must also end by D1 * 64 + 64, not only
/// start inside the size (box16 64 bytes, box32 112). v3c (senior 109):
/// the end limit is D1 * 64 + 64 for every kind, also where nodes must
/// start inside D1 * 64 exactly (0x8002 / 0x8004): fresh bottom levels put
/// a box32 root at D1 * 64 - 64 that ends 48 bytes past D1 * 64 (run V105
/// 20261127-164739: 115/115 "ends past the size" there, 5/5 raw 0x8004
/// headers; same shape as a top level's box32 root, senior 100). Measured,
/// unverified, for Fable.
fn check_core(read: &dyn Fn(u64) -> Option<u32>, base: u64, extra: u64, v2: Option<Centre>, node_end: bool) -> Check {
    let mut c = Check::default();
    let fail = |c: &mut Check, why: String| {
        if c.why.is_none() {
            c.why = Some(why);
        }
    };
    let Some(h) = (0..16).map(|i| read(base + 4 * i)).collect::<Option<Vec<u32>>>() else {
        c.why = Some("header not read".into());
        return c;
    };
    if !is_header(h[0], h[1]) {
        fail(&mut c, format!("dword 0/1 {:#x}/{}", h[0], h[1]));
    } else if v2.is_some() && (h[0] >> 16) & PACKED != 0 {
        return check_packed(read, base, &h);
    }
    let bytes = u64::from(h[1]) * 64 + extra;
    let node_limit = u64::from(h[1]) * 64 + 64;
    let root = h[12];
    let at = |p: u32| u64::from(p & !7) << 3;
    if root == u32::MAX || root & 7 == 7 || at(root) == 0 || at(root) >= bytes {
        fail(&mut c, format!("root +48 {root:#x} (size {bytes:#x})"));
    }
    let f = |i: usize| f32::from_bits(h[i]);
    let bounds: Aabb = ([f(4), f(5), f(6)], [f(7), f(8), f(9)]);
    if (4..10).any(|i| f(i).abs() == f32::MAX) {
        fail(&mut c, "bounds +-FLT_MAX (start values)".into());
    } else if (4..10).any(|i| !f(i).is_finite()) || empty(&bounds) {
        fail(&mut c, format!("bounds {bounds:?}"));
    }
    let parents = h[14] as i32 as i64;
    // v2: node boxes of a structure with a scale word (rehydrated ones) are
    // in stored units, the header's bounds in object units.
    let start = match v2 {
        Some(centre) => to_stored(h[15], &bounds, centre),
        None => bounds,
    };
    let mut queue = vec![(root, start, true)];
    let mut seen = std::collections::HashSet::new();
    while let Some((p, bounds, is_root)) = queue.pop() {
        let kind = p & 7;
        if !(kind == 4 || kind == 5) || at(p) == 0 || at(p) >= bytes || !seen.insert(p) || seen.len() > 4096 {
            continue;
        }
        let n = if kind == 4 { 16 } else { 28 };
        if node_end && at(p) + 4 * n > node_limit {
            c.outside += 1;
            fail(&mut c, format!("box node {p:#x} ends past the size {node_limit:#x} (D1 * 64 + 64)"));
            continue;
        }
        let Some(w) = (0..n).map(|i| read(base + at(p) + 4 * i)).collect::<Option<Vec<u32>>>() else {
            c.missing += 1;
            continue;
        };
        c.boxes += 1;
        let mut children = 0;
        for i in 0..4 {
            let child = w[i];
            let b: Aabb = if kind == 4 {
                let (a, b, d) = (w[4 + 3 * i], w[5 + 3 * i], w[6 + 3 * i]);
                ([half(a), half(a >> 16), half(b)], [half(b >> 16), half(d), half(d >> 16)])
            } else {
                let g = |k: usize| f32::from_bits(w[4 + 6 * i + k]);
                ([g(0), g(1), g(2)], [g(3), g(4), g(5)])
            };
            if child == u32::MAX {
                continue;
            }
            children += 1;
            if at(child) >= bytes {
                c.outside += 1;
                fail(&mut c, format!("child {child:#x} of {p:#x} past the size"));
                continue;
            }
            if empty(&b) {
                c.outside += 1;
                fail(&mut c, format!("empty box with child {child:#x} in {p:#x}"));
                continue;
            }
            if !inside(&b, &bounds) {
                c.outside += 1;
                // Outside only because it reaches the origin: a box that took
                // in a point read as zeros (runs 180153-182804).
                let with_origin: Aabb = (
                    [bounds.0[0].min(0.0), bounds.0[1].min(0.0), bounds.0[2].min(0.0)],
                    [bounds.1[0].max(0.0), bounds.1[1].max(0.0), bounds.1[2].max(0.0)],
                );
                if inside(&b, &with_origin) {
                    c.origin += 1;
                }
                fail(&mut c, format!("child box of {child:#x} outside {p:#x}'s"));
            }
            let index = i64::from(child & !7) >> 3;
            let entry = |k: i64| read((base as i64 + parents + 4 * k) as u64);
            c.parents.0 += 1;
            c.parents.1 += u32::from(entry(index) == Some(p));
            c.parents.2 += u32::from(entry(-1 - index) == Some(p));
            queue.push((child, b, false));
        }
        if is_root && children == 0 && h[3] > 0 {
            fail(&mut c, format!("root node empty with {} elements", h[3]));
        }
    }
    c.valid = c.why.is_none();
    c
}

/// Header dword 0 mark (high half): the structure is still packed.
const PACKED: u32 = 0x100;

/// How the centre of a structure with a scale word is taken ([`to_stored`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Centre {
    /// As rt/packed-validator e56de5c (senior 81): with bit 30 every axis
    /// around 0.5 (min + max) of the header bounds, without it around 0.
    /// "Without the bit-30 rule" in the v2 lines.
    Bit30AllAxes,
    /// Per axis the given centre in object units, with or without bit 30
    /// (Fable Aufruf 5 point 9: cb0..2 of the 5e687674 dispatch of this
    /// structure, the title's values, also 0; or, where no such dispatch was
    /// seen, [`one_sided_centre`]). "With the rule" in the v2 lines.
    Given([f32; 3]),
}

/// The header bounds in the units the nodes are stored in. Dword 15 with
/// bit 31 set is a scale word (packed and rehydrated bottom levels, e.g.
/// 0x808e8e8d): axis a is stored in units of 2^(e_a - 127), e_a = byte a
/// (x = byte 0), relative to the centre `centre` names. Without bit 31
/// (built ones, 0x007f7f7f) the units are the same. Read from capture data
/// (senior 81, 85, 93), unverified.
fn to_stored(d15: u32, b: &Aabb, centre: Centre) -> Aabb {
    if d15 & 0x8000_0000 == 0 {
        return *b;
    }
    let mut out = *b;
    for a in 0..3 {
        let unit = 2f32.powi(127 - ((d15 >> (8 * a)) & 0xff) as i32);
        let c = match centre {
            Centre::Bit30AllAxes if d15 & 0x4000_0000 != 0 => 0.5 * (b.0[a] + b.1[a]),
            Centre::Bit30AllAxes => 0.0,
            Centre::Given(c) => c[a],
        };
        out.0[a] = (b.0[a] - c) / unit;
        out.1[a] = (b.1[a] - c) / unit;
    }
    out
}

/// The centre when no 5e687674 dispatch of the structure was seen (senior
/// 85, header one-sided): per axis 0.5 (min + max) where the header box lies
/// wholly on one side of 0 (min > 0 or max < 0), else 0. In the captures
/// cb_a != 0 exactly on such axes (54/54); unverified, for Fable.
pub(crate) fn one_sided_centre(h: &[u32]) -> [f32; 3] {
    let f = |i: usize| f32::from_bits(h[i]);
    std::array::from_fn(|a| {
        let (lo, hi) = (f(4 + a), f(7 + a));
        if lo > 0.0 || hi < 0.0 { 0.5 * (lo + hi) } else { 0.0 }
    })
}

/// A packed node word: its children (bits 24..31, 1 to 4) and the mask of
/// the box components stored as f32 after its group's pointers (bits 0..23,
/// bit 6 x child + component; components min x, y, z, max x, y, z).
fn node_word(w: u32) -> Option<(usize, u32)> {
    let children = (w >> 24) as usize;
    (1..=4).contains(&children).then_some((children, w & 0xff_ffff))
}

/// A group of packed box nodes: where it starts and each node's children.
struct Group {
    at: u64,
    children: Vec<Vec<u32>>,
}

/// The box nodes of a packed structure ([`check_packed`]), group by group.
fn packed_groups(read: &dyn Fn(u64) -> Option<u32>, base: u64, h: &[u32]) -> Result<Vec<Group>, String> {
    let size = u64::from(h[1]) * 64;
    let word = |o: u64| read(base + o).ok_or_else(|| format!("node word +{o:#x} not read"));
    let mut groups = Vec::new();
    let mut s = u64::from(h[10]);
    while s < size {
        let w0 = word(s)?;
        if w0 == 0 {
            // Padding (a box32 node's second half, a group's unused slots),
            // or the trailer some large structures end with (e.g. 0x1080,
            // or 0x4b00 0x5100 0x6600 in the last words): nothing past it is
            // parsed, nodes there count as missing in the walk.
            let mut empty = true;
            for i in 0..16 {
                empty &= word(s + 4 * i)? == 0;
            }
            if !empty {
                break;
            }
            s += 64;
            continue;
        }
        let mut words = Vec::new();
        while let Some(w) = node_word(word(s + 4 * words.len() as u64)?) {
            words.push(w);
        }
        if words.is_empty() {
            return Err(format!("slot +{s:#x} word 0 {w0:#x} not a node word"));
        }
        let mut o = s + 4 * words.len() as u64;
        let mut children = Vec::with_capacity(words.len());
        for &(n, _) in &words {
            let mut node = Vec::with_capacity(n);
            for _ in 0..n {
                node.push(word(o)?);
                o += 4;
            }
            children.push(node);
        }
        o += 4 * words.iter().map(|&(_, mask)| u64::from(mask.count_ones())).sum::<u64>();
        if o > size {
            return Err(format!("group +{s:#x} past the size"));
        }
        let next = (o + 63) & !63;
        while o < next {
            if word(o)? != 0 {
                return Err(format!("group +{s:#x} tail not zero at +{o:#x}"));
            }
            o += 4;
        }
        groups.push(Group { at: s, children });
        s = next;
    }
    Ok(groups)
}

/// From the root down through packed groups: every box node reached once,
/// leaves (types 0-3) in the leaf area and each reached once. A group's
/// stride is 64 bytes for box16 nodes (pointer type 4) and 128 for box32
/// (type 5), known only from the pointers. Box nodes and leaves reached.
fn packed_walk(groups: &[Group], root: u32, h: &[u32]) -> Result<(u32, u32), String> {
    let size = u64::from(h[1]) * 64;
    let (boxes, leaves_at) = (u64::from(h[10]), u64::from(h[13]));
    let at = |p: u32| u64::from(p & !7) << 3;
    let mut stride = vec![0u64; groups.len()];
    let mut seen = std::collections::HashSet::new();
    let mut leaves = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        let (a, step) = (at(p), if p & 7 == 4 { 64 } else { 128 });
        let node = groups
            .partition_point(|g| g.at <= a)
            .checked_sub(1)
            .map(|g| (g, a - groups[g].at))
            .filter(|&(g, off)| off % step == 0 && ((off / step) as usize) < groups[g].children.len());
        let Some((g, off)) = node else {
            return Err(format!("box pointer {p:#x} to no node"));
        };
        if stride[g] != 0 && stride[g] != step {
            return Err(format!("group +{:#x} reached as box16 and box32", groups[g].at));
        }
        stride[g] = step;
        if !seen.insert(a) {
            return Err(format!("node +{a:#x} reached twice"));
        }
        for &child in &groups[g].children[(off / step) as usize] {
            match child & 7 {
                4 | 5 if (boxes..size).contains(&at(child)) => stack.push(child),
                0..=3 if (leaves_at..boxes).contains(&at(child)) => {
                    if !leaves.insert(at(child)) {
                        return Err(format!("leaf {child:#x} reached twice"));
                    }
                }
                _ => return Err(format!("child {child:#x} of +{a:#x} outside its area")),
            }
        }
    }
    let nodes: usize = groups.iter().map(|g| g.children.len()).sum();
    if seen.len() != nodes {
        return Err(format!("{} of {nodes} box nodes reached", seen.len()));
    }
    for (g, group) in groups.iter().enumerate() {
        let end = groups.get(g + 1).map_or(size, |next| next.at);
        if group.at + group.children.len() as u64 * stride[g] > end {
            return Err(format!("group +{:#x} of {} nodes overlaps the next", group.at, group.children.len()));
        }
    }
    Ok((seen.len() as u32, leaves.len() as u32))
}

/// Checks a still packed bottom level (mark 0x100, e.g. 0x8102_0006; format
/// read from capture data by senior 81, unverified, see
/// bvh-harness-design-fable.md "Gepacktes BLAS-Format"). Its box nodes hold
/// no boxes, only the tree: from dword 10 on, groups of n nodes stored as n
/// node words ([`node_word`]), then all their child pointers in node order,
/// then the f32 box components the masks name, zeros up to the next 64-byte
/// slot. The leaves [dword 13, dword 10) are a compressed stream (vertices
/// come from the geometry at rehydration) and are not decoded. Checked: the
/// header bounds (finite, not the start values), the offsets, the scale
/// word (dword 15), and the tree: every node reached once, every leaf once,
/// as many leaves as dword 3 counts. Words not read make it not valid.
pub(crate) fn check_packed(read: &dyn Fn(u64) -> Option<u32>, base: u64, h: &[u32]) -> Check {
    let mut c = Check::default();
    let why = (|| {
        let size = u64::from(h[1]) * 64;
        let (boxes, leaves, elements) = (u64::from(h[10]), u64::from(h[13]), u64::from(h[3]));
        let f = |i: usize| f32::from_bits(h[i]);
        let bounds: Aabb = ([f(4), f(5), f(6)], [f(7), f(8), f(9)]);
        if h[0] >> 16 & 0x8000 == 0 {
            return Err(format!("dword 0 {:#x}: packed but not bottom level", h[0]));
        }
        if (4..10).any(|i| f(i).abs() == f32::MAX) {
            return Err("bounds +-FLT_MAX (start values)".into());
        }
        if (4..10).any(|i| !f(i).is_finite()) || empty(&bounds) {
            return Err(format!("bounds {bounds:?}"));
        }
        if !(64 <= leaves && leaves < boxes && boxes < size && leaves % 64 == 0 && boxes % 64 == 0) {
            return Err(format!("offsets leaves {leaves:#x} boxes {boxes:#x} (size {size:#x})"));
        }
        if leaves + 64 * elements > boxes {
            return Err(format!("{elements} leaves past the box nodes"));
        }
        if h[15] & 0x8000_0000 == 0 || h[15] & 0x3f00_0000 != 0 {
            return Err(format!("dword 15 {:#x} not a scale word", h[15]));
        }
        let root = h[12];
        let at = u64::from(root & !7) << 3;
        if !(root & 7 == 4 || root & 7 == 5) || !(boxes..size).contains(&at) {
            return Err(format!("root +48 {root:#x} (size {size:#x})"));
        }
        let groups = packed_groups(read, base, h).inspect_err(|_| c.missing += 1)?;
        let (nodes, reached) = packed_walk(&groups, root, h)?;
        c.boxes = nodes;
        if u64::from(reached) != elements {
            return Err(format!("{reached} leaves reached, dword 3 counts {elements}"));
        }
        Ok(())
    })();
    c.why = why.err();
    c.valid = c.why.is_none();
    c
}

/// What a bottom-level header's dword 0 marks it as, for the v2 counts:
/// still packed (0x100), rehydrated (0x8002), built fresh (0x8004), or
/// another bottom level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Bottom {
    Packed,
    Rehydrated,
    Fresh,
    Other,
}

/// [`Bottom`] of a bottom-level header's dword 0; none for a top level.
pub(crate) fn bottom_kind(d0: u32) -> Option<Bottom> {
    let marks = d0 >> 16;
    Some(if marks & PACKED != 0 {
        Bottom::Packed
    } else if marks & 0x8000 == 0 {
        return None;
    } else if marks & 0x2 != 0 {
        Bottom::Rehydrated
    } else if marks & 0x4 != 0 {
        Bottom::Fresh
    } else {
        Bottom::Other
    })
}

/// A top-level structure's v2 check ([`check_top`]).
#[derive(Default, Debug, Clone)]
pub(crate) struct TopCheck {
    pub(crate) valid: bool,
    /// The first reason it is not.
    pub(crate) why: Option<String>,
    /// Instances by the header (dword 3).
    pub(crate) instances: u32,
    /// Instances reached from the root at least once, never, more than once.
    pub(crate) reached: u32,
    pub(crate) never: u32,
    pub(crate) twice: u32,
    /// Slots whose pointer is neither empty (0xffffffff), a box node inside
    /// the size nor an instance of the list; box nodes reached twice.
    pub(crate) bad_slots: u32,
    pub(crate) nodes_twice: u32,
    /// Box nodes or instances not in what was read.
    pub(crate) missing: u32,
    /// Reached instances by the kind of their bottom level, and those whose
    /// bottom-level header was not read or is no header.
    pub(crate) on: [u32; 4],
    pub(crate) blas_unread: u32,
    /// Of the reached instances on rehydrated (index 1) and fresh (2) bottom
    /// levels, those whose box in their parent's slot is finite (not
    /// +-FLT_MAX, min <= max).
    pub(crate) finite: [u32; 4],
    /// The header bounds still hold +-FLT_MAX (instances on packed bottom
    /// levels keep the builder's start values: 190fbf0e leaves them off).
    pub(crate) bounds_start: bool,
    /// v3: of [`TopCheck::blas_unread`], those whose bottom-level page was
    /// read but holds no bottom-level header there.
    pub(crate) blas_no_header: u32,
    /// v3: the first reason that is a defect in what was read (a bad slot, a
    /// box node or instance twice, the list or root outside the size, a
    /// bottom level that is no header, a rehydrated instance's open box),
    /// and the first thing not read (box node, instance, bottom level).
    pub(crate) first_defect: Option<String>,
    pub(crate) first_unread: Option<String>,
    /// v3b (senior 105): whether a word that decides [`TopCheck::first_defect`]
    /// came from title memory (the defect word itself, the pointers and
    /// header words that led to it, the bottom-level header for "no header"
    /// or an open box); whether any word of the tree (header, box nodes,
    /// instance entries' dword 15) did; reached instances whose
    /// bottom-level header (read for its kind only) did; whether any word
    /// read did ([`top_v3_over`] sets it); and, of a verdict that
    /// [`top_v3_over`] turned into "not judged", its index before
    /// ([`Verdict::index`]).
    pub(crate) defect_from_title: bool,
    pub(crate) tree_from_title: bool,
    pub(crate) kind_from_title: u32,
    pub(crate) any_title: bool,
    pub(crate) title_before: Option<usize>,
    /// v3d (senior 112, only with a build map): reached instances by class
    /// [a, b, c, d, e] (module doc); those whose search for a header past
    /// the parent table hit a word not read (not judged, in
    /// [`TopCheck::blas_unread`]); the first (c) or (e) instance, which
    /// leaves the top level not judged; the (e) bottom levels (at most
    /// 4096); the first (b) instance and its header's offset.
    pub(crate) classes: [u32; 5],
    pub(crate) search_unread: u32,
    pub(crate) first_order: Option<String>,
    pub(crate) never_built: Vec<u64>,
    pub(crate) first_b: Option<String>,
}

/// Index of `b` in [`TopCheck::on`] and [`TopCheck::finite`].
fn bottom_index(b: Bottom) -> usize {
    match b {
        Bottom::Packed => 0,
        Bottom::Rehydrated => 1,
        Bottom::Fresh => 2,
        Bottom::Other => 3,
    }
}

/// Checks the top-level structure whose header is at `base` (Fable Aufruf
/// 6 point 4; the method of senior 90's reach90/node90): from the root
/// (dword 12) down every box node (pointer types 4, 5), each instance slot
/// (type 6) naming an entry of the instance list (dword 13 on, 128 bytes
/// each, dword 3 of them); every instance reached exactly once, no other
/// pointer, no box node twice. Each reached instance's bottom level (entry
/// dword 15 << 8) is classed by its header ([`bottom_kind`]); only those on
/// rehydrated bottom levels (0x8002) must have a finite box in their slot,
/// instances on still packed ones keep +-FLT_MAX (190fbf0e leaves them
/// off, senior 85), and the header bounds may then hold +-FLT_MAX too.
/// Child boxes are not compared with their parent's (that is [`check`]'s,
/// which counts the packed instances' +-FLT_MAX as outside).
pub(crate) fn check_top(read: &dyn Fn(u64) -> Option<u32>, base: u64) -> TopCheck {
    check_top_core(read, base, 64, None, None)
}

/// [`check_top`] with the instance list ending by D1 * 64 + `list_extra`
/// (v2: 64; v3: 0, the list lies within D1 * 64). Box nodes may end up to
/// D1 * 64 + 64 in both: on all 8 top levels of the s89-1 harness dumps
/// (senior 92, 100) the root is the last node and starts at D1 * 64 - 64
/// (box32) or at D1 * 64 (box16), so a top level's D1 does not count its
/// header, unlike a bottom level's (spec point 1). Measured, unverified, for
/// Fable.
///
/// With `title` (v3b, senior 105: which words came from title memory) a box
/// node must also end by D1 * 64 + 64 (box16 64 bytes, box32 112), and the
/// first defect, the tree and the bottom levels' kinds note whether title
/// words decided them ([`TopCheck::defect_from_title`]). v2 passes none
/// and is unchanged.
///
/// With `built` (v3d, senior 112: [`Built`] of a bottom level against this
/// top level) each reached instance is classed (module doc,
/// [`TopCheck::classes`]): only (d) is a defect, (c) and (e) are noted in
/// [`TopCheck::first_order`], (b) counts as the header it found.
fn check_top_core(
    read: &dyn Fn(u64) -> Option<u32>,
    base: u64,
    list_extra: u64,
    title: Option<&dyn Fn(u64) -> bool>,
    built: Option<&dyn Fn(u64) -> Built>,
) -> TopCheck {
    let mut t = TopCheck::default();
    let mut why: Option<String> = None;
    let (mut defect, mut unread): (Option<String>, Option<String>) = (None, None);
    let mut order: Option<String> = None;
    let mut defect_title = false;
    let tw = |a: u64| title.is_some_and(|f| f(a));
    let fail = |why: &mut Option<String>, w: String| {
        if why.is_none() {
            *why = Some(w);
        }
    };
    let note = |first: &mut Option<String>, w: &str| {
        if first.is_none() {
            *first = Some(w.to_string());
        }
    };
    // The first defect and whether title words decided it.
    let note_defect = |first: &mut Option<String>, from_title: &mut bool, w: &str, by_title: bool| {
        if first.is_none() {
            *first = Some(w.to_string());
            *from_title = by_title;
        }
    };
    let Some(h) = (0..16).map(|i| read(base + 4 * i)).collect::<Option<Vec<u32>>>() else {
        t.why = Some("header not read".into());
        return t;
    };
    let f = |i: usize| f32::from_bits(h[i]);
    t.bounds_start = (4..10).any(|i| f(i).abs() == f32::MAX);
    t.instances = h[3];
    let bytes = u64::from(h[1]) * 64 + 64;
    let list_end = u64::from(h[1]) * 64 + list_extra;
    let at = |p: u32| u64::from(p & !7) << 3;
    let list = u64::from(h[13]);
    let root = h[12];
    // Header words that decide the size, root, list and count.
    let header_title = [0u64, 1, 3, 12, 13].iter().any(|&i| tw(base + 4 * i));
    let mut tree_title = header_title;
    let size_of = |p: u32| if p & 7 == 4 { 64 } else { 112 };
    let node_end = title.is_some();
    if !is_header(h[0], h[1]) || bottom_kind(h[0]).is_some() {
        fail(&mut why, format!("dword 0/1 {:#x}/{}: not a top level", h[0], h[1]));
    }
    if !(root & 7 == 4 || root & 7 == 5) || at(root) == 0 || at(root) >= bytes || (node_end && at(root) + size_of(root) > bytes) {
        fail(&mut why, format!("root +48 {root:#x} (size {bytes:#x})"));
    }
    if list == 0 || list % 64 != 0 || list + 128 * u64::from(h[3]) > list_end {
        fail(&mut why, format!("instance list +{list:#x} of {} (size {list_end:#x})", h[3]));
    }
    if let Some(w) = &why {
        note_defect(&mut defect, &mut defect_title, w, header_title);
    }
    let mut count = std::collections::HashMap::<u64, u32>::new();
    // Of each reached instance, whether a title word led to it first.
    let mut reach_title = std::collections::HashMap::<u64, bool>::new();
    let mut twice_title = false;
    let mut seen = std::collections::HashSet::new();
    // Box nodes to visit and whether a title word led there.
    let mut stack = if why.is_none() { vec![(root, header_title)] } else { Vec::new() };
    while let Some((p, path_title)) = stack.pop() {
        if !seen.insert(at(p)) {
            t.nodes_twice += 1;
            let w = format!("box node {p:#x} reached twice");
            note_defect(&mut defect, &mut defect_title, &w, path_title);
            fail(&mut why, w);
            continue;
        }
        if seen.len() > 1 << 16 {
            note(&mut unread, "more than 65536 box nodes");
            fail(&mut why, "more than 65536 box nodes".into());
            break;
        }
        let n = if p & 7 == 4 { 16 } else { 28 };
        let Some(w) = (0..n).map(|i| read(base + at(p) + 4 * i)).collect::<Option<Vec<u32>>>() else {
            t.missing += 1;
            let w = format!("box node {p:#x} not read");
            note(&mut unread, &w);
            fail(&mut why, w);
            continue;
        };
        let node_title = (0..n).any(|i| tw(base + at(p) + 4 * i));
        tree_title |= path_title || node_title;
        for i in 0..4 {
            let child = w[i];
            if child == u32::MAX {
                continue;
            }
            // The slot's pointer and what led to its node.
            let child_title = path_title || tw(base + at(p) + 4 * i as u64);
            let b: Aabb = if p & 7 == 4 {
                let (a, b, d) = (w[4 + 3 * i], w[5 + 3 * i], w[6 + 3 * i]);
                ([half(a), half(a >> 16), half(b)], [half(b >> 16), half(d), half(d >> 16)])
            } else {
                let g = |k: usize| f32::from_bits(w[4 + 6 * i + k]);
                ([g(0), g(1), g(2)], [g(3), g(4), g(5)])
            };
            match child & 7 {
                4 | 5 if at(child) != 0 && at(child) < bytes && (!node_end || at(child) + size_of(child) <= bytes) => {
                    stack.push((child, child_title))
                }
                6 if at(child) >= list && (at(child) - list) % 128 == 0 && (at(child) - list) / 128 < u64::from(h[3]) => {
                    let index = (at(child) - list) / 128;
                    let k = count.entry(index).or_default();
                    *k += 1;
                    if *k > 1 {
                        twice_title |= child_title || reach_title.get(&index).copied().unwrap_or(false);
                        continue;
                    }
                    reach_title.insert(index, child_title);
                    let entry = base + at(child);
                    let entry_title = child_title || tw(entry + 60);
                    tree_title |= entry_title;
                    let Some(e15) = read(entry + 60) else {
                        t.missing += 1;
                        let w = format!("instance {index} not read");
                        note(&mut unread, &w);
                        fail(&mut why, w);
                        continue;
                    };
                    let blas = u64::from(e15) << 8;
                    let header = (read(blas), read(blas + 4));
                    let mut kind = match header {
                        (Some(d0), Some(d1)) if is_header(d0, d1) => bottom_kind(d0),
                        _ => None,
                    };
                    let mut blas_title = tw(blas) || tw(blas + 4);
                    // v3d: one class per instance.
                    if let Some(built) = built {
                        if kind.is_some() {
                            t.classes[0] += 1;
                        } else if let (Some(_), Some(_)) = header {
                            match header_past_parents(read, blas) {
                                Search::Found(h_at, k) => {
                                    t.classes[1] += 1;
                                    if t.first_b.is_none() {
                                        t.first_b = Some(format!("instance {index} {blas:#x} header +{:#x}", h_at - blas));
                                    }
                                    kind = Some(k);
                                    blas_title |= tw(h_at) || tw(h_at + 4) || tw(h_at + 56);
                                }
                                Search::NotRead(a) => {
                                    t.blas_unread += 1;
                                    t.search_unread += 1;
                                    let w = format!(
                                        "instance {index}: no bottom-level header at {blas:#x}, search past its parent table: bottom level {a:#x} not read"
                                    );
                                    note(&mut unread, &w);
                                    fail(&mut why, w);
                                    continue;
                                }
                                Search::Nothing => {
                                    t.blas_unread += 1;
                                    t.blas_no_header += 1;
                                    let w = match built(blas) {
                                        Built::Before(after) => {
                                            t.classes[3] += 1;
                                            let w = format!(
                                                "instance {index}: no bottom-level header at {blas:#x}, built before the top level (first bound after s{after}) (v3d d)"
                                            );
                                            note_defect(&mut defect, &mut defect_title, &w, entry_title || blas_title);
                                            w
                                        }
                                        Built::After(_) => {
                                            t.classes[2] += 1;
                                            let w = format!("instance {index}: bottom level {blas:#x} built after the top level (v3d c)");
                                            note(&mut order, &w);
                                            w
                                        }
                                        Built::Never => {
                                            t.classes[4] += 1;
                                            if t.never_built.len() < 4096 {
                                                t.never_built.push(blas);
                                            }
                                            let w = format!("instance {index}: bottom level {blas:#x} never a build target so far (v3d e)");
                                            note(&mut order, &w);
                                            w
                                        }
                                    };
                                    fail(&mut why, w);
                                    continue;
                                }
                            }
                        }
                    }
                    let Some(kind) = kind else {
                        t.blas_unread += 1;
                        if let (Some(_), Some(_)) = header {
                            t.blas_no_header += 1;
                            note_defect(
                                &mut defect,
                                &mut defect_title,
                                &format!("instance {index}: no bottom-level header at {blas:#x}"),
                                entry_title || blas_title,
                            );
                        } else {
                            note(&mut unread, &format!("instance {index}: bottom level {blas:#x} not read"));
                        }
                        fail(&mut why, format!("instance {index}: bottom level {blas:#x} not read or no header"));
                        continue;
                    };
                    t.kind_from_title += u32::from(blas_title);
                    let k = bottom_index(kind);
                    t.on[k] += 1;
                    let finite = (0..3).all(|a| {
                        b.0[a].is_finite() && b.1[a].is_finite() && b.0[a].abs() < f32::MAX && b.1[a].abs() < f32::MAX && b.0[a] <= b.1[a]
                    });
                    t.finite[k] += u32::from(finite);
                    if kind == Bottom::Rehydrated && !finite {
                        let w = format!("instance {index} on rehydrated {blas:#x}: box {b:?}");
                        note_defect(&mut defect, &mut defect_title, &w, child_title || node_title || blas_title);
                        fail(&mut why, w);
                    }
                }
                _ => {
                    t.bad_slots += 1;
                    let w = format!("slot {i} of {p:#x}: pointer {child:#x}");
                    note_defect(&mut defect, &mut defect_title, &w, child_title);
                    fail(&mut why, w);
                }
            }
        }
    }
    t.reached = count.len() as u32;
    t.twice = count.values().filter(|&&k| k > 1).count() as u32;
    t.never = t.instances.saturating_sub(t.reached);
    if why.is_none() && t.twice > 0 {
        why = Some(format!("{} instances reached more than once", t.twice));
    }
    if why.is_none() && t.never > 0 {
        why = Some(format!("{} of {} instances never reached", t.never, t.instances));
    }
    if t.twice > 0 {
        note_defect(&mut defect, &mut defect_title, &format!("{} instances reached more than once", t.twice), twice_title);
    }
    t.defect_from_title = defect_title;
    t.tree_from_title = tree_title;
    t.valid = why.is_none();
    t.why = why;
    t.first_defect = defect;
    t.first_unread = unread;
    t.first_order = order;
    t
}

/// `VICTORY_BVH_GT` v2: cb0..2 (user data 16..18) of the builder's
/// rehydrating stage 5e687674 by the bottom level it rehydrates (user data
/// 35 and 36, its address), as the host bound them (senior 85 ax85.py,
/// senior 89 capture layout). The first value is kept; later dispatches
/// with other values are counted.
static CB: std::sync::Mutex<Option<std::collections::HashMap<u64, [u32; 3]>>> = std::sync::Mutex::new(None);
static CB_DIFFER: AtomicU64 = AtomicU64::new(0);

/// The rehydrating stage's shader hash (prefix) and where its values lie.
const REHYDRATE: &str = "5e687674";

/// A compute bind's root constants `values` (by user-data index) of the
/// pipeline named `name` (`crate::draw`, only with `VICTORY_BVH_GT`): a
/// 5e687674 bind notes its cb0..2 for its bottom level.
pub(crate) fn note_bind(name: &str, values: &[u32]) {
    for &(hash, at) in &BUILD_TARGETS {
        if name.contains(hash) && values.len() > at + 1 {
            note_build(u64::from(values[at]) | u64::from(values[at + 1]) << 32);
        }
    }
    if values.len() <= 36 || !name.contains(REHYDRATE) {
        return;
    }
    note_cb(u64::from(values[35]) | u64::from(values[36]) << 32, [values[16], values[17], values[18]]);
}

fn note_cb(blas: u64, cb: [u32; 3]) {
    let mut g = CB.lock().unwrap_or_else(|e| e.into_inner());
    let map = g.get_or_insert_with(Default::default);
    match map.get(&blas) {
        Some(old) if *old != cb => {
            CB_DIFFER.fetch_add(1, Ordering::Relaxed);
        }
        Some(_) => {}
        None if map.len() < 1 << 20 => {
            map.insert(blas, cb);
        }
        None => {}
    }
}

fn cb_of(blas: u64) -> Option<[u32; 3]> {
    CB.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&blas).copied()
}

/// v3d (senior 112, Fable Aufruf 8 point 2): the builder stages whose root
/// constants name the structure a dispatch writes: (shader hash prefix, user
/// data index of the address's low dword; the high dword follows). Read from
/// the dispatch lists of the s89-1 captures (s6191, s6215, s6759, s7072):
/// 5552dac6 ud16/17 names every structure built (s7072: 166 distinct, the
/// 162 bottom levels the other fresh-build stages 6f6a359d ud16, 3df629ff
/// ud30, f28a33a8 ud22, 162216b9 ud22, f6e47efa ud24 name, plus the 4 top
/// levels of 190fbf0e ud22); 5e687674 ud35/36 every rehydrated one (69, the
/// same set as 4ddd6f97 and caa78b8c ud35).
const BUILD_TARGETS: [(&str, usize); 2] = [("5552dac6", 16), (REHYDRATE, 35)];

/// One build target: the first and last bind (sequence numbers of
/// [`Builds::seq`]) and the submission compared last before the first bind
/// ([`Gt::take`]; the build runs in a later submission).
#[derive(Clone, Copy, Debug)]
struct Build {
    first: u64,
    last: u64,
    after: u64,
}

/// The map "builder output address -> binds" of the run (every bind of the
/// stages in [`BUILD_TARGETS`], noted at record time in `crate::draw`), and
/// the (e) instances still waiting for a build.
struct Builds {
    seq: u64,
    map: std::collections::BTreeMap<u64, Build>,
    /// Binds not noted because the map was full (then (e) is not sure).
    full: u64,
    taken: u64,
    /// (e) bottom levels and the sequence number when they were found;
    /// those built later, and those not kept.
    pending: Vec<(u64, u64)>,
    later: u64,
    dropped: u64,
}

static BUILDS: std::sync::Mutex<Builds> = std::sync::Mutex::new(Builds {
    seq: 0,
    map: std::collections::BTreeMap::new(),
    full: 0,
    taken: 0,
    pending: Vec::new(),
    later: 0,
    dropped: 0,
});

fn builds() -> std::sync::MutexGuard<'static, Builds> {
    BUILDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A bind of a builder stage that writes the structure at `at`.
fn note_build(at: u64) {
    note_build_in(&mut builds(), at);
}

/// [`note_build`] into `b`.
fn note_build_in(b: &mut Builds, at: u64) {
    if at == 0 || at & 0xff != 0 {
        return;
    }
    b.seq += 1;
    let (seq, after, n) = (b.seq, b.taken, b.map.len());
    match b.map.get_mut(&at) {
        Some(e) => e.last = seq,
        None if n < 1 << 20 => {
            b.map.insert(at, Build { first: seq, last: seq, after });
        }
        None => b.full += 1,
    }
}

/// (e) bottom levels of earlier submissions bound since they were found:
/// counted in [`Builds::later`] and no longer pending.
fn resolve_pending(b: &mut Builds) {
    let pending = std::mem::take(&mut b.pending);
    let before = pending.len();
    let keep: Vec<(u64, u64)> = pending.into_iter().filter(|(blas, seq)| b.map.get(blas).is_none_or(|e| e.last <= *seq)).collect();
    b.later += (before - keep.len()) as u64;
    b.pending = keep;
}

/// Where a bottom level stands against a top level in the build map (v3d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Built {
    /// First bound before the top level's last bind; the submission compared
    /// last before that first bind ([`Build::after`]).
    Before(u64),
    /// Bound only after it.
    After(u64),
    /// Never bound so far.
    Never,
}

/// [`Built`] of the bottom level `blas` against the top level `top` (its
/// last bind; one never bound counts as bound now). The order is the record
/// order of the binds, not the GPU's order of the submissions.
fn built_of(b: &Builds, top: u64, blas: u64) -> Built {
    let Some(e) = b.map.get(&blas) else { return Built::Never };
    if e.first < b.map.get(&top).map_or(b.seq, |t| t.last) { Built::Before(e.after) } else { Built::After(e.after) }
}

/// What [`validate`] asks the build map: [`Built`] of (top, bottom level),
/// and the next build target past an address.
pub(crate) struct Targets<'a> {
    pub(crate) built: &'a dyn Fn(u64, u64) -> Built,
    pub(crate) next: &'a dyn Fn(u64) -> Option<u64>,
}

/// The most 64-byte steps past an instance's address that
/// [`header_past_parents`] looks for a header (64 KiB).
const PARENT_STEPS: u64 = 1024;

/// What [`header_past_parents`] found.
#[derive(Debug, PartialEq)]
enum Search {
    Found(u64, Bottom),
    NotRead(u64),
    Nothing,
}

/// v3d (b): a bottom-level header at `blas` + k * 64 (k = 1 ..=
/// [`PARENT_STEPS`]) whose parent table (dword 14, signed, from the header)
/// starts at `blas`: the instance names the block start, the parent table
/// lies before the header (V105: 2 of 8 "no header" 0x400 below a header).
/// Headers that point elsewhere are passed by; a word not read ends the
/// search.
fn header_past_parents(read: &dyn Fn(u64) -> Option<u32>, blas: u64) -> Search {
    for k in 1..=PARENT_STEPS {
        let at = blas + 64 * k;
        let Some(d0) = read(at) else { return Search::NotRead(at) };
        if d0 & 0xffff != 6 {
            continue;
        }
        let (Some(d1), Some(d14)) = (read(at + 4), read(at + 56)) else { return Search::NotRead(at) };
        if !is_header(d0, d1) {
            continue;
        }
        if let Some(kind) = bottom_kind(d0)
            && at as i64 + i64::from(d14 as i32) == blas as i64
        {
            return Search::Found(at, kind);
        }
    }
    Search::Nothing
}

/// v2 of one bottom level: valid as e56de5c ([`Centre::Bit30AllAxes`]),
/// valid with the rule (centre cb0..2 if a 5e687674 dispatch of it was
/// seen, else [`one_sided_centre`]), whether cb was known, and whether a
/// known cb differs from the one-sided prediction on an axis (Fable Aufruf
/// 5 point 9, reported apart, no validity criterion).
#[derive(Debug)]
pub(crate) struct BottomCheck {
    pub(crate) without_rule: Check,
    pub(crate) with_rule: Check,
    pub(crate) cb_known: bool,
    pub(crate) cb_not_predicted: bool,
}

/// [`BottomCheck`] of the bottom level at `base` (header dword 0 `d0`),
/// with `cb` its 5e687674 values if known.
pub(crate) fn check_bottom(read: &dyn Fn(u64) -> Option<u32>, base: u64, kind: Bottom, cb: Option<[u32; 3]>) -> BottomCheck {
    let without_rule = check_v2(read, base, Centre::Bit30AllAxes);
    let h: Option<Vec<u32>> = (0..16).map(|i| read(base + 4 * i)).collect();
    let predicted = h.as_deref().map(one_sided_centre);
    let centre = match (cb, predicted) {
        (Some(cb), _) => Some(cb.map(f32::from_bits)),
        (None, p) => p,
    };
    let with_rule = match centre {
        Some(c) if kind != Bottom::Packed => check_v2(read, base, Centre::Given(c)),
        // Packed ones hold no node boxes: the centre does not apply.
        _ => check_v2(read, base, Centre::Bit30AllAxes),
    };
    let cb_not_predicted = match (cb, predicted) {
        (Some(cb), Some(p)) => (0..3).any(|a| {
            let c = f32::from_bits(cb[a]);
            (c - p[a]).abs() > 1e-5 * c.abs().max(p[a].abs()).max(1.0)
        }),
        _ => false,
    };
    BottomCheck { without_rule, with_rule, cb_known: cb.is_some(), cb_not_predicted }
}

/// The v2 counts of one submission's headers ([`v2_of`]), and of the run.
#[derive(Default, Debug, Clone)]
pub(crate) struct V2 {
    /// By [`bottom_index`]: headers, valid without the bit-30 rule
    /// ([`Centre::Bit30AllAxes`]), valid with it ([`check_bottom`]).
    pub(crate) bottom: [[u32; 3]; 4],
    /// Rehydrated ones: centre from a 5e687674 dispatch's cb, from the
    /// header one-sided rule (no dispatch seen); cb known and not as the
    /// one-sided rule predicts.
    pub(crate) cb_known: u32,
    pub(crate) cb_fallback: u32,
    pub(crate) cb_not_predicted: u32,
    /// Top levels with finite header bounds and with +-FLT_MAX in them:
    /// headers, valid ([`check_top`]).
    pub(crate) tops: [[u32; 2]; 2],
    /// [`TopCheck`] summed over the top levels.
    pub(crate) instances: u64,
    pub(crate) reached: u64,
    pub(crate) never: u64,
    pub(crate) twice: u64,
    pub(crate) bad_slots: u64,
    pub(crate) nodes_twice: u64,
    pub(crate) missing: u64,
    pub(crate) on: [u64; 4],
    pub(crate) blas_unread: u64,
    pub(crate) finite: [u64; 4],
    /// Pages read besides the compared ones for the top levels: from the
    /// arena, from title memory, wanted past the budget.
    pub(crate) extra: [u64; 3],
    /// The first invalid ones (rehydrated with the rule, top levels).
    pub(crate) first: Vec<String>,
}

impl V2 {
    fn add(&mut self, o: &V2) {
        for (a, b) in self.bottom.iter_mut().flatten().zip(o.bottom.iter().flatten()) {
            *a += b;
        }
        for (a, b) in self.tops.iter_mut().flatten().zip(o.tops.iter().flatten()) {
            *a += b;
        }
        self.cb_known += o.cb_known;
        self.cb_fallback += o.cb_fallback;
        self.cb_not_predicted += o.cb_not_predicted;
        self.instances += o.instances;
        self.reached += o.reached;
        self.never += o.never;
        self.twice += o.twice;
        self.bad_slots += o.bad_slots;
        self.nodes_twice += o.nodes_twice;
        self.missing += o.missing;
        self.blas_unread += o.blas_unread;
        for k in 0..4 {
            self.on[k] += o.on[k];
            self.finite[k] += o.finite[k];
        }
        for k in 0..3 {
            self.extra[k] += o.extra[k];
        }
    }

    fn add_top(&mut self, t: &TopCheck) {
        let k = usize::from(t.bounds_start);
        self.tops[k][0] += 1;
        self.tops[k][1] += u32::from(t.valid);
        self.instances += u64::from(t.instances);
        self.reached += u64::from(t.reached);
        self.never += u64::from(t.never);
        self.twice += u64::from(t.twice);
        self.bad_slots += u64::from(t.bad_slots);
        self.nodes_twice += u64::from(t.nodes_twice);
        self.missing += u64::from(t.missing);
        self.blas_unread += u64::from(t.blas_unread);
        for k in 0..4 {
            self.on[k] += u64::from(t.on[k]);
            self.finite[k] += u64::from(t.finite[k]);
        }
    }

    /// The counts as one log line's text.
    pub(crate) fn text(&self) -> String {
        let b = |k: usize| format!("{}/{} (with the rule {})", self.bottom[k][1], self.bottom[k][0], self.bottom[k][2]);
        format!(
            "bottom fresh {}; rehydrated without the bit-30 rule {}/{}, with it {}/{} (centre from cb {}, from header one-sided {}; cb not as one-sided predicts {}); other bottom {}; packed {}/{}; top {}/{}, top with +-FLT_MAX header bounds {}/{} (instances {}, reached {}, never {}, more than once {}, bad slots {}, box nodes twice {}, not read {}; on packed {}, on 0x8002 {} (finite {}), on 0x8004 {} (finite {}), on other bottom {}, bottom level not read {}; pages read besides {} from the arena, {} from title memory, {} past the budget)",
            b(2),
            self.bottom[1][1],
            self.bottom[1][0],
            self.bottom[1][2],
            self.bottom[1][0],
            self.cb_known,
            self.cb_fallback,
            self.cb_not_predicted,
            b(3),
            self.bottom[0][1],
            self.bottom[0][0],
            self.tops[0][1],
            self.tops[0][0],
            self.tops[1][1],
            self.tops[1][0],
            self.instances,
            self.reached,
            self.never,
            self.twice,
            self.bad_slots,
            self.nodes_twice,
            self.missing,
            self.on[0],
            self.on[1],
            self.finite[1],
            self.on[2],
            self.finite[2],
            self.on[3],
            self.blas_unread,
            self.extra[0],
            self.extra[1],
            self.extra[2],
        )
    }
}

/// The third validator's verdict on one structure (Fable Aufruf 7 point
/// 4b): valid, invalid (a defect in what was read), or not judged (part of
/// it not read: header, box nodes, instances, a bottom level, pages past
/// the budget; or the header is no plausible top level). Not judged counts
/// neither as valid nor as invalid.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Verdict {
    Valid,
    Invalid(String),
    NotJudged(String),
}

impl Verdict {
    /// Index in a [valid, invalid, not judged] count.
    fn index(&self) -> usize {
        match self {
            Verdict::Valid => 0,
            Verdict::Invalid(_) => 1,
            Verdict::NotJudged(_) => 2,
        }
    }

    fn why(&self) -> &str {
        match self {
            Verdict::Valid => "",
            Verdict::Invalid(w) | Verdict::NotJudged(w) => w,
        }
    }
}

/// v3 of one bottom level: as [`check_v2`], but the structure ends exactly
/// at D1 * 64 (senior 98 point 5a: v2's D1 * 64 + 64 lets a child up to 64
/// bytes past the end pass), and a structure with words or box nodes not
/// read is not judged instead of valid or invalid.
///
/// v3b (senior 105): box nodes must end inside the size, and the exact
/// D1 * 64 holds for the 0x8002 / 0x8004 (and packed) bottom levels only.
/// Other bottom levels (mark 0x8000 alone) put their root at exactly
/// D1 * 64 (runs s103: root 0x10c at size 0x840, 0x55fc at 0x2afc0, like
/// runs 164431/171123 in [`check`]): their nodes may end at D1 * 64 + 64,
/// as a top level's ([`check_top_core`]). Measured, unverified, for Fable.
///
/// v3c (senior 109): box nodes of every kind may end by D1 * 64 + 64 (a
/// fresh bottom level's box32 root at D1 * 64 - 64, see [`check_core`]);
/// where they start stays as above (0x8002 / 0x8004 inside D1 * 64).
///
/// v3d (senior 112, Fable Aufruf 8 points 7, 8): the node end as a rule by
/// kind: 0x8002 / 0x8004 start inside D1 * 64 and end by D1 * 64 + 64;
/// packed ones stay exactly within D1 * 64 ([`check_packed`]); other bottom
/// levels keep v3b's measured rule (start before, end by D1 * 64 + 64; not
/// part of Aufruf 8 point 8, for Fable). A bottom level with dword 15 bits
/// 24-29 set is not judged: the bit-30 rule is derived from capture data
/// (31/31 against 4/31), not from the shader source, and says nothing about
/// those bits.
pub(crate) fn check_v3(read: &dyn Fn(u64) -> Option<u32>, base: u64, centre: Centre) -> Verdict {
    if let Some(d15) = read(base + 60)
        && d15 & D15_UNKNOWN != 0
    {
        return Verdict::NotJudged(format!("dword 15 {d15:#x}: bits 24-29 set"));
    }
    let extra = match read(base).and_then(bottom_kind) {
        Some(Bottom::Other) => 64,
        _ => 0,
    };
    verdict_of(&check_core(read, base, extra, Some(centre), true))
}

/// [`Verdict`] of a bottom-level [`Check`]: a reason that names something
/// not read (header, a packed node word) or box nodes skipped as not read
/// make it not judged; any other reason is a defect in what was read.
fn verdict_of(c: &Check) -> Verdict {
    match &c.why {
        Some(w) if w.ends_with("not read") => Verdict::NotJudged(w.clone()),
        Some(w) => Verdict::Invalid(w.clone()),
        None if c.missing > 0 => Verdict::NotJudged(format!("{} box nodes not read", c.missing)),
        None => Verdict::Valid,
    }
}

/// v3 of one bottom level ([`check_bottom`] with [`check_v3`]): without
/// the bit-30 rule, with it, and whether the centre came from cb (the rule)
/// or from the header one-sided fallback, which v3 counts apart (senior 98
/// point 5b).
#[derive(Debug)]
pub(crate) struct BottomV3 {
    pub(crate) without_rule: Verdict,
    pub(crate) with_rule: Verdict,
    pub(crate) cb_known: bool,
}

pub(crate) fn check_bottom_v3(read: &dyn Fn(u64) -> Option<u32>, base: u64, kind: Bottom, cb: Option<[u32; 3]>) -> BottomV3 {
    let without_rule = check_v3(read, base, Centre::Bit30AllAxes);
    let h: Option<Vec<u32>> = (0..16).map(|i| read(base + 4 * i)).collect();
    let centre = match cb {
        Some(cb) => Some(cb.map(f32::from_bits)),
        None => h.as_deref().map(one_sided_centre),
    };
    let with_rule = match centre {
        Some(c) if kind != Bottom::Packed => check_v3(read, base, Centre::Given(c)),
        _ => without_rule.clone(),
    };
    BottomV3 { without_rule, with_rule, cb_known: cb.is_some() }
}

/// Dword 15 bits the bit-30 rule does not cover (v3d).
const D15_UNKNOWN: u32 = 0x3f00_0000;

/// The most instances a plausible top level holds (v3).
const MOST_INSTANCES: u64 = 1 << 20;

/// Why the header `h` is no plausible top level (v3, "not judged, no top
/// level"): no version-6 header without bottom-level marks, a root pointer
/// of no box-node type (4, 5) or outside the size, an instance list offset
/// of 0 or no 64-byte multiple (11 such headers in the s7072 harness dump,
/// senior 100), or an instance count of 0, above
/// [`MOST_INSTANCES`], or one whose list of 128 bytes per instance could
/// not lie within D1 * 64 at all (the 3.2e9 / 4.3e9 counts
/// of runs s94v2-bvh and 3070, Fable Aufruf 7 point 2). A plausible count
/// whose list does not lie within D1 * 64 is a defect ([`check_top_v3`]).
pub(crate) fn no_top_level(h: &[u32]) -> Option<String> {
    if !is_header(h[0], h[1]) || bottom_kind(h[0]).is_some() {
        return Some(format!("no top level: dword 0/1 {:#x}/{}", h[0], h[1]));
    }
    let (n, bytes) = (u64::from(h[3]), u64::from(h[1]) * 64);
    if n == 0 || n > MOST_INSTANCES || 128 * n > bytes {
        return Some(format!("no top level: {n} instances for size {bytes:#x}"));
    }
    // The root: a box node inside the size (box nodes up to D1 * 64 + 64,
    // [`check_top_core`]); the list: a 64-byte multiple past the header.
    let root = u64::from(h[12] & !7) << 3;
    if !(h[12] & 7 == 4 || h[12] & 7 == 5) || root == 0 || root >= bytes + 64 {
        return Some(format!("no top level: root pointer {:#x} names no box node inside the size", h[12]));
    }
    if h[13] == 0 || h[13] % 64 != 0 {
        return Some(format!("no top level: instance list offset {:#x}", h[13]));
    }
    None
}

/// v3 of the top level at `base` (Fable Aufruf 7 point 4b): no plausible
/// top level ([`no_top_level`]) is not judged; else [`check_top`] with the
/// instance list [dword 13, dword 13 + 128 * dword 3) within D1 * 64 (box
/// nodes up to D1 * 64 + 64, see [`check_top_core`]). A defect in what was
/// read (bad slot, box node or instance twice, the list past D1 * 64, a
/// bottom level read but no header,
/// a rehydrated instance's open box) makes it invalid; else anything not
/// read (box node, instance entry, bottom-level header) makes it not judged,
/// and "never reached" counts only when all was read.
///
/// Tests: every bottom level counts as built before the top level (v3d
/// class (d) for an instance with no header, as before v3d).
#[cfg(test)]
pub(crate) fn check_top_v3(read: &dyn Fn(u64) -> Option<u32>, base: u64) -> (TopCheck, Verdict) {
    check_top_v3_with(read, base, &|_| false, &|_| Built::Before(0))
}

/// [`check_top_v3`] with `title` naming the words read from title memory
/// (v3b, senior 105; see [`check_top_core`]) and `built` the build map's
/// [`Built`] of a bottom level against this top level (v3d): an instance
/// built after it (c) or never built (e) leaves it not judged, after defects
/// and words not read.
pub(crate) fn check_top_v3_with(
    read: &dyn Fn(u64) -> Option<u32>,
    base: u64,
    title: &dyn Fn(u64) -> bool,
    built: &dyn Fn(u64) -> Built,
) -> (TopCheck, Verdict) {
    let Some(h) = (0..16).map(|i| read(base + 4 * i)).collect::<Option<Vec<u32>>>() else {
        return (TopCheck::default(), Verdict::NotJudged("header not read".into()));
    };
    if let Some(w) = no_top_level(&h) {
        return (TopCheck { instances: h[3], ..Default::default() }, Verdict::NotJudged(w));
    }
    let t = check_top_core(read, base, 0, Some(title), Some(built));
    let v = if let Some(d) = &t.first_defect {
        Verdict::Invalid(d.clone())
    } else if let Some(u) = &t.first_unread {
        Verdict::NotJudged(u.clone())
    } else if let Some(o) = &t.first_order {
        Verdict::NotJudged(o.clone())
    } else if t.never > 0 {
        Verdict::Invalid(format!("{} of {} instances never reached", t.never, t.instances))
    } else {
        Verdict::Valid
    };
    (t, v)
}

/// [`check_top_v3`] over the compared pages (`first`, G) and the pages read
/// besides (`besides`), `title` naming the words of those that came from
/// title memory (senior 99: the 4 bad slots per top level were a root of
/// zeros on a title-memory page read before the GPU built it, valid one
/// submission later). Pages from the arena or a copy hold the GPU's bytes
/// and count.
///
/// v3b (senior 105, review senior 104 point 5): a verdict turns into "not
/// judged, read from title memory" only when title words decide it: an
/// invalid one when a word that decides its first defect came from there
/// ([`TopCheck::defect_from_title`]; "never reached": any word of the
/// tree), a valid one when any word of the tree did. Bottom-level headers
/// read from there only for their kind leave the verdict as it is and are
/// counted ([`TopCheck::kind_from_title`]). The verdict before is kept in
/// [`TopCheck::title_before`]. Tests: built before, as [`check_top_v3`].
#[cfg(test)]
pub(crate) fn top_v3_over(
    first: &dyn Fn(u64) -> Option<u32>,
    besides: &dyn Fn(u64) -> Option<u32>,
    title: &dyn Fn(u64) -> bool,
    at: u64,
) -> (TopCheck, Verdict) {
    top_v3_over_with(first, besides, title, &|_| Built::Before(0), at)
}

/// [`top_v3_over`] with the build map's [`Built`] (v3d).
pub(crate) fn top_v3_over_with(
    first: &dyn Fn(u64) -> Option<u32>,
    besides: &dyn Fn(u64) -> Option<u32>,
    title: &dyn Fn(u64) -> bool,
    built: &dyn Fn(u64) -> Built,
    at: u64,
) -> (TopCheck, Verdict) {
    let used = std::cell::Cell::new(false);
    let read = |a: u64| {
        first(a).or_else(|| {
            let w = besides(a);
            if w.is_some() && title(a) {
                used.set(true);
            }
            w
        })
    };
    let from_title = |a: u64| first(a).is_none() && title(a) && besides(a).is_some();
    let (mut t, v) = check_top_v3_with(&read, at, &from_title, built);
    t.any_title = used.get();
    let decided = match &v {
        Verdict::Invalid(w) if w.ends_with("instances never reached") => t.tree_from_title,
        Verdict::Invalid(_) => t.defect_from_title,
        Verdict::Valid => t.tree_from_title,
        Verdict::NotJudged(_) => false,
    };
    let v = if decided {
        t.title_before = Some(v.index());
        Verdict::NotJudged(format!("read from title memory besides (may predate the GPU's build): {}", match &v {
            Verdict::Valid => "valid",
            Verdict::Invalid(w) | Verdict::NotJudged(w) => w,
        }))
    } else {
        v
    };
    (t, v)
}

/// The v3 counts of one submission's headers ([`validate`]), and of the
/// run. Every count of verdicts is [valid, invalid, not judged].
#[derive(Default, Debug, Clone)]
pub(crate) struct V3 {
    /// By [`bottom_index`]; rehydrated ones without the bit-30 rule.
    pub(crate) bottom: [[u32; 3]; 4],
    /// Rehydrated ones with the rule, centre from a 5e687674 dispatch's cb
    /// (the rule's count), and apart those with the header one-sided
    /// fallback (no dispatch seen; senior 98 point 5b).
    pub(crate) rule: [u32; 3],
    pub(crate) fallback: [u32; 3],
    /// Top levels with finite header bounds and with +-FLT_MAX in them.
    pub(crate) tops: [[u32; 3]; 2],
    /// Of the headers not judged: no plausible top level; top levels with a
    /// box node or instance entry not read; with a bottom level not read;
    /// judged by a page read besides from title memory ([`top_v3_over`]).
    pub(crate) no_top: u32,
    pub(crate) top_unread: [u32; 3],
    /// v3b (senior 105): of those read from title memory besides, the
    /// verdict before ([valid, invalid, -]: "downgraded, of them invalid");
    /// judged top levels with title words that decided nothing (invalid
    /// ones apart); reached instances whose bottom-level kind came from
    /// title memory; pages read besides from a copy's GPU bytes.
    pub(crate) title_down: [u32; 3],
    pub(crate) judged_with_title: [u32; 2],
    pub(crate) kind_from_title: u64,
    pub(crate) besides_from_copies: u64,
    /// Summed over the judged top levels only (valid or invalid), so that
    /// "never" holds no instance merely not read.
    pub(crate) instances: u64,
    pub(crate) reached: u64,
    pub(crate) never: u64,
    pub(crate) twice: u64,
    pub(crate) bad_slots: u64,
    pub(crate) nodes_twice: u64,
    pub(crate) no_header: u64,
    pub(crate) on: [u64; 4],
    pub(crate) finite: [u64; 4],
    /// The first invalid and the first not judged ones (rehydrated with the
    /// rule, top levels).
    pub(crate) first: Vec<String>,
    pub(crate) first_not_judged: Vec<String>,
    /// v3d (senior 112): reached instances by class [a, b, c, d, e] over
    /// every plausible top level (judged or not), and those whose header
    /// search hit a word not read; top levels not judged for a (c) or an (e)
    /// instance; bottom levels not judged for dword 15 bits 24-29; fresh and
    /// rehydrated bottom levels with a later build target, of them those
    /// whose D1 * 64 + 64 and D1 * 64 reach it (the title's output buffer is
    /// at most the gap); the (e) bottom levels ([`Builds::pending`]); first
    /// (b) instances and overlaps.
    pub(crate) classes: [u64; 5],
    pub(crate) search_unread: u64,
    pub(crate) order_tops: [u32; 2],
    pub(crate) d15_unknown: u32,
    pub(crate) gap: [u32; 3],
    pub(crate) never_built: Vec<u64>,
    pub(crate) first_v3d: Vec<String>,
}

impl V3 {
    fn add(&mut self, o: &V3) {
        let pairs = self
            .bottom
            .iter_mut()
            .flatten()
            .zip(o.bottom.iter().flatten())
            .chain(self.tops.iter_mut().flatten().zip(o.tops.iter().flatten()))
            .chain(self.rule.iter_mut().zip(&o.rule))
            .chain(self.fallback.iter_mut().zip(&o.fallback))
            .chain(self.top_unread.iter_mut().zip(&o.top_unread))
            .chain(self.title_down.iter_mut().zip(&o.title_down))
            .chain(self.judged_with_title.iter_mut().zip(&o.judged_with_title))
            .chain(self.order_tops.iter_mut().zip(&o.order_tops))
            .chain(self.gap.iter_mut().zip(&o.gap));
        for (a, b) in pairs {
            *a += b;
        }
        for (a, b) in self.classes.iter_mut().zip(&o.classes) {
            *a += b;
        }
        self.search_unread += o.search_unread;
        self.d15_unknown += o.d15_unknown;
        self.no_top += o.no_top;
        self.kind_from_title += o.kind_from_title;
        self.besides_from_copies += o.besides_from_copies;
        self.instances += o.instances;
        self.reached += o.reached;
        self.never += o.never;
        self.twice += o.twice;
        self.bad_slots += o.bad_slots;
        self.nodes_twice += o.nodes_twice;
        self.no_header += o.no_header;
        for k in 0..4 {
            self.on[k] += o.on[k];
            self.finite[k] += o.finite[k];
        }
    }

    fn add_bottom(&mut self, at: u64, kind: Bottom, c: &BottomV3) {
        self.bottom[bottom_index(kind)][c.without_rule.index()] += 1;
        self.d15_unknown += u32::from(matches!(&c.without_rule, Verdict::NotJudged(w) if w.starts_with("dword 15")));
        if kind != Bottom::Rehydrated {
            self.note(format!("{kind:?} {at:#x}"), &c.without_rule);
            return;
        }
        if c.cb_known {
            self.rule[c.with_rule.index()] += 1;
        } else {
            self.fallback[c.with_rule.index()] += 1;
        }
        self.note(format!("rehydrated {at:#x}"), &c.with_rule);
    }

    fn add_top(&mut self, at: u64, t: &TopCheck, v: &Verdict) {
        let start = usize::from(t.bounds_start);
        if !matches!(v, Verdict::NotJudged(w) if w.starts_with("no top level")) {
            for (a, b) in self.classes.iter_mut().zip(&t.classes) {
                *a += u64::from(*b);
            }
            self.search_unread += u64::from(t.search_unread);
            let room = 4096usize.saturating_sub(self.never_built.len());
            self.never_built.extend(t.never_built.iter().take(room));
            if let Some(b) = &t.first_b
                && self.first_v3d.len() < 3
            {
                self.first_v3d.push(format!("top {at:#x} (b) {b}"));
            }
        }
        match v {
            Verdict::NotJudged(w) if w.starts_with("no top level") => self.no_top += 1,
            Verdict::NotJudged(w) if w.ends_with("(v3d c)") || w.ends_with("(v3d e)") => {
                self.tops[start][2] += 1;
                self.order_tops[usize::from(w.ends_with("(v3d e)"))] += 1;
            }
            Verdict::NotJudged(w) => {
                self.tops[start][2] += 1;
                let k = if w.starts_with("read from title memory") {
                    if let Some(b) = t.title_before {
                        self.title_down[b] += 1;
                    }
                    2
                } else {
                    usize::from(t.first_unread.as_deref().is_some_and(|u| u.contains("bottom level")))
                };
                self.top_unread[k] += 1;
            }
            _ => {
                self.tops[start][v.index()] += 1;
                if t.any_title {
                    self.judged_with_title[usize::from(matches!(v, Verdict::Invalid(_)))] += 1;
                }
                self.kind_from_title += u64::from(t.kind_from_title);
                self.instances += u64::from(t.instances);
                self.reached += u64::from(t.reached);
                self.never += u64::from(t.never);
                self.twice += u64::from(t.twice);
                self.bad_slots += u64::from(t.bad_slots);
                self.nodes_twice += u64::from(t.nodes_twice);
                self.no_header += u64::from(t.blas_no_header);
                for k in 0..4 {
                    self.on[k] += u64::from(t.on[k]);
                    self.finite[k] += u64::from(t.finite[k]);
                }
            }
        }
        if !matches!(v, Verdict::NotJudged(w) if w.starts_with("no top level")) {
            self.note(format!("top {at:#x}"), v);
        }
    }

    fn note(&mut self, what: String, v: &Verdict) {
        let list = match v {
            Verdict::Valid => return,
            Verdict::Invalid(_) => &mut self.first,
            Verdict::NotJudged(_) => &mut self.first_not_judged,
        };
        if list.len() < 3 {
            list.push(format!("{what} {}", v.why()));
        }
    }

    /// The counts as one log line's text.
    pub(crate) fn text(&self) -> String {
        let c = |k: &[u32; 3]| format!("{}/{}/{}", k[0], k[1], k[2]);
        format!(
            "valid/invalid/not judged: bottom fresh {}, rehydrated without the bit-30 rule {}, with the rule (centre from cb) {}, header one-sided fallback apart {}, other bottom {}, packed {}; top {}, top with +-FLT_MAX header bounds {}; not judged: no top level {}, top with a box node or instance not read {}, with a bottom level not read {}, read from title memory besides {}; judged top levels: instances {}, reached {}, never {}, more than once {}, bad slots {}, box nodes twice {}, bottom level no header {}; on packed {}, on 0x8002 {} (finite {}), on 0x8004 {} (finite {}), on other bottom {}; v3b: read from title memory besides, verdict before valid/invalid {}/{}; judged with title words that decided nothing valid/invalid {}/{}; bottom-level kind from title memory {}; pages read besides from copies {}; v3d: instances a header / b past the parent table / c built after the top level / d built before, no header / e never a build target {}/{}/{}/{}/{} (header search not read {}); top not judged for c {}, for e {}; bottom not judged for dword 15 bits 24-29 {}; fresh/rehydrated with a later build target {}, D1*64+64 past it {}, D1*64 past it {}",
            c(&self.bottom[2]),
            c(&self.bottom[1]),
            c(&self.rule),
            c(&self.fallback),
            c(&self.bottom[3]),
            c(&self.bottom[0]),
            c(&self.tops[0]),
            c(&self.tops[1]),
            self.no_top,
            self.top_unread[0],
            self.top_unread[1],
            self.top_unread[2],
            self.instances,
            self.reached,
            self.never,
            self.twice,
            self.bad_slots,
            self.nodes_twice,
            self.no_header,
            self.on[0],
            self.on[1],
            self.finite[1],
            self.on[2],
            self.finite[2],
            self.on[3],
            self.title_down[0],
            self.title_down[1],
            self.judged_with_title[0],
            self.judged_with_title[1],
            self.kind_from_title,
            self.besides_from_copies,
            self.classes[0],
            self.classes[1],
            self.classes[2],
            self.classes[3],
            self.classes[4],
            self.search_unread,
            self.order_tops[0],
            self.order_tops[1],
            self.d15_unknown,
            self.gap[0],
            self.gap[1],
            self.gap[2],
        )
    }
}

/// v3 totals of the run ([`Gt::take`]).
static V3_TOTAL: std::sync::Mutex<Option<V3>> = std::sync::Mutex::new(None);

/// v2 totals of the run ([`Gt::take`]).
static V2_TOTAL: std::sync::Mutex<Option<V2>> = std::sync::Mutex::new(None);

/// `VICTORY_BVH_GT_TOP_PAGES` (256): the most 64 KiB pages a submission
/// reads besides the compared ones to follow its top levels' instances to
/// their bottom levels.
fn most_top_pages() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_GT_TOP_PAGES", 256) as usize)
}

/// Pages for the top levels' check that the compared ones lack (title
/// addresses of 64 KiB pages): the arena's bytes where it holds them, else
/// (v3b, senior 105) a copy's GPU bytes where one copy holds the whole page
/// (the GPU's state after the fence, as for the compared pages; title
/// memory gets it only after this check), else title memory. Counts into
/// `extra` ([`V2::extra`]: arena and copies under 0) and `copies`.
fn more_pages(
    want: &std::collections::BTreeSet<u64>,
    budget: &mut usize,
    extra: &mut [u64; 3],
    title: &mut std::collections::HashSet<u64>,
    copies: &mut u64,
) -> Vec<(usize, Vec<u8>)> {
    let wanted: Vec<u64> = want.iter().copied().take(*budget).collect();
    extra[2] += (want.len() - wanted.len()) as u64;
    *budget -= wanted.len();
    let mut got: Vec<(usize, Vec<u8>)> = crate::arena::fetch(&wanted).into_iter().map(|(at, b)| (at as usize, b)).collect();
    extra[0] += got.len() as u64;
    // `VICTORY_BVH_GT_COPIES=0`: title memory as in v3 90102a3.
    static COPIES: OnceLock<bool> = OnceLock::new();
    let from_copies = *COPIES.get_or_init(|| number("VICTORY_BVH_GT_COPIES", 1) != 0);
    let all = if from_copies && wanted.len() > got.len() { crate::shadow::all() } else { Vec::new() };
    for &page in &wanted {
        if got.iter().any(|(at, _)| *at as u64 == page) {
            continue;
        }
        let whole = all.iter().find(|c| {
            let (start, len) = c.range();
            start <= page && page + (1 << 16) <= start + len && c.title_changed(page as usize) != Some(true)
        });
        if let Some((from, b)) = whole.and_then(|c| c.gpu_bytes(page as usize))
            && from as u64 == page
            && b.len() == 1 << 16
        {
            extra[0] += 1;
            *copies += 1;
            got.push((from, b));
            continue;
        }
        let mut b = vec![0u8; 1 << 16];
        if crate::host::read_into(page as *const u8, &mut b) {
            extra[1] += 1;
            title.insert(page);
            got.push((page as usize, b));
        }
    }
    got
}

/// The v2 counts of the structures with headers at `headers` in `pages`
/// (start, bytes). With `fetch`, a top level's box nodes, instances and
/// their bottom levels' headers not in `pages` are read besides
/// ([`more_pages`], within [`most_top_pages`]); without it they count as
/// not read.
#[cfg(test)]
pub(crate) fn v2_of(pages: &[(usize, &[u8])], headers: &[u64], fetch: bool) -> V2 {
    validate(pages, headers, fetch, &Targets { built: &|_, _| Built::Before(0), next: &|_| None }).0
}

/// The v2 and v3 counts ([`V2`], [`V3`]) of the structures with headers at
/// `headers` in `pages`, as [`v2_of`] says; v3 sees the same pages,
/// including those read besides for the top levels, and asks `targets` (the
/// build map, v3d) where a bottom level without a header stands and how far
/// the next build target past a bottom level lies.
pub(crate) fn validate(pages: &[(usize, &[u8])], headers: &[u64], fetch: bool, targets: &Targets) -> (V2, V3) {
    let mut v = V2::default();
    let mut v3 = V3::default();
    let read = reader(pages);
    let mut extra_pages: Vec<(usize, Vec<u8>)> = Vec::new();
    // Pages read besides from title memory (not the arena): v3 judges no
    // top level by them ([`top_v3_over`]).
    let mut title_pages = std::collections::HashSet::new();
    let mut budget = if fetch { most_top_pages() } else { 0 };
    for &at in headers {
        let (Some(d0), Some(d1)) = (read(at), read(at + 4)) else { continue };
        if !is_header(d0, d1) {
            continue;
        }
        match bottom_kind(d0) {
            Some(kind) => {
                let cb = if kind == Bottom::Rehydrated { cb_of(at) } else { None };
                let c = check_bottom(&read, at, kind, cb);
                let k = bottom_index(kind);
                v.bottom[k][0] += 1;
                v.bottom[k][1] += u32::from(c.without_rule.valid);
                v.bottom[k][2] += u32::from(c.with_rule.valid);
                if kind == Bottom::Rehydrated {
                    if c.cb_known {
                        v.cb_known += 1;
                    } else {
                        v.cb_fallback += 1;
                    }
                    v.cb_not_predicted += u32::from(c.cb_not_predicted);
                    if !c.with_rule.valid && v.first.len() < 3 {
                        v.first.push(format!("rehydrated {at:#x} {}", c.with_rule.why.as_deref().unwrap_or("?")));
                    }
                }
                v3.add_bottom(at, kind, &check_bottom_v3(&read, at, kind, cb));
                // v3d (Fable Aufruf 8 point 8): is the title's output buffer
                // at least D1 * 64 + 64? At most the gap to the next build
                // target; a root past it would overwrite the neighbour.
                if matches!(kind, Bottom::Fresh | Bottom::Rehydrated)
                    && let Some(next) = (targets.next)(at)
                {
                    let (gap, size) = (next - at, u64::from(d1) * 64);
                    v3.gap[0] += 1;
                    v3.gap[1] += u32::from(gap < size + 64);
                    v3.gap[2] += u32::from(gap < size);
                    if gap < size + 64 && v3.first_v3d.len() < 3 {
                        v3.first_v3d.push(format!("{kind:?} {at:#x} D1*64 {size:#x}, next build target {next:#x} (gap {gap:#x})"));
                    }
                }
            }
            None => {
                let mut t = TopCheck::default();
                for round in 0..3 {
                    let missed = std::cell::RefCell::new(std::collections::BTreeSet::new());
                    {
                        let more = reader(&extra_pages);
                        let read_all = |a: u64| {
                            let w = read(a).or_else(|| more(a));
                            if w.is_none() {
                                missed.borrow_mut().insert(a & !0xffff);
                            }
                            w
                        };
                        t = check_top(&read_all, at);
                    }
                    let missed = missed.into_inner();
                    if missed.is_empty() || round == 2 || !fetch {
                        break;
                    }
                    let got = more_pages(&missed, &mut budget, &mut v.extra, &mut title_pages, &mut v3.besides_from_copies);
                    if got.is_empty() {
                        break;
                    }
                    extra_pages.extend(got);
                }
                v.add_top(&t);
                if !t.valid && v.first.len() < 3 {
                    v.first.push(format!("top {at:#x} {}", t.why.as_deref().unwrap_or("?")));
                }
                // v3 on the same bytes: the compared pages and those read
                // besides; what is still missing (past the budget) leaves the
                // structure not judged.
                let more = reader(&extra_pages);
                let title = |a: u64| title_pages.contains(&(a & !0xffff));
                let built = |blas: u64| (targets.built)(at, blas);
                let (t3, verdict) = top_v3_over_with(&read, &more, &title, &built, at);
                v3.add_top(at, &t3, &verdict);
            }
        }
    }
    (v, v3)
}

/// A small valid bottom-level structure of 64 dwords (256 bytes): header,
/// a box16 root at +64 with one child box (0,0,0)-(1,1,1) and three empty
/// ones, a triangle at +128, a parent table at +192 (forward form).
#[cfg(test)]
pub(crate) fn mini_bvh() -> [u32; 64] {
    let mut w = [0u32; 64];
    let one = 1f32.to_bits();
    // Header.
    w[0] = 0x8000 << 16 | 6;
    w[1] = 4;
    w[3] = 1;
    w[4..10].copy_from_slice(&[0, 0, 0, one, one, one]);
    w[10] = 64;
    w[12] = (64 >> 3) | 4;
    w[13] = 128;
    w[14] = 192;
    // Box16 root: child 0 the triangle (type 0), the rest empty.
    w[16..20].copy_from_slice(&[128 >> 3, u32::MAX, u32::MAX, u32::MAX]);
    w[20..23].copy_from_slice(&[0x0000_0000, 0x3c00_0000, 0x3c00_3c00]);
    for i in 1..4 {
        w[20 + 3 * i..23 + 3 * i].copy_from_slice(&[0x7c00_7c00, 0xfc00_7c00, 0xfc00_fc00]);
    }
    // Triangle (0,0,0) (1,0,0) (0,1,0).
    w[32..41].copy_from_slice(&[0, 0, 0, one, 0, 0, 0, one, 0]);
    // Parent table by node index: the root has none, the triangle the root.
    let root = w[12];
    w[48..51].copy_from_slice(&[u32::MAX, u32::MAX, root]);
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_of(words: &[u32], base: u64) -> impl Fn(u64) -> Option<u32> + '_ {
        move |at| words.get(at.checked_sub(base)? as usize / 4).copied()
    }

    #[test]
    fn a_reasons_class_drops_its_addresses_and_numbers() {
        let a = why_class(Some("child box of 0x1234 outside 0x40's"));
        let b = why_class(Some("child box of 0x99 outside 0x80's"));
        assert_eq!(a, b);
        assert_eq!(a, "child box of outside");
        assert_ne!(a, why_class(Some("child 0x12 of 0x40 past the size")));
        assert_eq!(why_class(Some("root +48 0xa4c (size 0x5240)")), "root (size");
        assert_eq!(why_class(None), "");
    }

    #[test]
    fn the_mini_structure_passes_and_broken_ones_do_not() {
        let base = 0x1000;
        let good = mini_bvh();
        let c = check(&read_of(&good, base), base);
        assert!(c.valid, "{c:?}");
        assert_eq!((c.boxes, c.outside, c.parents), (1, 0, (1, 1, 0)));
        // Zeros (inputs from pages with no copy).
        assert!(!check(&read_of(&[0; 64], base), base).valid);
        // Root pointer 0 at +48.
        let mut w = good;
        w[12] = 0;
        assert!(check(&read_of(&w, base), base).why.unwrap().starts_with("root"));
        // The builder's start values left in the bounds.
        let mut w = good;
        w[4] = f32::MAX.to_bits();
        w[7] = (-f32::MAX).to_bits();
        assert!(check(&read_of(&w, base), base).why.unwrap().contains("FLT_MAX"));
        // An empty box where the child is.
        let mut w = good;
        w[20..23].copy_from_slice(&[0x7c00_7c00, 0xfc00_7c00, 0xfc00_fc00]);
        assert!(!check(&read_of(&w, base), base).valid);
        // A child box outside the header's bounds (max x 4.0).
        let mut w = good;
        w[21] = 0x4400_0000;
        assert!(check(&read_of(&w, base), base).why.unwrap().contains("outside"));
    }

    /// A small packed bottom level of 144 dwords (rt/packed-validator
    /// e56de5c): header, parent area at +64, three leaf slots at +128, a
    /// group of two box16 nodes at +320 (node 0: leaves +128 and +192; node
    /// 1: leaf +256 with its box stored as six f32, mask 0x3f), a box32 root
    /// at +448 with the two nodes.
    fn mini_packed() -> [u32; 144] {
        let mut w = [0u32; 144];
        let one = 1f32.to_bits();
        w[0] = 0x8102 << 16 | 6;
        w[1] = 9;
        w[3] = 3;
        w[4..10].copy_from_slice(&[0, 0, 0, one, one, one]);
        w[10] = 320;
        w[12] = (448 >> 3) | 5;
        w[13] = 128;
        w[14] = 64;
        w[15] = 0x8090_8f8f;
        w[80..85].copy_from_slice(&[0x0200_0000, 0x0100_003f, 128 >> 3, 192 >> 3, 256 >> 3]);
        w[85..91].copy_from_slice(&[0, 0, 0, 0x4000_0000, 0x4000_0000, 0x4000_0000]);
        w[112..115].copy_from_slice(&[0x0200_0000, (320 >> 3) | 4, (384 >> 3) | 4]);
        w
    }

    const V2_ALL: Centre = Centre::Bit30AllAxes;

    /// v2 checks still packed structures (the old validator has no such
    /// check: every one is invalid there, s94 "packed 0/186").
    #[test]
    fn a_packed_structure_passes_v2_and_broken_ones_do_not() {
        let base = 0x2000;
        let good = mini_packed();
        assert!(!check(&read_of(&good, base), base).valid, "old logic: not valid");
        let c = check_v2(&read_of(&good, base), base, V2_ALL);
        assert!(c.valid, "{c:?}");
        assert_eq!(c.boxes, 3);
        let why = |w: &[u32]| check_v2(&read_of(w, base), base, V2_ALL).why.unwrap_or_default();
        // A leaf reached twice (node 1 names node 0's first leaf).
        let mut w = good;
        w[84] = 128 >> 3;
        assert!(why(&w).contains("reached twice"), "{}", why(&w));
        // A child pointer to no node.
        let mut w = good;
        w[114] = (512 >> 3) | 4;
        assert!(!why(&w).is_empty());
        // More elements than leaves in the tree.
        let mut w = good;
        w[3] = 4;
        assert!(why(&w).contains("leaves"), "{}", why(&w));
        // The builder's start values in the bounds.
        let mut w = good;
        w[4] = f32::MAX.to_bits();
        assert!(why(&w).contains("FLT_MAX"));
        // A mask naming one stored f32 too few: the sixth is left over.
        let mut w = good;
        w[81] = 0x0100_001f;
        assert!(!why(&w).is_empty());
    }

    /// v2 compares a rehydrated structure's node boxes in its stored units
    /// (scale word, dword 15); the old validator compares them with the
    /// header's object units.
    #[test]
    fn rehydrated_boxes_are_compared_in_stored_units() {
        // Scale word 0x8080_8080: every axis stored in units of 2^(127 - 128),
        // so the header's 0..1 is 0..2 in the nodes.
        let base = 0x1000;
        let mut w = mini_bvh();
        w[0] = 0x8002 << 16 | 6;
        w[15] = 0x8080_8080;
        w[20..23].copy_from_slice(&[0x0000_0000, 0x4000_0000, 0x4000_4000]);
        assert!(!check(&read_of(&w, base), base).valid, "old logic: 0..2 outside 0..1");
        let c = check_v2(&read_of(&w, base), base, V2_ALL);
        assert!(c.valid, "{c:?}");
        // 0..4 stored is 0..2 in object units, outside the header's 0..1.
        w[21] = 0x4400_0000;
        assert!(check_v2(&read_of(&w, base), base, V2_ALL).why.unwrap().contains("outside"));
        // With the centre bit the stored box is around the centre 0.5.
        let mut w = mini_bvh();
        w[15] = 0xc080_8080;
        w[20..23].copy_from_slice(&[0xbc00_bc00, 0x3c00_bc00, 0x3c00_3c00]);
        assert!(check_v2(&read_of(&w, base), base, V2_ALL).valid);
    }

    /// A rehydrated bottom level with bit 30 whose header box covers 0 on x
    /// (-1..3) and lies on one side on y (2..4): the title centres y only
    /// (cb = 0, 3, 0), as the rule says; centring every axis (e56de5c) puts
    /// x's stored box off by half the extent.
    fn bit30_bvh() -> [u32; 64] {
        let mut w = mini_bvh();
        let f = |x: f32| x.to_bits();
        w[0] = 0x8002 << 16 | 6;
        w[4..10].copy_from_slice(&[f(-1.0), f(2.0), f(-1.0), f(3.0), f(4.0), f(1.0)]);
        // Stored = object x 2 on every axis (exponent 0x80), bit 30.
        w[15] = 0xc080_8080;
        // Child box = the header box in stored units around (0, 3, 0):
        // x -2..6, y -2..2, z -2..2 (f16 c000, 4600, 4000).
        w[20..23].copy_from_slice(&[0xc000_c000, 0x4600_c000, 0x4000_4000]);
        w
    }

    #[test]
    fn the_bit_30_rule_takes_the_centre_from_cb_per_axis() {
        let base = 0x3000;
        let w = bit30_bvh();
        let read = read_of(&w, base);
        let cb = [0f32, 3.0, 0.0].map(f32::to_bits);
        // Without the rule (e56de5c: every axis centred) not valid, with it valid.
        let c = check_bottom(&read, base, Bottom::Rehydrated, Some(cb));
        assert!(!c.without_rule.valid, "{c:?}");
        assert!(c.without_rule.why.as_deref().unwrap().contains("outside"));
        assert!(c.with_rule.valid, "{c:?}");
        assert!(c.cb_known && !c.cb_not_predicted);
        // The rule can fail: cb centring x too is wrong.
        let c = check_bottom(&read, base, Bottom::Rehydrated, Some([1f32, 3.0, 0.0].map(f32::to_bits)));
        assert!(!c.with_rule.valid, "{c:?}");
        assert!(c.cb_not_predicted, "x = 1 where the header covers 0");
    }

    #[test]
    fn without_cb_the_header_one_sided_rule_gives_the_centre() {
        let base = 0x3000;
        let w = bit30_bvh();
        assert_eq!(one_sided_centre(&w[..16]), [0.0, 3.0, 0.0]);
        let c = check_bottom(&read_of(&w, base), base, Bottom::Rehydrated, None);
        assert!(!c.without_rule.valid);
        assert!(c.with_rule.valid, "{c:?}");
        assert!(!c.cb_known && !c.cb_not_predicted);
    }

    #[test]
    fn bottom_levels_are_classed_by_their_marks() {
        assert_eq!(bottom_kind(0x8004_0006), Some(Bottom::Fresh));
        assert_eq!(bottom_kind(0x8002_0006), Some(Bottom::Rehydrated));
        assert_eq!(bottom_kind(0x8102_0006), Some(Bottom::Packed));
        assert_eq!(bottom_kind(0x8000_0006), Some(Bottom::Other));
        assert_eq!(bottom_kind(0x0000_0006), None);
    }

    /// A 5e687674 bind notes cb0..2 (user data 16..18) for the bottom level
    /// at user data 35/36; other programs are not looked at.
    #[test]
    fn a_rehydrating_bind_notes_its_centre_values() {
        let mut values = vec![0u32; 39];
        values[16..19].copy_from_slice(&[1, 2, 3]);
        values[35] = 0x5f00_0000;
        values[36] = 0x77;
        note_bind("[\"cs 5e68767498c57eb6\"]", &values);
        assert_eq!(cb_of(0x77_5f00_0000), Some([1, 2, 3]));
        values[36] = 0x78;
        note_bind("[\"cs 190fbf0e00000000\"]", &values);
        assert_eq!(cb_of(0x78_5f00_0000), None);
        note_bind("cs 5e687674", &values[..36]);
        assert_eq!(cb_of(0x78_5f00_0000), None, "too few values");
    }

    /// A top level at 0x10000: a box32 root at +64 with three instances
    /// (list at +192, 128 bytes each) on a rehydrated (0x20000), a still
    /// packed (0x20100) and a fresh (0x20200) bottom level. The packed one's
    /// slot box and the header bounds keep +-FLT_MAX (190fbf0e leaves it off).
    fn mini_top() -> std::collections::HashMap<u64, u32> {
        let base = 0x10000u64;
        let mut m = std::collections::HashMap::new();
        let mut put = |at: u64, w: u32| {
            m.insert(at, w);
        };
        let f = |x: f32| x.to_bits();
        let header = [6, 9, 0, 3, f(f32::MAX), f(f32::MAX), f(f32::MAX), f(-f32::MAX), f(-f32::MAX), f(-f32::MAX), 64, 0, (64 >> 3) | 5, 192, 0, 0];
        for (i, w) in header.iter().enumerate() {
            put(base + 4 * i as u64, *w);
        }
        let boxes: [[f32; 6]; 3] = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [f32::MAX, f32::MAX, f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX], [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]];
        for i in 0..4u64 {
            let child = if i < 3 { ((192 + 128 * i as u32) >> 3) | 6 } else { u32::MAX };
            put(base + 64 + 4 * i, child);
            for k in 0..6u64 {
                let v = if i < 3 { f(boxes[i as usize][k as usize]) } else { 0 };
                put(base + 64 + 16 + 4 * (6 * i + k), v);
            }
        }
        for (i, blas) in [0x20000u64, 0x20100, 0x20200].into_iter().enumerate() {
            let entry = base + 192 + 128 * i as u64;
            for k in 0..32u64 {
                put(entry + 4 * k, 0);
            }
            put(entry + 60, (blas >> 8) as u32);
        }
        for (blas, d0) in [(0x20000u64, 0x8002_0006u32), (0x20100, 0x8102_0006), (0x20200, 0x8004_0006)] {
            put(blas, d0);
            put(blas + 4, 4);
        }
        m
    }

    fn read_map(m: &std::collections::HashMap<u64, u32>) -> impl Fn(u64) -> Option<u32> + '_ {
        move |at| m.get(&at).copied()
    }

    // v3 (senior 100, Fable Aufruf 7 point 4b). Each test is red against
    // the v2 logic (size D1 * 64 + 64, two states, fallback in the rule's
    // count; red proof in the JOURNAL entry "senior 100").

    /// A child 32 bytes past D1 * 64 is invalid: v2's size D1 * 64 + 64 let
    /// it pass (senior 98 point 5a). The last byte inside still passes.
    #[test]
    fn v3_a_child_32_bytes_past_the_size_is_invalid() {
        let base = 0x5000;
        let mut w = mini_bvh();
        // A fresh (0x8004) bottom level: the exact size holds for 0x8002 /
        // 0x8004 (v3b; other bottom levels end at D1 * 64 + 64).
        w[0] = 0x8004 << 16 | 6;
        // Child 0 of the root: a triangle at 256 + 32 (the size is 4 * 64).
        w[16] = (256 + 32) >> 3;
        let v = check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("past the size")), "{v:?}");
        // At 256 exactly: past the end too; at 248: inside.
        w[16] = 256 >> 3;
        assert!(matches!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Invalid(_)));
        w[16] = 248 >> 3;
        assert_eq!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Valid);
        assert_eq!(check_v3(&read_of(&mini_bvh(), base), base, Centre::Bit30AllAxes), Verdict::Valid);
    }

    // v3b (senior 105, review senior 104 points 4 and 5, field data senior
    // 103). Each test is red against v3 90102a3 (red proof in the JOURNAL
    // entry "senior 105").

    /// [`mini_top`] with its three bottom-level headers moved out of `first`
    /// (G) to `besides`, as the game reads them when no arena page holds
    /// them.
    fn heads_besides(m: &std::collections::HashMap<u64, u32>) -> (std::collections::HashMap<u64, u32>, std::collections::HashMap<u64, u32>) {
        let heads: Vec<u64> = [0x20000u64, 0x20100, 0x20200].iter().flat_map(|&b| [b, b + 4]).collect();
        let first = m.iter().filter(|(a, _)| !heads.contains(a)).map(|(a, w)| (*a, *w)).collect();
        let besides = m.iter().filter(|(a, _)| heads.contains(a)).map(|(a, w)| (*a, *w)).collect();
        (first, besides)
    }

    /// senior 104's probe: a bad slot in the root read from G, only the
    /// bottom-level headers from title memory: invalid (v3 made it "not
    /// judged" as soon as any word came from title memory). The kinds read
    /// from there are counted.
    #[test]
    fn v3b_a_bad_slot_in_g_with_bottom_heads_from_title_is_invalid() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.insert(base + 64 + 12, 0x5);
        let (first, besides) = heads_besides(&m);
        let (t, v) = top_v3_over(&read_map(&first), &read_map(&besides), &|_| true, base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("slot 3")), "{v:?}");
        assert_eq!((t.kind_from_title, t.any_title, t.title_before), (3, true, None));
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.top_unread, c.title_down, c.judged_with_title, c.kind_from_title), ([0, 1, 0], [0; 3], [0; 3], [0, 1], 3));
        // The same structure without the bad slot: valid, kinds counted.
        let (first, besides) = heads_besides(&mini_top());
        let (t, v) = top_v3_over(&read_map(&first), &read_map(&besides), &|_| true, base);
        assert_eq!((v, t.kind_from_title), (Verdict::Valid, 3));
    }

    /// The defect word itself from title memory: not judged, counted as
    /// "downgraded, of them invalid"; a valid tree with box-node words from
    /// there: not judged, "downgraded, of them valid".
    #[test]
    fn v3b_a_defect_word_from_title_is_downgraded_and_counted() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.insert(base + 64 + 12, 0x5);
        // Only slot 3's word of the root from title memory.
        let bad = base + 64 + 12;
        let first = |a: u64| if a == bad { None } else { m.get(&a).copied() };
        let besides = |a: u64| (a == bad).then(|| m[&a]);
        let (t, v) = top_v3_over(&first, &besides, &|_| true, base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.starts_with("read from title memory") && why.contains("slot 3")), "{v:?}");
        assert_eq!(t.title_before, Some(1));
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.top_unread, c.title_down, c.bad_slots), ([0, 0, 1], [0, 0, 1], [0, 1, 0], 0));
        // A valid tree whose root node came from title memory.
        let m = mini_top();
        let root = base + 64..base + 64 + 112;
        let first = |a: u64| if root.contains(&a) { None } else { m.get(&a).copied() };
        let besides = |a: u64| root.contains(&a).then(|| m.get(&a).copied().unwrap_or(0));
        let (t, v) = top_v3_over(&first, &besides, &|_| true, base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("valid")), "{v:?}");
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!(c.title_down, [1, 0, 0]);
        // A bad slot in G but reached through a pointer read from title
        // memory: the pointer decides, not judged.
        let mut m = mini_top();
        m.insert(base + 64 + 12, (576 >> 3) | 4);
        for i in 0..16u64 {
            m.insert(base + 576 + 4 * i, if i == 0 { 0x5 } else if i < 4 { u32::MAX } else { 0 });
        }
        let ptr = base + 64 + 12;
        let first = |a: u64| if a == ptr { None } else { m.get(&a).copied() };
        let besides = |a: u64| (a == ptr).then(|| m[&a]);
        let (t, v) = top_v3_over(&first, &besides, &|_| true, base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("slot 0")), "{v:?}");
        assert_eq!(t.title_before, Some(1));
    }

    /// A top level's box node must end by D1 * 64 + 64, not only start
    /// before it: a box32 at D1 * 64 (112 bytes, ends 48 past) is a bad
    /// slot, and so is a box32 root there; a box16 at D1 * 64 stays inside
    /// ([`v3_a_top_level_box_node_may_start_at_d1_times_64`]).
    #[test]
    fn v3b_a_top_level_box_node_must_end_inside() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.insert(base + 64 + 12, (576 >> 3) | 5);
        for i in 0..28u64 {
            m.insert(base + 576 + 4 * i, if i < 4 { u32::MAX } else { 0 });
        }
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("slot 3")), "{v:?}");
        assert_eq!(t.bad_slots, 1);
        // The same box32 at 512 ends at 624, inside 640.
        let mut m = mini_top();
        m.insert(base + 64 + 12, (512 >> 3) | 5);
        for i in 0..28u64 {
            m.insert(base + 512 + 4 * i, if i < 4 { u32::MAX } else { 0 });
        }
        assert_eq!(check_top_v3(&read_map(&m), base).1, Verdict::Valid);
        // A box32 root at D1 * 64 holding the three instances: invalid root.
        let mut m = mini_top();
        for i in 0..28u64 {
            let w = m.get(&(base + 64 + 4 * i)).copied().unwrap_or(0);
            m.insert(base + 576 + 4 * i, w);
        }
        m.insert(base + 48, (576 >> 3) | 5);
        let (_, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("root")), "{v:?}");
    }

    /// The mini structure's root as a box32 node (28 dwords): child 0 the
    /// triangle with box (0,0,0)-(1,1,1), the rest empty.
    fn box32_root() -> [u32; 28] {
        let mut n = [0u32; 28];
        n[..4].copy_from_slice(&[128 >> 3, u32::MAX, u32::MAX, u32::MAX]);
        n[4..10].copy_from_slice(&[0f32, 0.0, 0.0, 1.0, 1.0, 1.0].map(f32::to_bits));
        n
    }

    /// v3c (senior 109): a fresh (0x8004) bottom level's box32 root at
    /// D1 * 64 - 64 (192 of a 256-byte structure) ends 48 bytes past
    /// D1 * 64 and is valid: run V105 20261127-164739 had 115/115 fresh
    /// "ends past the size" at exactly this place and 5/5 raw 0x8004
    /// headers with the root there (v3b: 0/513 fresh valid). A box32 root
    /// at D1 * 64 still starts past the size; at 64 it is inside.
    #[test]
    fn v3c_a_fresh_box32_root_at_d1_times_64_minus_64_is_valid() {
        let base = 0x5000;
        let mut w = mini_bvh().to_vec();
        w.resize(96, 0);
        w[0] = 0x8004 << 16 | 6;
        w[48..76].copy_from_slice(&box32_root());
        w[12] = (192 >> 3) | 5;
        assert_eq!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Valid);
        let mut x = mini_bvh().to_vec();
        x.resize(100, 0);
        x[0] = 0x8004 << 16 | 6;
        x[64..92].copy_from_slice(&box32_root());
        x[12] = (256 >> 3) | 5;
        let v = check_v3(&read_of(&x, base), base, Centre::Bit30AllAxes);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("root +48")), "{v:?}");
        let mut w = mini_bvh().to_vec();
        w[0] = 0x8004 << 16 | 6;
        w[16..44].copy_from_slice(&box32_root());
        w[12] = (64 >> 3) | 5;
        assert_eq!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Valid);
    }

    /// Other bottom levels (mark 0x8000 alone) put the root at exactly
    /// D1 * 64 (s103: root 0x10c at size 0x840): valid in v3b, as in v2
    /// (v3 90102a3: "root +48 .. (size ..)", 1.7 % valid); a fresh one
    /// with its root there stays invalid (exact D1 * 64).
    #[test]
    fn v3b_other_bottom_levels_may_end_at_d1_times_64_plus_64() {
        let base = 0x5000;
        let mut w = mini_bvh().to_vec();
        w.resize(80, 0);
        let node: Vec<u32> = w[16..32].to_vec();
        w[64..80].copy_from_slice(&node);
        w[12] = (256 >> 3) | 4;
        assert_eq!(bottom_kind(w[0]), Some(Bottom::Other));
        assert_eq!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Valid);
        assert!(check_v2(&read_of(&w, base), base, Centre::Bit30AllAxes).valid, "v2 agrees");
        // Past D1 * 64 + 64 still invalid for them: a box32 root at
        // D1 * 64 ends 48 bytes past it.
        let mut x = w.clone();
        x.resize(96, 0);
        x[64..92].copy_from_slice(&box32_root());
        x[12] = (256 >> 3) | 5;
        assert!(matches!(check_v3(&read_of(&x, base), base, Centre::Bit30AllAxes), Verdict::Invalid(why) if why.contains("ends past")));
        // A fresh one with its root at D1 * 64: invalid.
        w[0] = 0x8004 << 16 | 6;
        assert!(matches!(check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes), Verdict::Invalid(_)));
    }

    /// A bottom level whose box nodes were not read is not judged, neither
    /// valid (v2: a node not read was skipped) nor invalid.
    #[test]
    fn v3_a_bottom_level_with_nodes_not_read_is_not_judged() {
        let base = 0x5000;
        let w = mini_bvh();
        // Only the header was read.
        let v = check_v3(&read_of(&w[..16], base), base, Centre::Bit30AllAxes);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("not read")), "{v:?}");
        let v = check_v3(&read_of(&w[..8], base), base, Centre::Bit30AllAxes);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("header not read")), "{v:?}");
        // A packed one with a node word not read.
        let p = mini_packed();
        let v = check_v3(&read_of(&p[..40], base), base, Centre::Bit30AllAxes);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("not read")), "{v:?}");
        assert_eq!(check_v3(&read_of(&p, base), base, Centre::Bit30AllAxes), Verdict::Valid);
    }

    /// One instance no slot names is "never reached" and invalid, when all
    /// was read; a box node not read leaves the top level not judged and
    /// its instances out of "never" (v2: invalid, all three never).
    #[test]
    fn v3_an_unreachable_instance_is_never_reached() {
        let base = 0x10000u64;
        let m = mini_top();
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert_eq!(v, Verdict::Valid, "{t:?}");
        // Slot 2 empty: instance 2 is in the list but unreachable.
        let mut m = mini_top();
        m.insert(base + 64 + 8, u32::MAX);
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("1 of 3 instances never reached")), "{v:?}");
        assert_eq!((t.reached, t.never), (2, 1));
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.never), ([0, 1, 0], 1));
        // The root box node not read: not judged, nothing counts as never.
        let mut m = mini_top();
        m.remove(&(base + 64));
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("not read")), "{v:?}");
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.never, c.instances, c.top_unread), ([0, 0, 1], 0, 0, [1, 0, 0]));
    }

    /// An instance whose bottom-level page is missing leaves the top level
    /// not judged (the instance is reached; v2: invalid "not read or no
    /// header"); a bottom level read but with no header there is a defect.
    #[test]
    fn v3_a_missing_bottom_level_page_is_not_judged_not_never_reached() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.remove(&0x20200);
        m.remove(&0x20204);
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.contains("bottom level") && why.contains("not read")), "{v:?}");
        assert_eq!((t.reached, t.never, t.blas_unread, t.blas_no_header), (3, 0, 1, 0));
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.never, c.top_unread), ([0, 0, 1], 0, [0, 1, 0]));
        // Read, but no bottom-level header there (v3d: and none past a
        // parent table, the words that search reads present; built before
        // the top level in [`check_top_v3`]: class (d)).
        let m = top_without_fresh_header();
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("no bottom-level header")), "{v:?}");
        assert_eq!(t.blas_no_header, 1);
    }

    /// A header with 3e9 instances is no top level (Fable Aufruf 7 point 2:
    /// counts up to 4.3e9 in the runs): not judged, apart from invalid.
    #[test]
    fn v3_a_header_with_3e9_instances_is_no_top_level() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.insert(base + 12, 3_000_000_000);
        let (_, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::NotJudged(why) if why.starts_with("no top level")), "{v:?}");
        let mut c = V3::default();
        c.add_top(base, &check_top_v3(&read_map(&m), base).0, &v);
        assert_eq!((c.no_top, c.tops, c.instances, c.never), (1, [[0; 3]; 2], 0, 0));
        assert!(c.first.is_empty() && c.first_not_judged.is_empty());
        // More instances than D1 * 64 could list (9 * 64 / 128 = 4.5): 5.
        m.insert(base + 12, 5);
        assert!(matches!(check_top_v3(&read_map(&m), base).1, Verdict::NotJudged(why) if why.starts_with("no top level")));
        assert!(no_top_level(&[6, 9, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0x45, 0x40, 0, 0]).is_none());
        assert!(no_top_level(&[6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x45, 0x40, 0, 0]).is_some());
    }

    /// A plausible count whose instance list ends past D1 * 64 is invalid
    /// (the list must lie within D1 * 64; v2 allowed 64 bytes more).
    #[test]
    fn v3_an_instance_list_past_the_size_is_invalid() {
        let base = 0x10000u64;
        let mut m = mini_top();
        // D1 8: size 512, the list [192, 576) ends 64 bytes past it.
        m.insert(base + 4, 8);
        let (_, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("instance list")), "{v:?}");
    }

    /// A top level's box nodes may lie up to D1 * 64 + 64 (the s89-1
    /// harness dumps: a box16 root at D1 * 64), not past it.
    #[test]
    fn v3_a_top_level_box_node_may_start_at_d1_times_64() {
        let base = 0x10000u64;
        let mut m = mini_top();
        // Slot 3 of the root names a box16 node at 576 = D1 * 64 with no
        // children.
        m.insert(base + 64 + 12, (576 >> 3) | 4);
        for i in 0..16u64 {
            m.insert(base + 576 + 4 * i, if i < 4 { u32::MAX } else { 0 });
        }
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert_eq!(v, Verdict::Valid, "{t:?}");
        // At D1 * 64 + 64: past the size, a bad slot.
        m.insert(base + 64 + 12, (640 >> 3) | 4);
        let (t, v) = check_top_v3(&read_map(&m), base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("slot 3")), "{v:?}");
        assert_eq!(t.bad_slots, 1);
        // A plausible count but a root of no box-node type: no top level.
        let mut m = mini_top();
        m.insert(base + 48, 0x61);
        assert!(matches!(check_top_v3(&read_map(&m), base).1, Verdict::NotJudged(why) if why.starts_with("no top level")));
        // A root far past the size, a list offset of no 64-byte multiple
        // (s7072: +0x29, root 0x7f1ad): no top level either.
        for (at, w) in [(base + 48, 0x7f1ad), (base + 48, 5), (base + 52, 0x29), (base + 52, 0)] {
            let mut m = mini_top();
            m.insert(at, w);
            assert!(matches!(check_top_v3(&read_map(&m), base).1, Verdict::NotJudged(why) if why.starts_with("no top level")), "{w:#x}");
        }
    }

    /// Counter-check on a capture through the harness path (senior 100;
    /// needs game data, so ignored): the pages a harness run left after a
    /// cut (`--dump`, `cut<k>-run0.index/.pages`) with the capture's PRE
    /// pages where the dump has none (as senior 86 mem86), headers found in
    /// the dump's pages as the game finds them in G, each checked by v3.
    /// cb of the 5e687674 dispatches is not read here, so rehydrated ones
    /// count as fallback. `VICTORY_V3_CAPTURE=<capture dir>`,
    /// `VICTORY_V3_PAGES=<dump dir>\cut<k>-run0`.
    #[test]
    #[ignore]
    fn v3_on_a_harness_dump() {
        use crate::bvh_replay::format::Stage;
        use std::cell::RefCell;
        use std::collections::HashMap;
        let (Ok(cap), Ok(prefix)) = (std::env::var("VICTORY_V3_CAPTURE"), std::env::var("VICTORY_V3_PAGES")) else {
            panic!("VICTORY_V3_CAPTURE and VICTORY_V3_PAGES");
        };
        let loaded = RefCell::new(crate::bvh_replay::harness::Loaded::open(std::path::Path::new(&cap)).unwrap());
        let pre: HashMap<u64, crate::bvh_replay::format::PageRec> =
            loaded.borrow().capture.pages_of(Stage::Pre).map(|p| (p.address & !0xffff, p.clone())).collect();
        let data = std::fs::read(format!("{prefix}.pages")).unwrap();
        let mut dump: Vec<(usize, Vec<u8>)> = Vec::new();
        for line in std::fs::read_to_string(format!("{prefix}.index")).unwrap().lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            let at = u64::from_str_radix(f[0].trim_start_matches("0x"), 16).unwrap();
            let (o, n): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            dump.push((at as usize, data[o..o + n].to_vec()));
        }
        let by_page: HashMap<u64, usize> = dump.iter().enumerate().map(|(i, (at, _))| (*at as u64 & !0xffff, i)).collect();
        let cache: RefCell<HashMap<u64, Option<Vec<u8>>>> = RefCell::new(HashMap::new());
        let (from_dump, from_pre, absent) = (RefCell::new(0u64), RefCell::new(0u64), RefCell::new(0u64));
        let read = |a: u64| -> Option<u32> {
            let page = a & !0xffff;
            let o = (a - page) as usize;
            let word = |b: &[u8]| b.get(o..o + 4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            if let Some(&i) = by_page.get(&page) {
                *from_dump.borrow_mut() += 1;
                return word(&dump[i].1);
            }
            let mut c = cache.borrow_mut();
            let b = c.entry(page).or_insert_with(|| pre.get(&page).and_then(|p| loaded.borrow_mut().bytes(p).ok()));
            match b {
                Some(b) => {
                    *from_pre.borrow_mut() += 1;
                    word(b)
                }
                None => {
                    *absent.borrow_mut() += 1;
                    None
                }
            }
        };
        let list: Vec<(usize, &[u8])> = dump.iter().map(|(a, b)| (*a, &b[..])).collect();
        let headers: Vec<u64> = scan(&list).headers.iter().map(|(at, _)| *at).collect();
        let mut v3 = V3::default();
        for &at in &headers {
            let (Some(d0), Some(d1)) = (read(at), read(at + 4)) else { continue };
            if !is_header(d0, d1) {
                continue;
            }
            match bottom_kind(d0) {
                Some(kind) => v3.add_bottom(at, kind, &check_bottom_v3(&read, at, kind, None)),
                None => {
                    let (t, v) = check_top_v3(&read, at);
                    let h: Vec<String> = (0..16).map(|i| read(at + 4 * i).map_or("-".into(), |w| format!("{w:x}"))).collect();
                    println!(
                        "top {at:#x}: {v:?}; instances {} reached {} never {} twice {} bad slots {} bottom level not read {} no header {}; header [{}]",
                        t.instances,
                        t.reached,
                        t.never,
                        t.twice,
                        t.bad_slots,
                        t.blas_unread - t.blas_no_header,
                        t.blas_no_header,
                        h.join(" ")
                    );
                    v3.add_top(at, &t, &v);
                }
            }
        }
        println!("capture {cap} pages {prefix}: headers {}; v3: {}", headers.len(), v3.text());
        println!("first invalid: {}; first not judged: {}", v3.first.join(", "), v3.first_not_judged.join(", "));
        println!("words read from the dump {}, from PRE {}, absent {}", from_dump.borrow(), from_pre.borrow(), absent.borrow());
    }

    /// senior 99 (1): the grid scan finds "headers" that are runs of small
    /// integers with dword 0 = 6; their dword 3 is no instance count. The
    /// header must be plausible first: else no top level, and its count
    /// goes into no total (v2 summed it: 3.2e9 and part of "never").
    #[test]
    fn v3_runs_of_small_integers_are_no_top_level_and_add_no_instances() {
        let base = 0x10000u64;
        let s6191 = [
            [6u32, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 9, 9, 9, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
            [6, 0x706, 0x706, 0x706, 0x706, 0x607, 0x607, 7, 7, 7, 7, 7, 6, 0x706, 0x706, 6],
            [6, 0x24, 0x39, 6, 5, 0x14, 0x39, 0x2d, 0x61, 0x39, 0x14, 0x2d, 0x61, 0x39, 0x14, 0x2d],
        ];
        let mut c = V3::default();
        for h in s6191 {
            let m: std::collections::HashMap<u64, u32> = h.iter().enumerate().map(|(i, w)| (base + 4 * i as u64, *w)).collect();
            let (t, v) = check_top_v3(&read_map(&m), base);
            assert!(matches!(&v, Verdict::NotJudged(why) if why.starts_with("no top level")), "{h:x?} {v:?}");
            c.add_top(base, &t, &v);
        }
        assert_eq!((c.no_top, c.instances, c.never, c.tops), (4, 0, 0, [[0; 3]; 2]));
    }

    /// senior 99 (2): a top level whose root node lies on a page read
    /// besides from title memory (zeros there: before the GPU built it; 4
    /// bad slots in v2) is not judged; the same bytes from the arena or G
    /// are judged.
    #[test]
    fn v3_a_top_level_read_from_title_memory_is_not_judged() {
        let base = 0x10000u64;
        let m = mini_top();
        // The header and instances in G; the root node's page read besides.
        let root = base + 64..base + 64 + 112;
        let first = |a: u64| if root.contains(&a) { None } else { m.get(&a).copied() };
        let zeros = |a: u64| root.contains(&a).then_some(0u32);
        let (t, v) = top_v3_over(&first, &zeros, &|_| true, base);
        assert_eq!(t.bad_slots, 4, "a root of zeros: four bad slots");
        assert!(matches!(&v, Verdict::NotJudged(why) if why.starts_with("read from title memory")), "{v:?}");
        let mut c = V3::default();
        c.add_top(base, &t, &v);
        assert_eq!((c.tops[1], c.top_unread, c.bad_slots), ([0, 0, 1], [0, 0, 1], 0));
        // The same zeros from the arena (no title page): invalid.
        let (_, v) = top_v3_over(&first, &zeros, &|_| false, base);
        assert!(matches!(&v, Verdict::Invalid(why) if why.contains("slot")), "{v:?}");
        // The built root read besides from the arena: valid.
        let built = |a: u64| m.get(&a).copied();
        assert_eq!(top_v3_over(&first, &built, &|_| false, base).1, Verdict::Valid);
    }

    /// Rehydrated ones with the header one-sided fallback (no 5e687674
    /// dispatch seen) count apart from the rule's (senior 98 point 5b).
    #[test]
    fn v3_counts_the_one_sided_fallback_apart_from_the_rule() {
        let mut page = vec![0u8; 0x1000];
        let put = |page: &mut Vec<u8>, at: usize, w: &[u32]| {
            for (i, x) in w.iter().enumerate() {
                page[at + 4 * i..at + 4 * i + 4].copy_from_slice(&x.to_le_bytes());
            }
        };
        let from = 0x4_1000_0000usize;
        put(&mut page, 0, &bit30_bvh());
        put(&mut page, 0x100, &bit30_bvh());
        let mut garbage = [0u32; 16];
        garbage[0] = 6;
        garbage[1] = 9;
        garbage[3] = 3_212_292_472;
        put(&mut page, 0x200, &garbage);
        note_cb(from as u64, [0f32, 3.0, 0.0].map(f32::to_bits));
        let none = Targets { built: &|_, _| Built::Before(0), next: &|_| None };
        let (v2, v3) = validate(&[(from, &page)], &[from as u64, from as u64 + 0x100, from as u64 + 0x200], false, &none);
        assert_eq!((v2.cb_known, v2.cb_fallback), (1, 1));
        assert_eq!(v3.rule, [1, 0, 0], "the rule's count: cb known only");
        assert_eq!(v3.fallback, [1, 0, 0], "the fallback apart");
        assert_eq!(v3.bottom[bottom_index(Bottom::Rehydrated)], [0, 2, 0], "without the rule");
        assert_eq!((v3.no_top, v3.tops[0]), (1, [0, 0, 0]));
        let text = v3.text();
        assert!(text.contains("with the rule (centre from cb) 1/0/0, header one-sided fallback apart 1/0/0"), "{text}");
        assert!(text.contains("no top level 1"), "{text}");
    }

    #[test]
    fn a_top_level_with_a_packed_instance_passes_v2() {
        let m = mini_top();
        let t = check_top(&read_map(&m), 0x10000);
        assert!(t.valid, "{t:?}");
        assert_eq!((t.instances, t.reached, t.never, t.twice, t.bad_slots), (3, 3, 0, 0, 0));
        assert_eq!(t.on, [1, 1, 1, 0]);
        assert_eq!((t.finite[1], t.finite[2], t.finite[0]), (1, 1, 0));
        assert!(t.bounds_start);
        // The old logic counts the packed instance's start values as invalid.
        assert!(!check(&read_map(&m), 0x10000).valid);
    }

    #[test]
    fn a_top_level_check_finds_twice_never_bad_slots_and_open_instances() {
        let base = 0x10000u64;
        // Slot 2 names instance 0 again: 0 twice, 2 never.
        let mut m = mini_top();
        m.insert(base + 64 + 8, (192 >> 3) | 6);
        let t = check_top(&read_map(&m), base);
        assert!(!t.valid);
        assert_eq!((t.twice, t.never), (1, 1));
        // A pointer of type 7 in slot 3.
        let mut m = mini_top();
        m.insert(base + 64 + 12, 0x1007);
        let t = check_top(&read_map(&m), base);
        assert_eq!(t.bad_slots, 1);
        assert!(t.why.unwrap().contains("slot 3"));
        // An instance slot between two entries.
        let mut m = mini_top();
        m.insert(base + 64 + 4, ((192 + 64) >> 3) | 6);
        let t = check_top(&read_map(&m), base);
        assert_eq!((t.bad_slots, t.never), (1, 1));
        // The rehydrated instance with the start values in its slot box.
        let mut m = mini_top();
        m.insert(base + 64 + 16, f32::MAX.to_bits());
        let t = check_top(&read_map(&m), base);
        assert!(!t.valid);
        assert_eq!(t.finite[1], 0);
        assert!(t.why.unwrap().contains("rehydrated"));
        // The fresh instance's bottom level not read.
        let mut m = mini_top();
        m.remove(&0x20200);
        let t = check_top(&read_map(&m), base);
        assert_eq!(t.blas_unread, 1);
        assert!(!t.valid);
    }

    #[test]
    fn v2_counts_fresh_rehydrated_packed_and_top_apart() {
        let mut page = vec![0u8; 0x1000];
        let put = |page: &mut Vec<u8>, at: usize, w: &[u32]| {
            for (i, x) in w.iter().enumerate() {
                page[at + 4 * i..at + 4 * i + 4].copy_from_slice(&x.to_le_bytes());
            }
        };
        let from = 0x4_0000_0000usize;
        let mut fresh = mini_bvh();
        fresh[0] = 0x8004 << 16 | 6;
        put(&mut page, 0, &fresh);
        put(&mut page, 0x100, &bit30_bvh());
        put(&mut page, 0x200, &mini_packed());
        // cb of the rehydrated one as the title bound it.
        note_cb(from as u64 + 0x100, [0f32, 3.0, 0.0].map(f32::to_bits));
        let v = v2_of(&[(from, &page)], &[from as u64, from as u64 + 0x100, from as u64 + 0x200], false);
        assert_eq!(v.bottom[bottom_index(Bottom::Fresh)], [1, 1, 1]);
        assert_eq!(v.bottom[bottom_index(Bottom::Rehydrated)], [1, 0, 1], "without / with the rule");
        assert_eq!(v.bottom[bottom_index(Bottom::Packed)], [1, 1, 1]);
        assert_eq!((v.cb_known, v.cb_fallback, v.cb_not_predicted), (1, 0, 0));
        assert!(v.text().contains("rehydrated without the bit-30 rule 0/1, with it 1/1"), "{}", v.text());
    }

    #[test]
    fn f16_rounding_far_from_the_origin_is_inside() {
        // Parent x 4000..4001 (f32); the child's f16 max rounds up to 4002
        // (f16 step 2 there): inside, not by a share of the 1 m extent.
        let parent: Aabb = ([4000.0, 0.0, 0.0], [4001.0, 1.0, 1.0]);
        let child: Aabb = ([half(0x6bd0), 0.0, 0.0], [half(0x6bd1), 1.0, 1.0]);
        assert_eq!((child.0[0], child.1[0]), (4000.0, 4002.0));
        assert!(inside(&child, &parent));
        assert!(!inside(&([4000.0, 0.0, 0.0], [4100.0, 1.0, 1.0]), &parent));
    }

    #[test]
    fn a_scan_finds_headers_and_patterns() {
        let mut page = vec![0u8; 1024];
        for (i, w) in mini_bvh().iter().enumerate() {
            page[256 + 4 * i..260 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        page[8..16].copy_from_slice(&0xff7f_ffff_7f7f_ffffu64.to_le_bytes());
        let s = scan(&[(0x10_0000, &page)]);
        assert_eq!(s.headers.len(), 1);
        assert_eq!(s.headers[0].0, 0x10_0100);
        assert!(s.headers[0].1.valid);
        assert_eq!(s.empty16.0, 3, "the root's three empty boxes");
        assert_eq!(s.flt_max, (1, vec![0x10_0008]));
    }

    // v3d (senior 112, Fable Aufruf 8 points 2, 7, 8). Each test is red
    // against c389677 or a mutation of the rule it names (red proof in the
    // JOURNAL entry "senior 112").

    /// The words [`header_past_parents`] reads past `blas`, as zeros where
    /// nothing else lies.
    fn search_room(m: &mut std::collections::HashMap<u64, u32>, blas: u64) {
        for k in 1..=PARENT_STEPS {
            m.entry(blas + 64 * k).or_insert(0);
        }
    }

    /// [`mini_top`] with data instead of a header at its fresh bottom level
    /// 0x20200, the words of the header search there present.
    fn top_without_fresh_header() -> std::collections::HashMap<u64, u32> {
        let mut m = mini_top();
        m.insert(0x20200, 0x1234_5678);
        search_room(&mut m, 0x20200);
        m
    }

    /// Fable Aufruf 8 point 2: each reached instance falls in exactly one of
    /// (a)-(e); only (d) makes the top level invalid, (c) and (e) leave it
    /// not judged, (b) counts as the header it found.
    #[test]
    fn v3d_each_instance_falls_in_one_class() {
        let base = 0x10000u64;
        let classes = |m: &std::collections::HashMap<u64, u32>, built: Built| {
            let (t, v) = check_top_v3_with(&read_map(m), base, &|_| false, &|_| built);
            (t.classes, v, t)
        };
        // (a) all three headers at their addresses.
        let (c, v, _) = classes(&mini_top(), Built::Never);
        assert_eq!((c, v), ([3, 0, 0, 0, 0], Verdict::Valid));
        // (b) the fresh header 0x400 past 0x20200, its parent table there.
        let mut m = top_without_fresh_header();
        m.insert(0x20600, 0x8004_0006);
        m.insert(0x20604, 4);
        m.insert(0x20600 + 56, (-0x400i32) as u32);
        let (c, v, t) = classes(&m, Built::Never);
        assert_eq!((c, &v), ([2, 1, 0, 0, 0], &Verdict::Valid), "{t:?}");
        assert_eq!((t.on[2], t.blas_unread, t.blas_no_header), (1, 0, 0));
        assert!(t.first_b.as_deref().is_some_and(|b| b.contains("header +0x400")), "{:?}", t.first_b);
        // A header there whose parent table starts elsewhere: no (b).
        m.insert(0x20600 + 56, (-0x3c0i32) as u32);
        assert_eq!(classes(&m, Built::Never).0, [2, 0, 0, 0, 1]);
        // (c) built after, (d) built before, (e) never built.
        let m = top_without_fresh_header();
        let (c, v, _) = classes(&m, Built::After(7));
        assert_eq!(c, [2, 0, 1, 0, 0]);
        assert!(matches!(&v, Verdict::NotJudged(w) if w.ends_with("(v3d c)")), "{v:?}");
        let (c, v, _) = classes(&m, Built::Before(7));
        assert_eq!(c, [2, 0, 0, 1, 0]);
        assert!(matches!(&v, Verdict::Invalid(w) if w.ends_with("(v3d d)") && w.contains("after s7")), "{v:?}");
        let (c, v, t) = classes(&m, Built::Never);
        assert_eq!((c, t.never_built.clone()), ([2, 0, 0, 0, 1], vec![0x20200]));
        assert!(matches!(&v, Verdict::NotJudged(w) if w.ends_with("(v3d e)")), "{v:?}");
        // Counted per class, the top levels apart by why they are not judged.
        let mut s = V3::default();
        for built in [Built::After(1), Built::Before(1), Built::Never] {
            let (_, v, t) = classes(&m, built);
            s.add_top(base, &t, &v);
        }
        assert_eq!(
            (s.classes, s.order_tops, s.tops[1], s.top_unread, s.never_built.clone()),
            ([6, 0, 1, 1, 1], [1, 1], [0, 1, 2], [0; 3], vec![0x20200])
        );
        assert!(
            s.text().contains("never a build target 6/0/1/1/1 (header search not read 0); top not judged for c 1, for e 1"),
            "{}",
            s.text()
        );
    }

    /// (d) only with the header search complete: a word past the parent
    /// table not read leaves the top level not judged.
    #[test]
    fn v3d_a_header_search_not_read_is_not_judged_not_d() {
        let base = 0x10000u64;
        let mut m = mini_top();
        m.insert(0x20200, 0x1234_5678);
        let (t, v) = check_top_v3_with(&read_map(&m), base, &|_| false, &|_| Built::Before(0));
        assert!(matches!(&v, Verdict::NotJudged(w) if w.contains("search past its parent table")), "{v:?}");
        assert_eq!((t.classes, t.search_unread, t.blas_unread), ([2, 0, 0, 0, 0], 1, 1));
    }

    /// The build map: binds in record order against the top level's last
    /// bind; addresses that are no 256-byte multiple are no build target; an
    /// (e) bottom level bound later leaves the pending list.
    #[test]
    fn v3d_the_build_map_orders_binds() {
        let mut b = Builds { seq: 0, map: std::collections::BTreeMap::new(), full: 0, taken: 0, pending: Vec::new(), later: 0, dropped: 0 };
        let (top, early, late, never) = (0x1_0000_0000u64, 0x2_0000_0100, 0x2_0000_0200, 0x2_0000_0300);
        b.taken = 5;
        note_build_in(&mut b, early);
        note_build_in(&mut b, top);
        b.taken = 6;
        note_build_in(&mut b, late);
        assert_eq!(built_of(&b, top, early), Built::Before(5));
        assert_eq!(built_of(&b, top, late), Built::After(6));
        assert_eq!(built_of(&b, top, never), Built::Never);
        // The top level built again: the bottom level is before its last build.
        note_build_in(&mut b, top);
        assert_eq!(built_of(&b, top, late), Built::Before(6));
        note_build_in(&mut b, 0x2_0000_0380);
        assert_eq!(built_of(&b, top, 0x2_0000_0380), Built::Never);
        let seq = b.seq;
        b.pending = vec![(never, seq), (0x2_0000_0400, seq)];
        resolve_pending(&mut b);
        assert_eq!((b.later, b.pending.len()), (0, 2));
        note_build_in(&mut b, never);
        resolve_pending(&mut b);
        assert_eq!((b.later, b.pending.clone()), (1, vec![(0x2_0000_0400, seq)]));
        // The binds the game notes: 5552dac6 ud16/17, 5e687674 ud35/36.
        let mut values = vec![0u32; 40];
        values[16..18].copy_from_slice(&[0x7a56_9000, 0xabc]);
        values[35..37].copy_from_slice(&[0x4e5b_1100, 0xabd]);
        note_bind("[\"cs 5552dac6c0ffee00\"]", &values);
        note_bind("[\"cs 5e68767498c57eb6\"]", &values);
        note_bind("[\"cs 6f6a359d00000000\"]", &[0x7a56_a000, 0xabc].repeat(20));
        let g = builds();
        assert!(g.map.contains_key(&0xabc_7a56_9000) && g.map.contains_key(&0xabd_4e5b_1100));
        assert!(!g.map.contains_key(&0xabc_7a56_a000), "no build target stage");
    }

    /// Fable Aufruf 8 point 8, once: the gap to the next build target
    /// bounds the title's output buffer; D1 * 64 + 64 past it is counted.
    #[test]
    fn v3d_the_gap_to_the_next_build_target_is_counted() {
        let mut page = vec![0u8; 0x1000];
        let mut fresh = mini_bvh();
        fresh[0] = 0x8004 << 16 | 6;
        for (i, x) in fresh.iter().enumerate() {
            page[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
        let from = 0x4_0000_0000u64;
        let gap = |next: Option<u64>| {
            let next_of = |_: u64| next.map(|n| from + n);
            validate(&[(from as usize, &page[..])], &[from], false, &Targets { built: &|_, _| Built::Before(0), next: &next_of }).1
        };
        // D1 * 64 = 0x100.
        assert_eq!(gap(Some(0x140)).gap, [1, 0, 0]);
        let v = gap(Some(0x120));
        assert_eq!(v.gap, [1, 1, 0]);
        assert!(v.first_v3d[0].contains("gap 0x120"), "{:?}", v.first_v3d);
        assert_eq!(gap(Some(0xc0)).gap, [1, 1, 1]);
        assert_eq!(gap(None).gap, [0, 0, 0]);
    }

    /// Fable Aufruf 8 point 7: dword 15 bits 24-29 set is not judged, not
    /// valid (the bit-30 rule says nothing about them).
    #[test]
    fn v3d_dword_15_bits_24_to_29_are_not_judged() {
        let base = 0x3000;
        let mut w = bit30_bvh();
        let cb = Some([0f32, 3.0, 0.0].map(f32::to_bits));
        assert_eq!(check_bottom_v3(&read_of(&w, base), base, Bottom::Rehydrated, cb).with_rule, Verdict::Valid);
        for bit in [24, 29] {
            w[15] = 0xc080_8080 | 1 << bit;
            let c = check_bottom_v3(&read_of(&w, base), base, Bottom::Rehydrated, cb);
            assert!(matches!(&c.with_rule, Verdict::NotJudged(x) if x.starts_with("dword 15")), "bit {bit}: {c:?}");
            let mut s = V3::default();
            s.add_bottom(base, Bottom::Rehydrated, &c);
            assert_eq!((s.d15_unknown, s.rule), (1, [0, 0, 1]));
        }
    }

    /// Fable Aufruf 8 point 8: the node end as a rule by kind. 0x8002 /
    /// 0x8004: a node starting at D1 * 64 is invalid (box16, ends at
    /// D1 * 64 + 64; box32, ends past it), one at D1 * 64 - 64 valid; other
    /// bottom levels keep v3b's rule (a box16 at D1 * 64 valid).
    #[test]
    fn v3d_the_node_end_is_a_rule_by_kind() {
        let base = 0x5000;
        let with_child = |mark: u32, at: usize, box32: bool| {
            let mut w = mini_bvh().to_vec();
            w.resize(100, 0);
            w[0] = mark << 16 | 6;
            if box32 {
                w[at / 4..at / 4 + 28].copy_from_slice(&box32_root());
            } else {
                let node: Vec<u32> = w[16..32].to_vec();
                w[at / 4..at / 4 + 16].copy_from_slice(&node);
            }
            w[16] = (at as u32 >> 3) | if box32 { 5 } else { 4 };
            check_v3(&read_of(&w, base), base, Centre::Bit30AllAxes)
        };
        for mark in [0x8002u32, 0x8004] {
            assert!(matches!(with_child(mark, 256, false), Verdict::Invalid(w) if w.contains("past the size")), "{mark:#x} box16 at D1*64");
            assert!(matches!(with_child(mark, 256, true), Verdict::Invalid(_)), "{mark:#x} box32 at D1*64");
            assert_eq!(with_child(mark, 192, false), Verdict::Valid, "{mark:#x} box16 at D1*64-64");
        }
        assert_eq!(with_child(0x8000, 256, false), Verdict::Valid, "other: v3b rule");
    }
}
