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

/// `VICTORY_BVH_ROOTS_LOG=1` (H-02 intersection, night 2): after each
/// submission with builder dispatches, one line with the root views 0 and 4
/// they read: pipeline, slot, address, and bytes from it to the end of its
/// title buffer (the range a root view refreshes and reaches, `crate::draw`;
/// 0 if no buffer holds it). With the memclass `seq` of the submission (0
/// with memclass off), to meet its `gpu_w_lost` records.
/// `VICTORY_BVH_ROOTS_MAX` (2000): the most lines in a run.
pub(crate) fn roots_log() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_ROOTS_LOG").is_ok_and(|v| v.trim() == "1"))
}

static ROOT_LINES: AtomicU64 = AtomicU64::new(0);

/// The line of [`roots_log`] for host submission `submission`.
pub(crate) fn roots(submission: u64, seq: u64, roots: &[(String, u8, u64, u64)]) {
    if roots.is_empty() || ROOT_LINES.fetch_add(1, Ordering::Relaxed) >= number("VICTORY_BVH_ROOTS_MAX", 2000) {
        return;
    }
    tracing::info!(target: "victory_gpu::bvh_gt", "{}", roots_line(submission, seq, roots));
}

fn roots_line(submission: u64, seq: u64, roots: &[(String, u8, u64, u64)]) -> String {
    use std::fmt::Write as _;
    let mut line = format!("BVH roots s{submission} seq {seq}: {} views", roots.len());
    for (pipeline, slot, at, len) in roots {
        let _ = write!(line, "; [{pipeline}] slot {slot} at {at:#x} len {len:#x}");
    }
    line
}

#[derive(Default)]
struct Last {
    submission: u64,
    headers: Vec<u64>,
    /// Copy (none: an arena page), page, where its bytes start.
    pages: Vec<(Option<Arc<Shadow>>, usize, usize)>,
}

static LAST: std::sync::Mutex<Last> = std::sync::Mutex::new(Last { submission: 0, headers: Vec::new(), pages: Vec::new() });
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
    for &at in &last.headers {
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
        // Remembered for the readers' check ([`reader_check`]): the pages
        // with headers, where they live.
        if reader_target().is_some() && !g.headers.is_empty() {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            last.headers = g.headers.iter().map(|(at, _)| *at).collect();
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
fn reader<'a, B: AsRef<[u8]>>(pages: &'a [(usize, B)]) -> impl Fn(u64) -> Option<u32> + 'a {
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
    }
    let bytes = u64::from(h[1]) * 64 + extra;
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
    let mut queue = vec![(root, bounds, true)];
    let mut seen = std::collections::HashSet::new();
    while let Some((p, bounds, is_root)) = queue.pop() {
        let kind = p & 7;
        if !(kind == 4 || kind == 5) || at(p) == 0 || at(p) >= bytes || !seen.insert(p) || seen.len() > 4096 {
            continue;
        }
        let n = if kind == 4 { 16 } else { 28 };
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

    /// The roots line names submission, seq and each view, as the
    /// intersection script reads it (`scripts/h02/intersect.py`).
    #[test]
    fn the_roots_line_has_every_view() {
        let roots = [("compute 5552dac6abc".to_owned(), 0u8, 0xd722_7a400u64, 0x1000u64), ("compute 5552dac6abc".to_owned(), 4, 0xd722_7a440, 0)];
        assert_eq!(
            roots_line(17, 3, &roots),
            "BVH roots s17 seq 3: 2 views; [compute 5552dac6abc] slot 0 at 0xd7227a400 len 0x1000; [compute 5552dac6abc] slot 4 at 0xd7227a440 len 0x0"
        );
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
}
