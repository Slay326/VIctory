//! Replayer of memclass event streams against the host copies (night 2,
//! F4 tool b, stream 4b): offline, no game, nothing runs on the GPU.
//!
//! It reads an event stream in memclass format v1 (`docs/memclass-format.md`)
//! and plays it against the real copies of title memory ([`Shadow`], with
//! `crate::watch`'s page tracking and guards): each classified region gets
//! a fresh write-watched pool of its size and a copy of it. Title (CPU)
//! writes are real writes into the pool (a guarded page faults into the
//! handler, as a title thread's write); shaders' (GPU) writes are written
//! into the copy, as the tests in `shadow::tests` do; refreshes, declared
//! ranges and copy-backs are the copies' own calls. A byte model (what one
//! unified memory would hold) checks the three invariants:
//!
//! 1. A page the CPU wrote is in the copy before any GPU read of it (and a
//!    byte the GPU wrote is not lost from the copy before it reads it again).
//! 2. A page the GPU wrote is in title memory at the sync point (the end of
//!    the submission) where it is copied back, and before any CPU touch of
//!    the page where it is not (memory treated as GPU-only).
//! 3. A demoted page is never treated as GPU-only again.
//!
//! What is mirrored and copied back is decided by a [`MemModel`]: that is
//! where the stage 3 logic (`VICTORY_MEMMODEL`, the safety net Fable writes)
//! hooks in. [`Today`] is the engine as it is (everything mirrored, every
//! GPU write copied back). [`NoNet`] and [`RefNet`] are TEST MODELS, NOT
//! PRODUCTION CODE: GPU-only without and with a reference safety net, to
//! show that a wrong classification turns the replay red.
//!
//! Input: the stream and a classification, regions with a class
//! ([`parse_classes`]). Output ([`Replay::report`]): per region the events,
//! contradictions to its class, demotions with reasons and violations per
//! invariant; then a verdict. The report holds stream addresses only (never
//! the pools'), so two replays of one stream give the same text.
//!
//! Limit (F4): a replay can refute a classification, never prove it; the
//! safety net and a game run stay required.
//!
//! Shortcuts (v1):
//! - Regions are address ranges; a `section` region of the table has no
//!   address and must be given by its views' ranges. Records outside every
//!   classified region are counted, not replayed. Views of one section are
//!   not aliased (each range its own pool). `REGION` and `remap` records are
//!   counted only.
//! - Bytes are sampled: a CPU record writes its first byte and one byte per
//!   64 KiB page it covers; a GPU write record (`gpu_w_back`, `gpu_w_copy`)
//!   writes its first and last byte and one per page. `gpu_w_mark` and
//!   `gpu_w_kept` are counted only (the bytes that went back are the
//!   `gpu_w_back` records of the same copy-back).
//! - A byte the CPU and the GPU both write within one submission is a race
//!   in the title too: it is not checked.
//! - GPU writes of a submission are made when their records come (after
//!   `gpu_done`); copy-backs run at `POINT end`: declared ranges by their
//!   snapshot ([`Shadow::copy_back`]), other written pages as written at
//!   computed addresses ([`Shadow::copy_back_page`]).
//! - Copies are kept in system memory (not [`cards`]).
//! - Pools are not freed (as `shadow::tests`); `max_bytes` bounds them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::*;
use crate::memclass::Class;

// ------------------------------------------------------------ the stream ---

/// A record of a memclass v1 stream (`docs/memclass-format.md`).
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // every field read, not every one used by the replay
pub(crate) enum Rec {
    Access {
        flag: u16,
        tid: u64,
        seq: u64,
        at: u64,
        len: u64,
    },
    Submit {
        seq: u64,
        lists: u64,
    },
    Point {
        kind: u8,
        seq: u64,
    },
    Sync {
        kind: u8,
        at: u64,
        len: u64,
        us: u64,
    },
    Frame(u64),
    Region {
        kind: u8,
        v: [u64; 4],
    },
    End(u64),
    Counters([u64; 4]),
}

/// A stream read: the unit shift of the header, the records, and whether it
/// ended with `END` (else it was cut off: the whole records up to the cut).
pub(crate) struct Stream {
    pub(crate) shift: u8,
    pub(crate) records: Vec<Rec>,
    pub(crate) ended: bool,
}

/// Reads a memclass stream; refuses another magic or version.
pub(crate) fn read_stream(data: &[u8]) -> Result<Stream, String> {
    use crate::memclass::{STREAM_MAGIC, STREAM_VERSION, rec};
    if data.len() < 8 || &data[..4] != STREAM_MAGIC {
        return Err("not a memclass stream".into());
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != STREAM_VERSION {
        return Err(format!(
            "stream version {version}, the replayer knows {STREAM_VERSION}"
        ));
    }
    let shift = data[6];
    let mut i = 8;
    let var = |i: &mut usize| -> Option<u64> {
        let (mut v, mut s) = (0u64, 0u32);
        loop {
            let b = *data.get(*i)?;
            *i += 1;
            if s < 64 {
                v |= u64::from(b & 0x7f) << s;
            }
            s += 7;
            if b < 0x80 {
                return Some(v);
            }
        }
    };
    let mut records = Vec::new();
    let mut last_at = 0u64;
    let mut ended = false;
    while i < data.len() {
        let start = i;
        let mut read = || -> Option<Result<Rec, String>> {
            let tag = var(&mut i)?;
            let r = match tag as u8 {
                rec::ACCESS => {
                    let (flag, tid, seq, z, len) = (
                        var(&mut i)?,
                        var(&mut i)?,
                        var(&mut i)?,
                        var(&mut i)?,
                        var(&mut i)?,
                    );
                    let d = ((z >> 1) as i64) ^ -((z & 1) as i64);
                    let at = last_at.wrapping_add(d as u64);
                    last_at = at;
                    Rec::Access {
                        flag: flag as u16,
                        tid,
                        seq,
                        at,
                        len,
                    }
                }
                rec::SUBMIT => Rec::Submit {
                    seq: var(&mut i)?,
                    lists: var(&mut i)?,
                },
                rec::POINT => Rec::Point {
                    kind: var(&mut i)? as u8,
                    seq: var(&mut i)?,
                },
                rec::SYNC => Rec::Sync {
                    kind: var(&mut i)? as u8,
                    at: var(&mut i)?,
                    len: var(&mut i)?,
                    us: var(&mut i)?,
                },
                rec::FRAME => Rec::Frame(var(&mut i)?),
                rec::REGION => Rec::Region {
                    kind: var(&mut i)? as u8,
                    v: [var(&mut i)?, var(&mut i)?, var(&mut i)?, var(&mut i)?],
                },
                rec::END => Rec::End(var(&mut i)?),
                rec::COUNTERS => {
                    Rec::Counters([var(&mut i)?, var(&mut i)?, var(&mut i)?, var(&mut i)?])
                }
                _ => return Some(Err(format!("unknown record {tag} at byte {start}"))),
            };
            Some(Ok(r))
        };
        match read() {
            None => break, // cut off mid-record
            Some(Err(e)) => return Err(e),
            Some(Ok(r)) => {
                let end = matches!(r, Rec::End(_));
                records.push(r);
                if end {
                    ended = true;
                    break;
                }
            }
        }
    }
    Ok(Stream {
        shift,
        records,
        ended,
    })
}

// ------------------------------------------------------ the classification ---

/// A classified region: stream (title) addresses `[at, at + len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClassRegion {
    pub(crate) at: u64,
    pub(crate) len: u64,
    pub(crate) class: Class,
}

fn class_name(c: Class) -> &'static str {
    match c {
        Class::GpuOnly => "gpu-only",
        Class::CpuOnly => "cpu-only",
        Class::Shared => "shared",
        Class::Unknown => "unknown",
    }
}

/// A classification, a region per line: `<at> <len> <class>` or
/// `at=<at> len=<len> class=<class>` (other `key=value`s are ignored, so a
/// `memclass region` line with a class added reads too). Numbers hex with
/// `0x` or decimal; classes `gpu-only`, `cpu-only`, `shared`, `unknown`;
/// `#` starts a comment. Regions must not overlap.
pub(crate) fn parse_classes(text: &str) -> Result<Vec<ClassRegion>, String> {
    let num = |s: &str| -> Result<u64, String> {
        let r = match s.strip_prefix("0x") {
            Some(h) => u64::from_str_radix(h, 16),
            None => s.parse(),
        };
        r.map_err(|_| format!("not a number: {s:?}"))
    };
    let class = |s: &str| -> Result<Class, String> {
        Ok(match s {
            "gpu-only" | "gpu_only" => Class::GpuOnly,
            "cpu-only" | "cpu_only" => Class::CpuOnly,
            "shared" => Class::Shared,
            "unknown" => Class::Unknown,
            _ => return Err(format!("unknown class {s:?}")),
        })
    };
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let (mut at, mut len, mut cl) = (None, None, None);
        if words.iter().any(|w| w.contains('=')) {
            for w in &words {
                match w.split_once('=') {
                    Some(("at", v)) => at = Some(num(v)?),
                    Some(("len", v)) => len = Some(num(v)?),
                    Some(("class", v)) => cl = Some(class(v)?),
                    _ => {}
                }
            }
        } else if let [a, l, c] = words[..] {
            (at, len, cl) = (Some(num(a)?), Some(num(l)?), Some(class(c)?));
        }
        match (at, len, cl) {
            (Some(at), Some(len), Some(class)) if len > 0 => {
                out.push(ClassRegion { at, len, class })
            }
            _ => {
                return Err(format!(
                    "line {}: want `<at> <len> <class>`: {line:?}",
                    n + 1
                ));
            }
        }
    }
    out.sort_by_key(|r| r.at);
    for w in out.windows(2) {
        if w[0].at + w[0].len > w[1].at {
            return Err(format!("regions {:#x} and {:#x} overlap", w[0].at, w[1].at));
        }
    }
    Ok(out)
}

// ------------------------------------------------------------ the model ---

/// A 64 KiB page of a classified region: the region's index and the page's
/// stream (title) address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Page {
    pub(crate) region: usize,
    pub(crate) at: u64,
}

/// Who the CPU writer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CpuKind {
    /// The title's own thread (`cpu_w_title`: a guard fault).
    Title,
    /// The runtime for the title (`cpu_w_runtime`: file reads, clearing).
    Runtime,
}

/// What the model does when the CPU is about to touch a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Touch {
    /// Nothing: title memory is current (copied back at each sync point).
    Nothing,
    /// The page's GPU bytes go into title memory first, the class stays
    /// (for models that sync without demoting; the test models do not).
    #[allow(dead_code)]
    Sync,
    /// Synced as `Sync`, and demoted to shared for good (the reason is
    /// logged and counted per region).
    Demote(&'static str),
}

/// The memory model's decisions, per page: where the stage 3 logic
/// (`VICTORY_MEMMODEL`, the safety net) hooks into the replay. The replayer
/// does what the model answers with the real copies and checks the
/// invariants; it never decides itself.
///
/// For the stage 3 implementation: implement this for (an adapter around)
/// the production decision code, replay a capture with it, and the replay
/// must stay green (invariants) with demotions only where the
/// classification was wrong.
pub(crate) trait MemModel {
    /// For the report.
    fn name(&self) -> String;
    /// The page's class now (after demotions).
    fn class(&self, page: Page) -> Class;
    /// Before a GPU read: is the copy brought up to date from title memory?
    fn refresh(&self, page: Page) -> bool;
    /// At the sync point: do the GPU's writes in the page go back into
    /// title memory?
    fn copy_back(&self, page: Page) -> bool;
    /// The CPU is about to write the page (detected by the guard or the
    /// runtime's report).
    fn cpu_touch(&mut self, page: Page, kind: CpuKind) -> Touch;
    /// A frame ended (a model may reclassify here).
    fn frame(&mut self, _frame: u64) {}
}

/// The engine as it is: every page mirrored and copied back, the
/// classification ignored.
pub(crate) struct Today;

impl MemModel for Today {
    fn name(&self) -> String {
        "today".into()
    }
    fn class(&self, _: Page) -> Class {
        Class::Shared
    }
    fn refresh(&self, _: Page) -> bool {
        true
    }
    fn copy_back(&self, _: Page) -> bool {
        true
    }
    fn cpu_touch(&mut self, _: Page, _: CpuKind) -> Touch {
        Touch::Nothing
    }
}

/// TEST MODEL, NOT PRODUCTION CODE: GPU-only regions are neither mirrored
/// nor copied back, and nothing watches them (no safety net). Shows what a
/// wrong classification costs without the net: violations.
pub(crate) struct NoNet {
    pub(crate) classes: Vec<Class>,
}

impl MemModel for NoNet {
    fn name(&self) -> String {
        "nonet (test model)".into()
    }
    fn class(&self, p: Page) -> Class {
        self.classes[p.region]
    }
    fn refresh(&self, p: Page) -> bool {
        self.classes[p.region] != Class::GpuOnly
    }
    fn copy_back(&self, p: Page) -> bool {
        self.classes[p.region] != Class::GpuOnly
    }
    fn cpu_touch(&mut self, _: Page, _: CpuKind) -> Touch {
        Touch::Nothing
    }
}

/// TEST MODEL, NOT PRODUCTION CODE (the stage 3 safety net is Fable's): a
/// reference for the replay's tests. GPU-only regions are neither mirrored
/// nor copied back; a CPU touch of such a page syncs it and demotes it to
/// shared for good. `forget_on_frame` is a planted fault for the tests: the
/// demotions are forgotten at each frame (a reclassification that undoes
/// them), which invariant 3 must catch.
pub(crate) struct RefNet {
    pub(crate) classes: Vec<Class>,
    pub(crate) demoted: BTreeSet<Page>,
    pub(crate) forget_on_frame: bool,
}

impl RefNet {
    pub(crate) fn new(classes: Vec<Class>) -> RefNet {
        RefNet {
            classes,
            demoted: BTreeSet::new(),
            forget_on_frame: false,
        }
    }
    fn gpu_only(&self, p: Page) -> bool {
        self.classes[p.region] == Class::GpuOnly && !self.demoted.contains(&p)
    }
}

impl MemModel for RefNet {
    fn name(&self) -> String {
        format!(
            "refnet (test model){}",
            if self.forget_on_frame {
                ", forgets demotions"
            } else {
                ""
            }
        )
    }
    fn class(&self, p: Page) -> Class {
        if self.gpu_only(p) {
            Class::GpuOnly
        } else if self.classes[p.region] == Class::GpuOnly {
            Class::Shared
        } else {
            self.classes[p.region]
        }
    }
    fn refresh(&self, p: Page) -> bool {
        !self.gpu_only(p)
    }
    fn copy_back(&self, p: Page) -> bool {
        !self.gpu_only(p)
    }
    fn cpu_touch(&mut self, p: Page, kind: CpuKind) -> Touch {
        if !self.gpu_only(p) {
            return Touch::Nothing;
        }
        self.demoted.insert(p);
        Touch::Demote(match kind {
            CpuKind::Title => "title write",
            CpuKind::Runtime => "runtime write",
        })
    }
    fn frame(&mut self, _: u64) {
        if self.forget_on_frame {
            self.demoted.clear();
        }
    }
}

// ----------------------------------------------------------- the replay ---

const KIB64: u64 = PAGE as u64;

/// Counts per region.
#[derive(Default, Clone)]
struct Counts {
    cpu_title: u64,
    cpu_runtime: u64,
    gpu_read: u64,
    gpu_decl: u64,
    gpu_write: u64,
    other: u64,
    /// Records whose kind the region's class says cannot happen there (a
    /// CPU write to GPU-only memory, a GPU access to CPU-only memory).
    contradictions: u64,
    demotions: BTreeMap<&'static str, u64>,
    violations: [u64; 3],
    checks: [u64; 3],
    refreshed_pages: u64,
    unrefreshed_pages: u64,
    synced_pages: u64,
    sync_up: u64,
    sync_back: u64,
}

struct Region {
    class: ClassRegion,
    /// The pool and its copy; none where the pools' budget was spent.
    p: usize,
    copy: Option<Arc<Shadow>>,
    /// Bytes anyone wrote, by offset: their value in one unified memory
    /// (all others are 0, as the fresh pool).
    truth: BTreeMap<u64, u8>,
    counts: Counts,
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.copy.take().is_some() {
            unregister(self.p, self.p as u64);
        }
    }
}

/// A broken invariant: which, where (stream address), when.
struct Violation {
    inv: usize,
    region: usize,
    at: u64,
    seq: u64,
    frame: u64,
    what: String,
}

/// The replay of one stream with one model.
pub(crate) struct Replay<M: MemModel> {
    pub(crate) model: M,
    regions: Vec<Region>,
    shift: u8,
    records: u64,
    outside: BTreeMap<&'static str, u64>,
    unreplayed: u64,
    submissions: u64,
    frame: u64,
    seq: u64,
    /// Values generated so far (deterministic byte values).
    n: u64,
    /// The open submission: declared ranges (region, offsets), the GPU's
    /// bytes written in it, the bytes the CPU wrote in it.
    decls: BTreeSet<(usize, u64, u64)>,
    decls_done: BTreeSet<(usize, u64, u64)>,
    sub_gpu: BTreeMap<(usize, u64), u8>,
    sub_cpu: BTreeSet<(usize, u64)>,
    demoted: BTreeSet<Page>,
    demotions: Vec<(Page, &'static str, u64, u64)>,
    violations: Vec<Violation>,
    /// Invariant 3 at frame ends: pages already reported (once each).
    inv3_reported: BTreeSet<Page>,
}

const SHOWN: usize = 20;

impl<M: MemModel> Replay<M> {
    /// Pools and copies for the regions (at most `max_bytes` of pools; the
    /// rest is not replayed and counted). None without a host GPU (no
    /// copies).
    pub(crate) fn new(classes: &[ClassRegion], model: M, max_bytes: u64) -> Option<Replay<M>> {
        crate::host::host()?;
        let mut regions = Vec::new();
        let mut used = 0u64;
        for &c in classes {
            let len = c.len.next_multiple_of(KIB64);
            let mut r = Region {
                class: c,
                p: 0,
                copy: None,
                truth: BTreeMap::new(),
                counts: Counts::default(),
            };
            if used + len <= max_bytes {
                used += len;
                let p = crate::watch::tests::pool((len / KIB64) as usize);
                register(p, p as u64, len);
                // In system memory: the copies' GPU sides are not simulated.
                super::tests::LEAST.with(|l| l.set(Some(u64::MAX)));
                let copy = find(p as u64, len);
                super::tests::LEAST.with(|l| l.set(None));
                let Some(copy) = copy else {
                    unregister(p, p as u64);
                    return None;
                };
                // Taken: guarded and clean, as memory the title has used.
                copy.update(p, len as usize);
                r.p = p;
                r.copy = Some(copy);
            }
            regions.push(r);
        }
        Some(Replay {
            model,
            regions,
            shift: 16,
            records: 0,
            outside: BTreeMap::new(),
            unreplayed: 0,
            submissions: 0,
            frame: 0,
            seq: 0,
            n: 0,
            decls: BTreeSet::new(),
            decls_done: BTreeSet::new(),
            sub_gpu: BTreeMap::new(),
            sub_cpu: BTreeSet::new(),
            demoted: BTreeSet::new(),
            demotions: Vec::new(),
            violations: Vec::new(),
            inv3_reported: BTreeSet::new(),
        })
    }

    /// `[at, at + len)` cut to each region it overlaps: (region, offsets).
    fn clip(&self, at: u64, len: u64) -> Vec<(usize, u64, u64)> {
        let end = at.saturating_add(len.max(1));
        let first = self
            .regions
            .partition_point(|r| r.class.at + r.class.len <= at);
        let mut out = Vec::new();
        for (i, r) in self.regions.iter().enumerate().skip(first) {
            if r.class.at >= end {
                break;
            }
            let (from, to) = (at.max(r.class.at), end.min(r.class.at + r.class.len));
            if from < to {
                out.push((i, from - r.class.at, to - r.class.at));
            }
        }
        out
    }

    fn page(&self, region: usize, off: u64) -> Page {
        Page {
            region,
            at: (self.regions[region].class.at + off) & !(KIB64 - 1),
        }
    }

    /// The pages `[from, to)` (offsets) touches.
    fn pages(&self, region: usize, from: u64, to: u64) -> Vec<Page> {
        let base = self.regions[region].class.at;
        let first = (base + from) & !(KIB64 - 1);
        (0..)
            .map(|k| first + k * KIB64)
            .take_while(|&a| a < base + to)
            .map(|at| Page { region, at })
            .collect()
    }

    /// The bytes a record writes in `[from, to)`: its first, one per page,
    /// and its last (`ends`).
    fn samples(&self, region: usize, from: u64, to: u64, ends: bool) -> Vec<u64> {
        let base = self.regions[region].class.at;
        let mut out: Vec<u64> = self
            .pages(region, from, to)
            .iter()
            .map(|p| (p.at - base).max(from))
            .collect();
        if ends && to - 1 != from {
            out.push(to - 1);
        }
        out.dedup();
        out
    }

    fn next_value(&mut self, old: u8) -> u8 {
        self.n += 1;
        old.wrapping_add(1 + (self.n % 251) as u8)
    }

    fn title(&self, r: usize, off: u64) -> u8 {
        // SAFETY: a byte of the region's pool (read-only guards allow reads).
        unsafe { ((self.regions[r].p + off as usize) as *const u8).read_volatile() }
    }

    fn copy_byte(&self, r: usize, off: u64) -> u8 {
        let reg = &self.regions[r];
        let at = reg.p as u64 + off;
        reg.copy.as_ref().expect("replayed").read_host(at & !3, 1)[0].to_le_bytes()
            [(at % 4) as usize]
    }

    fn truth(&self, r: usize, off: u64) -> u8 {
        self.regions[r].truth.get(&off).copied().unwrap_or(0)
    }

    fn violate(&mut self, inv: usize, region: usize, off: u64, what: String) {
        self.regions[region].counts.violations[inv - 1] += 1;
        let at = self.regions[region].class.at + off;
        // The frame in progress (FRAME n ends frame n).
        self.violations.push(Violation {
            inv,
            region,
            at,
            seq: self.seq,
            frame: self.frame + 1,
            what,
        });
    }

    /// Invariant 3: a demoted page asked about must not be answered as
    /// GPU-only.
    fn check_demoted(&mut self, p: Page, gpu_only: bool, asked: &str) {
        if self.demoted.contains(&p) {
            self.regions[p.region].counts.checks[2] += 1;
            if gpu_only {
                let off = p.at - self.regions[p.region].class.at;
                self.violate(
                    3,
                    p.region,
                    off,
                    format!("demoted page treated as GPU-only again ({asked})"),
                );
            }
        }
    }

    /// Plays every record; then [`Replay::report`].
    pub(crate) fn run(&mut self, stream: &Stream) {
        self.shift = stream.shift;
        for r in &stream.records {
            self.records += 1;
            self.step(r);
        }
    }

    fn step(&mut self, rec: &Rec) {
        use crate::memclass::f;
        match *rec {
            Rec::Access {
                flag, seq, at, len, ..
            } => {
                self.seq = self.seq.max(seq);
                let parts = self.clip(at, len);
                let kind = match flag {
                    f::CPU_W_TITLE => "cpu_w_title",
                    f::CPU_W_RUNTIME => "cpu_w_runtime",
                    f::GPU_R_REFRESH | f::GPU_R_HEAP | f::GPU_R_TABLE => "gpu_r",
                    f::GPU_W_DECL => "gpu_w_decl",
                    f::GPU_W_BACK | f::GPU_W_COPY => "gpu_w",
                    _ => "other",
                };
                if parts.is_empty() {
                    *self.outside.entry(kind).or_default() += 1;
                    return;
                }
                for (r, from, to) in parts {
                    if self.regions[r].copy.is_none() {
                        self.unreplayed += 1;
                        continue;
                    }
                    match flag {
                        f::CPU_W_TITLE => self.cpu_write(r, from, to, CpuKind::Title),
                        f::CPU_W_RUNTIME => self.cpu_write(r, from, to, CpuKind::Runtime),
                        f::GPU_R_REFRESH | f::GPU_R_HEAP | f::GPU_R_TABLE => {
                            self.gpu_read(r, from, to)
                        }
                        f::GPU_W_DECL => self.gpu_decl(r, from, to),
                        f::GPU_W_BACK | f::GPU_W_COPY => self.gpu_write(r, from, to),
                        _ => {
                            self.regions[r].counts.other += 1;
                            if flag & (f::GPU_W_MARK | f::GPU_W_KEPT) != 0
                                && self.regions[r].class.class == Class::CpuOnly
                            {
                                self.regions[r].counts.contradictions += 1;
                            }
                        }
                    }
                }
            }
            Rec::Submit { seq, .. } => {
                self.seq = seq;
                self.submissions += 1;
                self.decls.clear();
                self.sub_gpu.clear();
                self.sub_cpu.clear();
            }
            Rec::Point { kind: 2, seq } => {
                self.seq = self.seq.max(seq);
                self.sync_point();
            }
            Rec::Point { .. } => {}
            Rec::Sync { kind, at, len, .. } => {
                for (r, from, to) in self.clip(at, len) {
                    let c = &mut self.regions[r].counts;
                    if kind == 0 {
                        c.sync_up += to - from
                    } else {
                        c.sync_back += to - from
                    }
                }
            }
            Rec::Frame(n) => {
                self.frame = n;
                self.model.frame(n);
                // Invariant 3 also between events: a demoted page's class.
                for p in self.demoted.clone() {
                    self.regions[p.region].counts.checks[2] += 1;
                    if self.model.class(p) == Class::GpuOnly && self.inv3_reported.insert(p) {
                        let off = p.at - self.regions[p.region].class.at;
                        self.violate(
                            3,
                            p.region,
                            off,
                            "demoted page classed GPU-only again after a frame".into(),
                        );
                    }
                }
            }
            Rec::Region { .. } | Rec::End(_) | Rec::Counters(_) => {}
        }
    }

    fn cpu_write(&mut self, r: usize, from: u64, to: u64, kind: CpuKind) {
        let c = &mut self.regions[r].counts;
        match kind {
            CpuKind::Title => c.cpu_title += 1,
            CpuKind::Runtime => c.cpu_runtime += 1,
        }
        if self.regions[r].class.class == Class::GpuOnly {
            self.regions[r].counts.contradictions += 1;
        }
        let samples = self.samples(r, from, to, false);
        let mut touched = BTreeSet::new();
        for off in samples {
            let p = self.page(r, off);
            if touched.insert(p) {
                self.touch(p, kind);
            }
            let v = self.next_value(self.truth(r, off));
            // SAFETY: a byte of the region's pool (a guarded page faults into
            // the handler, as a title thread's write).
            unsafe { ((self.regions[r].p + off as usize) as *mut u8).write_volatile(v) };
            self.regions[r].truth.insert(off, v);
            self.sub_cpu.insert((r, off));
        }
    }

    /// The CPU is about to write page `p`: the model's answer, done; then
    /// invariant 2 (the CPU sees every GPU byte of finished submissions).
    fn touch(&mut self, p: Page, kind: CpuKind) {
        let t = self.model.cpu_touch(p, kind);
        self.check_demoted(p, self.model.class(p) == Class::GpuOnly, "cpu touch");
        let r = p.region;
        let pool_page = self.regions[r].p + (p.at - self.regions[r].class.at) as usize;
        match t {
            Touch::Nothing => {}
            Touch::Sync | Touch::Demote(_) => {
                self.regions[r]
                    .copy
                    .as_ref()
                    .expect("replayed")
                    .copy_back_page(pool_page);
                self.regions[r].counts.synced_pages += 1;
            }
        }
        if let Touch::Demote(why) = t {
            *self.regions[r].counts.demotions.entry(why).or_default() += 1;
            self.demoted.insert(p);
            self.demotions.push((p, why, self.seq, self.frame + 1));
        }
        let base = self.regions[r].class.at;
        let (from, to) = (p.at - base, p.at - base + KIB64);
        let written: Vec<(u64, u8)> = self.regions[r]
            .truth
            .range(from..to)
            .filter(|&(&o, _)| !self.sub_gpu.contains_key(&(r, o)))
            .map(|(&o, &v)| (o, v))
            .collect();
        for (off, v) in written {
            self.regions[r].counts.checks[1] += 1;
            let t = self.title(r, off);
            if t != v {
                self.violate(2, r, off, format!("the CPU would read {t:#04x} where the GPU wrote {v:#04x} (no sync before the touch)"));
            }
        }
    }

    fn gpu_read(&mut self, r: usize, from: u64, to: u64) {
        let reg = &mut self.regions[r];
        reg.counts.gpu_read += 1;
        if reg.class.class == Class::CpuOnly {
            reg.counts.contradictions += 1;
        }
        // Runs of pages the model refreshes, cut to the read range.
        let pages = self.pages(r, from, to);
        let base = self.regions[r].class.at;
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for p in pages {
            let refresh = self.model.refresh(p);
            self.check_demoted(p, !refresh, "refresh");
            let c = &mut self.regions[r].counts;
            if refresh {
                c.refreshed_pages += 1;
                let (a, b) = ((p.at - base).max(from), (p.at - base + KIB64).min(to));
                match runs.last_mut() {
                    Some((_, e)) if *e == a => *e = b,
                    _ => runs.push((a, b)),
                }
            } else {
                c.unrefreshed_pages += 1;
            }
        }
        let p = self.regions[r].p;
        for (a, b) in runs {
            self.regions[r]
                .copy
                .as_ref()
                .expect("replayed")
                .update(p + a as usize, (b - a) as usize);
        }
        // Invariant 1: every byte written so far is in the copy as one
        // memory holds it (this submission's GPU bytes as written).
        let written: BTreeMap<u64, u8> = self.regions[r]
            .truth
            .range(from..to)
            .map(|(&o, &v)| (o, v))
            .collect();
        let mut sub: Vec<(u64, u8)> = self
            .sub_gpu
            .range((r, from)..(r, to))
            .map(|(&(_, o), &v)| (o, v))
            .collect();
        let mut expect = written;
        expect.extend(sub.drain(..));
        for (off, v) in expect {
            if self.sub_cpu.contains(&(r, off)) && self.sub_gpu.contains_key(&(r, off)) {
                continue; // a race of both writers
            }
            self.regions[r].counts.checks[0] += 1;
            let c = self.copy_byte(r, off);
            if c != v {
                let who = if self.sub_cpu.contains(&(r, off)) {
                    "the CPU's"
                } else {
                    "a"
                };
                self.violate(
                    1,
                    r,
                    off,
                    format!("the GPU would read {c:#04x} where {who} byte is {v:#04x}"),
                );
            }
        }
    }

    fn gpu_decl(&mut self, r: usize, from: u64, to: u64) {
        let reg = &mut self.regions[r];
        reg.counts.gpu_decl += 1;
        if reg.class.class == Class::CpuOnly {
            reg.counts.contradictions += 1;
        }
        if self.decls.insert((r, from, to)) {
            let p = self.regions[r].p;
            self.regions[r].copy.as_ref().expect("replayed").writing(
                p + from as usize,
                (to - from) as usize,
                false,
            );
        }
    }

    fn gpu_write(&mut self, r: usize, from: u64, to: u64) {
        let reg = &mut self.regions[r];
        reg.counts.gpu_write += 1;
        if reg.class.class == Class::CpuOnly {
            reg.counts.contradictions += 1;
        }
        for off in self.samples(r, from, to, true) {
            let v = self.next_value(self.copy_byte(r, off));
            let at = self.regions[r].p as u64 + off;
            self.regions[r]
                .copy
                .as_ref()
                .expect("replayed")
                .write_host(at, &[v]);
            self.sub_gpu.insert((r, off), v);
        }
    }

    /// `POINT end`: copy-backs as the model says, then invariant 2 where
    /// they went back.
    fn sync_point(&mut self) {
        // Declared ranges: back if any page of them goes back.
        for (r, from, to) in std::mem::take(&mut self.decls) {
            let pages = self.pages(r, from, to);
            let mut back = false;
            for p in pages {
                let b = self.model.copy_back(p);
                self.check_demoted(p, !b, "copy-back");
                back |= b;
            }
            let p = self.regions[r].p + from as usize;
            let copy = self.regions[r].copy.as_ref().expect("replayed");
            if back {
                copy.copy_back(p, (to - from) as usize);
            } else {
                copy.keep_writes(p, (to - from) as usize);
            }
            self.decls_done.insert((r, from, to));
        }
        // Written pages no declared range covers: as at computed addresses.
        let mut loose: BTreeSet<Page> = BTreeSet::new();
        for &(r, off) in self.sub_gpu.keys() {
            if !self
                .decls_done
                .iter()
                .any(|&(dr, a, b)| dr == r && a <= off && off < b)
            {
                loose.insert(self.page(r, off));
            }
        }
        for p in loose {
            let b = self.model.copy_back(p);
            self.check_demoted(p, !b, "copy-back");
            if b {
                let r = p.region;
                let pool_page = self.regions[r].p + (p.at - self.regions[r].class.at) as usize;
                self.regions[r]
                    .copy
                    .as_ref()
                    .expect("replayed")
                    .copy_back_page(pool_page);
            }
        }
        self.decls_done.clear();
        // The GPU's bytes are now what one memory holds; where they went
        // back, title memory must have them.
        for ((r, off), v) in std::mem::take(&mut self.sub_gpu) {
            if self.sub_cpu.contains(&(r, off)) {
                // A race of both writers: whatever title memory has now.
                let t = self.title(r, off);
                self.regions[r].truth.insert(off, t);
                continue;
            }
            self.regions[r].truth.insert(off, v);
            let p = self.page(r, off);
            if self.model.copy_back(p) {
                self.regions[r].counts.checks[1] += 1;
                let t = self.title(r, off);
                if t != v {
                    self.violate(
                        2,
                        r,
                        off,
                        format!(
                            "title memory has {t:#04x} at the sync point, the GPU wrote {v:#04x}"
                        ),
                    );
                }
            }
        }
        self.sub_cpu.clear();
    }

    /// Violations of any invariant.
    pub(crate) fn violations(&self) -> [u64; 3] {
        let mut v = [0; 3];
        for r in &self.regions {
            for (sum, n) in v.iter_mut().zip(r.counts.violations) {
                *sum += n;
            }
        }
        v
    }

    /// Checks made per invariant.
    pub(crate) fn checks(&self) -> [u64; 3] {
        let mut v = [0; 3];
        for r in &self.regions {
            for (sum, n) in v.iter_mut().zip(r.counts.checks) {
                *sum += n;
            }
        }
        v
    }

    /// Demotions per region index.
    pub(crate) fn demotions_of(&self, region: usize) -> u64 {
        self.regions[region].counts.demotions.values().sum()
    }

    /// Regions whose class the stream contradicts or whose pages were demoted.
    pub(crate) fn refuted(&self) -> Vec<usize> {
        (0..self.regions.len())
            .filter(|&i| self.regions[i].counts.contradictions > 0 || self.demotions_of(i) > 0)
            .collect()
    }

    /// Green: no violation and no region refuted.
    pub(crate) fn green(&self) -> bool {
        self.violations() == [0; 3] && self.refuted().is_empty()
    }

    /// The report: stream addresses only, the same text for the same stream,
    /// classification and model.
    pub(crate) fn report(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "replay model={} unit={} records={} submissions={} frames={} regions={}",
            self.model.name(),
            1u64 << self.shift,
            self.records,
            self.submissions,
            self.frame,
            self.regions.len()
        );
        for (i, r) in self.regions.iter().enumerate() {
            let c = &r.counts;
            let dem: Vec<String> = c
                .demotions
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect();
            let _ = writeln!(
                s,
                "region {i} at={:#x} len={:#x} class={}{}: cpu_w_title={} cpu_w_runtime={} gpu_r={} gpu_w_decl={} gpu_w={} other={} \
                 contradictions={} demotions={} [{}] synced_pages={} refreshed_pages={} unrefreshed_pages={} \
                 sync_up_bytes={} sync_back_bytes={} checks inv1={} inv2={} inv3={} violations inv1={} inv2={} inv3={}",
                r.class.at,
                r.class.len,
                class_name(r.class.class),
                if r.copy.is_none() {
                    " NOT REPLAYED (pool budget)"
                } else {
                    ""
                },
                c.cpu_title,
                c.cpu_runtime,
                c.gpu_read,
                c.gpu_decl,
                c.gpu_write,
                c.other,
                c.contradictions,
                c.demotions.values().sum::<u64>(),
                dem.join(" "),
                c.synced_pages,
                c.refreshed_pages,
                c.unrefreshed_pages,
                c.sync_up,
                c.sync_back,
                c.checks[0],
                c.checks[1],
                c.checks[2],
                c.violations[0],
                c.violations[1],
                c.violations[2],
            );
        }
        let out: Vec<String> = self
            .outside
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        let _ = writeln!(
            s,
            "outside every region: [{}]; parts in regions not replayed: {}",
            out.join(" "),
            self.unreplayed
        );
        for (p, why, seq, frame) in self.demotions.iter().take(SHOWN) {
            let _ = writeln!(
                s,
                "demotion region={} page={:#x} reason={why} seq={seq} frame={frame}",
                p.region, p.at
            );
        }
        if self.demotions.len() > SHOWN {
            let _ = writeln!(s, "... {} demotions in all", self.demotions.len());
        }
        for v in self.violations.iter().take(SHOWN) {
            let _ = writeln!(
                s,
                "violation inv{} region={} at={:#x} seq={} frame={}: {}",
                v.inv, v.region, v.at, v.seq, v.frame, v.what
            );
        }
        if self.violations.len() > SHOWN {
            let _ = writeln!(s, "... {} violations in all", self.violations.len());
        }
        let v = self.violations();
        let _ = writeln!(
            s,
            "invariants: {}",
            if v == [0; 3] {
                "ok".to_string()
            } else {
                format!("VIOLATED inv1={} inv2={} inv3={}", v[0], v[1], v[2])
            }
        );
        let refuted = self.refuted();
        let _ = writeln!(
            s,
            "classification: {}",
            if refuted.is_empty() {
                "consistent".to_string()
            } else {
                format!("REFUTED in regions {refuted:?}")
            }
        );
        let _ = writeln!(s, "verdict: {}", if self.green() { "GREEN" } else { "RED" });
        s
    }
}

#[path = "shadow_replay_tests.rs"]
mod cases;
