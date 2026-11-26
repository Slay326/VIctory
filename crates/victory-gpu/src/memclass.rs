//! Memory model, stage 1: which title memory the CPU and the GPU touch, so
//! that it can be classified (GPU-only, CPU-only, shared, unknown) before
//! anything about the copies changes. Measuring only: no hook here changes
//! what is copied, guarded or written.
//!
//! `VICTORY_MEMCLASS=1` turns it on, read once by [`init`] when the host
//! starts (before the title runs). Hooks never initialize: each is one load
//! of a flag and nothing else while off or before `init` (no array, no
//! lock, no atomics written, nothing allocated; the guard handler is one of
//! them).
//!
//! Runs with `VICTORY_MEMCLASS` or `VICTORY_MEMCLASS_STREAM` are for
//! classifying and capturing only: their frame times, fps and memory are no
//! valid performance numbers.
//!
//! Each unit of title memory (`VICTORY_MEMCLASS_GRAN`: `64k`, the guards'
//! page and the default, or `4k`) collects per frame which kinds of access
//! reached it ([`f`]), set lock-free (`fetch_or`) where the access is seen:
//! the guard handler (title CPU writes), the runtime's own writes, the
//! steps of a submission (GPU-read proxies: refreshes asked, heap views,
//! the memory page table), what shaders declared or marked as written and
//! what the copy-backs put into title memory. At each presented frame the
//! units touched are folded into their regions (a `XMemVirtualAlloc`
//! allocation, a physical-page section: every view of a section counts
//! once, at the section's offset; memory no region names goes by 4 GiB
//! window). A unit's flags over the run (`ever`) decide its class; the
//! rule is not fixed here: the table logs units per region and flag
//! combination, and a reader (docs/memclass-format.md) applies a rule.
//!
//! Variants left open on purpose (Fable's question F1): granularity, which
//! GPU-read proxy counts, whether the runtime's writes are CPU writes.
//! Each proxy has its own flag; [`f::CPU_R_AFTER_GPU`] has a hook
//! ([`cpu_read`]) but nothing calls it yet.
//!
//! `VICTORY_MEMCLASS_STREAM=<file>` also writes every hook's call, in
//! order, with submissions, sync points and frames, to a compact event
//! stream (format: `docs/memclass-format.md`), for a later replay;
//! `VICTORY_MEMCLASS_STREAM_FRAMES` (default 0: the whole run) ends it
//! early. Each frame record carries counters the stream is checked against
//! (docs/memclass-format.md, "Check against counters").
//!
//! shortcut: the guards see a title write only the first time after a
//! copy took the page, so per-frame CPU-write counts are low bounds; the
//! class uses the whole run. Memory freed and allocated again keeps its
//! old region until a new allocation or view names it.

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU16, AtomicU64, Ordering::*};
use std::sync::{Mutex, OnceLock, RwLock};

/// Access kinds, a bit each.
pub(crate) mod f {
    /// The title's CPU wrote (a guard fault).
    pub(crate) const CPU_W_TITLE: u16 = 1 << 0;
    /// The runtime wrote for the title (kernel file reads, DirectStorage,
    /// page clearing: `kernel_will_write`).
    pub(crate) const CPU_W_RUNTIME: u16 = 1 << 1;
    /// The title's CPU read what the GPU wrote (not measured yet).
    pub(crate) const CPU_R_AFTER_GPU: u16 = 1 << 2;
    /// Shaders marked the page written (64 KiB marks).
    pub(crate) const GPU_W_MARK: u16 = 1 << 3;
    /// A submission declared shaders write the range.
    pub(crate) const GPU_W_DECL: u16 = 1 << 4;
    /// A copy-back put GPU-written bytes into title memory.
    pub(crate) const GPU_W_BACK: u16 = 1 << 5;
    /// A GPU copy wrote title memory (copy steps, texture read-backs).
    pub(crate) const GPU_W_COPY: u16 = 1 << 6;
    /// GPU-read proxy: a list asked for the range to be brought up to date.
    pub(crate) const GPU_R_REFRESH: u16 = 1 << 7;
    /// GPU-read proxy: a heap's views reach the range.
    pub(crate) const GPU_R_HEAP: u16 = 1 << 8;
    /// GPU-read proxy: the memory page table names the copy.
    pub(crate) const GPU_R_TABLE: u16 = 1 << 9;
    /// Shaders wrote it and the copies dropped it (withheld, never back).
    pub(crate) const GPU_W_KEPT: u16 = 1 << 10;
    /// The runtime mapped or unmapped the memory.
    pub(crate) const REMAP: u16 = 1 << 11;
    /// A refresh copied the unit into a host copy (sync up).
    pub(crate) const COPIED_UP: u16 = 1 << 12;
    /// Derived at the frame's end: a refresh copied the unit again although
    /// it had been copied before and nothing of ours wrote it in this frame
    /// or the one before: the title's CPU wrote it, seen or not by a fault
    /// (a page left open after its first write faults no more).
    pub(crate) const CPU_W_DIRTY: u16 = 1 << 13;
    /// Shaders wrote a page at addresses they computed and the copy-back
    /// dropped it: the title wrote the page since the copy took it, its
    /// bytes kept (`Shadow::copy_back_page`, "both wrote"). The whole page
    /// (cut to the copy): which of its bytes the GPU wrote is not known
    /// there. In no class rule (a diagnosis of H-02, local test branch).
    pub(crate) const GPU_W_LOST: u16 = 1 << 14;
    /// Our own writes that make a refresh copy a unit again (not the CPU's).
    pub(crate) const OURS: u16 = GPU_W_MARK | GPU_W_BACK | GPU_W_COPY | GPU_W_KEPT | CPU_W_RUNTIME | REMAP;
}

/// Names of the flags, by bit (logs, the stream's documentation).
pub(crate) const FLAG_NAMES: [&str; 15] = [
    "cpu_w_title",
    "cpu_w_runtime",
    "cpu_r_after_gpu",
    "gpu_w_mark",
    "gpu_w_decl",
    "gpu_w_back",
    "gpu_w_copy",
    "gpu_r_refresh",
    "gpu_r_heap",
    "gpu_r_table",
    "gpu_w_kept",
    "remap",
    "copied_up",
    "cpu_w_dirty",
    "gpu_w_lost",
];

/// Traffic that keeps the copies and title memory in step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sync {
    /// Title memory copied into a host copy.
    Up = 0,
    /// GPU-written bytes put back into title memory.
    Back = 1,
}
const SYNC_KINDS: usize = 2;
const SYNC_NAMES: [&str; SYNC_KINDS] = ["up", "back"];

/// Points of a submission, in the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Point {
    /// Prepared: the lists go to the GPU.
    Executed = 0,
    /// The GPU is done; copy-backs follow.
    GpuDone = 1,
    /// Copy-backs done, the submission ends.
    End = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    /// log2 of the unit: 12 or 16.
    pub(crate) shift: u32,
    pub(crate) stream: Option<std::path::PathBuf>,
    pub(crate) stream_frames: u32,
    /// Seconds between interval lines and between tables.
    pub(crate) interval_s: u64,
    pub(crate) table_s: u64,
}

impl Config {
    /// From the environment (`get` reads a variable); `None` when off.
    pub(crate) fn from(get: impl Fn(&str) -> Option<String>) -> Option<Config> {
        if get("VICTORY_MEMCLASS").as_deref().map(str::trim) != Some("1") {
            return None;
        }
        let shift = match get("VICTORY_MEMCLASS_GRAN").as_deref().map(str::trim) {
            Some("4k" | "4K" | "4096") => 12,
            Some("64k" | "64K" | "65536") | None => 16,
            Some(other) => {
                tracing::warn!(other, "VICTORY_MEMCLASS_GRAN is 4k or 64k: 64k taken");
                16
            }
        };
        let number = |name: &str, default: u64| {
            get(name).and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        Some(Config {
            shift,
            stream: get("VICTORY_MEMCLASS_STREAM").filter(|s| !s.trim().is_empty()).map(Into::into),
            stream_frames: number("VICTORY_MEMCLASS_STREAM_FRAMES", 0) as u32,
            interval_s: number("VICTORY_MEMCLASS_INTERVAL_S", 5).max(1),
            table_s: number("VICTORY_MEMCLASS_TABLE_S", 60).max(1),
        })
    }
}

static ON: AtomicBool = AtomicBool::new(false);
static GLOBAL: OnceLock<Option<Classifier>> = OnceLock::new();

/// Reads the environment and makes the process's classifier if asked for:
/// once, when the host starts, before any hook can run (a hook in the guard
/// handler must not read the environment, allocate or log).
pub fn init() {
    let c = GLOBAL.get_or_init(|| {
        let c = Config::from(|k| std::env::var(k).ok()).map(Classifier::new);
        if let Some(c) = &c {
            tracing::info!(
                target: "victory_gpu::memclass",
                "memclass on: unit {} KiB, stream {:?}; this run's frame times, fps and memory \
                 are no valid performance numbers",
                1usize << c.cfg.shift >> 10,
                c.cfg.stream
            );
        }
        c
    });
    ON.store(c.is_some(), Release);
}

#[cfg(test)]
thread_local! {
    /// Tests: the classifier this thread's hooks use instead of the
    /// process's ([`on_this_thread`]).
    static ON_THREAD: std::cell::Cell<Option<&'static Classifier>> = const { std::cell::Cell::new(None) };
}

/// Tests: hooks called on this thread go to `c` (None: as before).
#[cfg(test)]
pub(crate) fn on_this_thread(c: Option<&'static Classifier>) {
    ON_THREAD.with(|t| t.set(c));
}

/// The process's classifier, if [`init`] made one; never makes it.
#[inline]
fn global() -> Option<&'static Classifier> {
    #[cfg(test)]
    if let Some(c) = ON_THREAD.with(std::cell::Cell::get) {
        return Some(c);
    }
    if !ON.load(Acquire) {
        return None;
    }
    GLOBAL.get().and_then(Option::as_ref)
}

/// Hook: `flag` reached title memory `[at, at + len)`.
#[inline]
pub(crate) fn tag(flag: u16, at: usize, len: usize) {
    if let Some(c) = global() {
        c.tag(flag, at, len);
    }
}

/// Hook: [`tag`] for the units only, not the stream (the page table's
/// every copy per list: too many records; the stream gets the copies the
/// table refreshes, [`record`]).
#[inline]
pub(crate) fn tag_units(flag: u16, at: usize, len: usize) {
    if let Some(c) = global() {
        c.units(flag, at, len);
    }
}

/// Hook: [`tag`] for the stream only.
#[inline]
pub(crate) fn record(flag: u16, at: usize, len: usize) {
    if let Some(c) = global()
        && let Some(s) = &c.stream
    {
        s.access(flag, c.submissions.load(Relaxed), at, len);
    }
}

/// Whether `key` is asked about for the first time in this frame (work a
/// hook does once per frame).
pub(crate) fn first_in_frame(key: usize) -> bool {
    global().is_some_and(|c| c.first_in_frame(key))
}

/// Whether classifying is on (callers skip work done only for it).
#[inline]
pub(crate) fn on() -> bool {
    global().is_some()
}

/// Hook (not called yet, F1): the title's CPU read `[at, at + len)` after
/// the GPU wrote it.
#[allow(dead_code)]
pub(crate) fn cpu_read(at: usize, len: usize) {
    tag(f::CPU_R_AFTER_GPU, at, len);
}

/// Hook: sync traffic of `len` bytes at `at`, taking `us` microseconds.
#[inline]
pub(crate) fn sync(kind: Sync, at: usize, len: usize, us: u64) {
    if let Some(c) = global() {
        c.sync(kind, at, len, us);
    }
}

/// Hook: a copy-back's spans (see [`Classifier::copied_back`]).
pub(crate) fn copied_back(at: usize, spans: &[(usize, usize)], landed: bool, us: u64) {
    if let Some(c) = global() {
        c.copied_back(at, spans, landed, us);
    }
}

/// Hook: a submission of `lists` lists begins; the returned value marks
/// its end when dropped.
pub(crate) fn submission(lists: usize) -> Option<Submission<'static>> {
    global().map(|c| c.submit(lists))
}

/// The number of the latest submission begun (the stream's `seq`); 0 with
/// classifying off or before the first.
pub(crate) fn seq() -> u64 {
    global().map_or(0, |c| c.submissions.load(Relaxed))
}

/// Hook: a point of the submission under way.
pub(crate) fn point(p: Point) {
    if let Some(c) = global() {
        c.point(p);
    }
}

/// Hook: a submission's refreshes copied `bytes` into host copies (the
/// counter the stream's sync records are checked against).
#[inline]
pub(crate) fn refresh_copied(bytes: usize) {
    if let Some(c) = global() {
        c.copied.fetch_add(bytes as u64, Relaxed);
    }
}

/// Hook: a presented frame.
pub(crate) fn frame_done() {
    if let Some(c) = global() {
        c.frame_done(std::time::Instant::now());
    }
}

/// The process ends (the title ended itself): the stream gets its END
/// record (the last frame), what it holds goes to the file and the file
/// closes. Again, or with classifying off: nothing.
pub fn finish() {
    if let Some(c) = global() {
        c.finish();
    }
}

/// Hook: the runtime allocated `[at, at + len)` (`watched`: write-watched,
/// so the guards see the title's writes there).
pub fn allocated(at: usize, len: usize, watched: bool) {
    if let Some(c) = global() {
        c.allocated(at, len, watched);
    }
}

/// Hook: the runtime made section `section` of `bytes`.
pub(crate) fn section(section: usize, bytes: usize) {
    if let Some(c) = global() {
        c.section(section, bytes);
    }
}

/// Hook: the runtime mapped `[at, at + len)` of `section` from `offset`.
pub(crate) fn view(at: usize, len: usize, section: usize, offset: usize) {
    if let Some(c) = global() {
        c.view(at, len, section, offset);
    }
}

/// Hook: the runtime mapped or unmapped `[at, at + len)`.
pub(crate) fn remapped(at: usize, len: usize) {
    if let Some(c) = global() {
        c.remapped(at, len);
    }
}

/// A submission under way (stream), ended when dropped.
pub(crate) struct Submission<'a>(&'a Classifier);

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        self.0.point(Point::End);
    }
}

/// Bits of the address a chunk covers.
const CHUNK_SHIFT: u32 = 32;
/// Chunk slots: 4 TiB of 4 GiB windows touched.
const SLOTS: usize = 1024;

/// Flags of the units of one 4 GiB window, and a bit per 64 units set
/// while any of them holds flags.
struct Chunk {
    flags: Box<[AtomicU16]>,
    summary: Box<[AtomicU64]>,
}

impl Chunk {
    fn new(units: usize) -> Chunk {
        Chunk {
            flags: (0..units).map(|_| AtomicU16::new(0)).collect(),
            summary: (0..units.div_ceil(64 * 64)).map(|_| AtomicU64::new(0)).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `XMemVirtualAlloc`'s; `watched`: the guards see the title's writes.
    Virtual { watched: bool },
    /// Physical pages (by section; views count here).
    Section,
    /// Memory no region names, by 4 GiB window: the guards do not see it.
    Unnamed,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Virtual { watched: true } => "virtual",
            Kind::Virtual { watched: false } => "virtual_unwatched",
            Kind::Section => "section",
            Kind::Unnamed => "unnamed",
        }
    }

    /// Whether the title's CPU writes there can be seen at all.
    pub(crate) fn observed(self) -> bool {
        matches!(self, Kind::Virtual { watched: true } | Kind::Section)
    }
}

#[derive(Clone, Debug)]
struct Meta {
    kind: Kind,
    /// Address (virtual), handle (section), window base (unnamed).
    at: usize,
    len: usize,
}

/// Where each address belongs.
#[derive(Default)]
struct Regions {
    list: Vec<Meta>,
    /// Address runs: start -> (end, region, offset of start in it).
    spans: BTreeMap<usize, (usize, u32, usize)>,
    sections: HashMap<usize, u32>,
    windows: HashMap<usize, u32>,
}

impl Regions {
    fn add(&mut self, m: Meta) -> u32 {
        self.list.push(m);
        (self.list.len() - 1) as u32
    }

    /// Removes the runs over `[at, end)` (`only`: those of regions it
    /// accepts), cutting runs that reach past it.
    fn cut(&mut self, at: usize, end: usize, only: impl Fn(&Meta) -> bool) {
        if at >= end {
            return;
        }
        let mut hit: Vec<(usize, (usize, u32, usize))> = Vec::new();
        if let Some((&s, &v)) = self.spans.range(..at).next_back()
            && v.0 > at
        {
            hit.push((s, v));
        }
        hit.extend(self.spans.range(at..end).map(|(&s, &v)| (s, v)));
        for (s, (e, r, o)) in hit {
            if !only(&self.list[r as usize]) {
                continue;
            }
            self.spans.remove(&s);
            if s < at {
                self.spans.insert(s, (at, r, o));
            }
            if e > end {
                self.spans.insert(end, (e, r, o + (end - s)));
            }
        }
    }

    fn insert(&mut self, at: usize, end: usize, region: u32, offset: usize) {
        self.cut(at, end, |_| true);
        if at < end {
            self.spans.insert(at, (end, region, offset));
        }
    }

    fn section_id(&mut self, section: usize) -> u32 {
        if let Some(&r) = self.sections.get(&section) {
            return r;
        }
        let r = self.add(Meta { kind: Kind::Section, at: section, len: 0 });
        self.sections.insert(section, r);
        r
    }

    /// `[at, at + len)` cut where regions change: region and bytes.
    fn split(&mut self, at: usize, len: usize) -> Vec<(u32, usize)> {
        let end = at.saturating_add(len);
        let mut out: Vec<(u32, usize)> = Vec::new();
        let mut pos = at;
        while pos < end {
            let (r, _) = self.place(pos);
            // Where the run holding `pos` ends: its span's end, else the
            // next span's start or the window's end.
            let run_end = match self.spans.range(..=pos).next_back() {
                Some((_, &(e, _, _))) if pos < e => e,
                _ => {
                    let window_end = ((pos >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
                    self.spans.range(pos + 1..).next().map_or(window_end, |(&s, _)| s.min(window_end))
                }
            };
            let next = run_end.min(end).max(pos + 1);
            match out.last_mut() {
                Some((last, bytes)) if *last == r => *bytes += next - pos,
                _ => out.push((r, next - pos)),
            }
            pos = next;
        }
        out
    }

    /// The region and offset in it of address `at`.
    fn place(&mut self, at: usize) -> (u32, usize) {
        if let Some((&s, &(e, r, o))) = self.spans.range(..=at).next_back()
            && at < e
        {
            return (r, o + (at - s));
        }
        let window = at >> CHUNK_SHIFT;
        let r = match self.windows.get(&window) {
            Some(&r) => r,
            None => {
                let r = self.add(Meta { kind: Kind::Unnamed, at: window << CHUNK_SHIFT, len: 1 << CHUNK_SHIFT });
                self.windows.insert(window, r);
                r
            }
        };
        (r, at - (window << CHUNK_SHIFT))
    }
}

/// A unit over the run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Unit {
    pub(crate) ever: u16,
    pub(crate) frames: u32,
    pub(crate) first: u32,
    pub(crate) last: u32,
    /// Flags of the unit's last frame.
    pub(crate) prev: u16,
}

/// A region over the run.
#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    /// Frames in which any unit had each flag.
    pub(crate) frames_with: [u32; 16],
    /// Units with each flag, summed over frames.
    pub(crate) hits: [u64; 16],
    /// Sync bytes and microseconds, by kind.
    pub(crate) sync: [(u64, u64); SYNC_KINDS],
}

#[derive(Default)]
struct State {
    frame: u32,
    units: HashMap<(u32, u64), Unit>,
    stats: HashMap<u32, Stats>,
    /// Since the last interval line: frames, units per flag, sync.
    interval: (u32, [u64; 16], [(u64, u64); SYNC_KINDS]),
    logged: Option<std::time::Instant>,
    tabled: Option<std::time::Instant>,
}

/// Units of the class each rule gives, in a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Class {
    GpuOnly,
    CpuOnly,
    Shared,
    Unknown,
}

/// Which flags count as CPU and as GPU access.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rule {
    pub(crate) cpu: u16,
    pub(crate) gpu: u16,
}

impl Rule {
    /// The proposal in the night queue (Q2): runtime writes apart, every
    /// GPU-read proxy counts.
    pub(crate) const PROPOSED: Rule = Rule {
        cpu: f::CPU_W_TITLE | f::CPU_W_DIRTY | f::CPU_R_AFTER_GPU,
        gpu: f::GPU_W_MARK
            | f::GPU_W_DECL
            | f::GPU_W_BACK
            | f::GPU_W_COPY
            | f::GPU_W_KEPT
            | f::GPU_R_REFRESH
            | f::GPU_R_HEAP
            | f::GPU_R_TABLE,
    };

    /// The class of a unit with flags `ever` in a region of `kind`: memory
    /// whose CPU writes cannot be seen is never GPU-only (unknown).
    pub(crate) fn class(self, ever: u16, kind: Kind) -> Class {
        let cpu = ever & self.cpu != 0;
        let gpu = ever & self.gpu != 0;
        match (cpu, gpu) {
            (true, true) => Class::Shared,
            (true, false) => Class::CpuOnly,
            (false, true) if kind.observed() => Class::GpuOnly,
            _ => Class::Unknown,
        }
    }
}

pub(crate) struct Classifier {
    cfg: Config,
    keys: Box<[AtomicU64]>,
    chunks: Box<[AtomicPtr<Chunk>]>,
    grow: Mutex<()>,
    /// Tags lost because every chunk slot was taken.
    lost: AtomicU64,
    regions: RwLock<Regions>,
    syncs: Mutex<Vec<(Sync, usize, usize, u64)>>,
    state: Mutex<State>,
    stream: Option<Stream>,
    submissions: AtomicU64,
    /// Bytes refreshes copied ([`refresh_copied`]), over the run.
    copied: AtomicU64,
    /// Keys asked about in this frame ([`first_in_frame`]).
    seen: Mutex<std::collections::HashSet<usize>>,
}

impl Drop for Classifier {
    fn drop(&mut self) {
        for c in self.chunks.iter() {
            let p = c.load(Acquire);
            if !p.is_null() {
                // SAFETY: made by `Box::into_raw` in `chunk`, freed once.
                drop(unsafe { Box::from_raw(p) });
            }
        }
    }
}

impl Classifier {
    pub(crate) fn new(cfg: Config) -> Classifier {
        let stream = cfg.stream.as_ref().and_then(|path| Stream::open(path, &cfg));
        Classifier {
            keys: (0..SLOTS).map(|_| AtomicU64::new(0)).collect(),
            chunks: (0..SLOTS).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect(),
            grow: Mutex::new(()),
            lost: AtomicU64::new(0),
            regions: RwLock::new(Regions::default()),
            syncs: Mutex::new(Vec::new()),
            state: Mutex::new(State::default()),
            stream,
            submissions: AtomicU64::new(0),
            copied: AtomicU64::new(0),
            seen: Mutex::new(std::collections::HashSet::new()),
            cfg,
        }
    }

    fn units_per_chunk(&self) -> usize {
        1 << (CHUNK_SHIFT - self.cfg.shift)
    }

    /// The chunk of 4 GiB window `hi`, made if `make`.
    fn chunk(&self, hi: u64, make: bool) -> Option<&Chunk> {
        let key = hi + 1;
        let start = (hi as usize).wrapping_mul(0x9E37_79B9) & (SLOTS - 1);
        let find = || {
            let mut s = start;
            for _ in 0..SLOTS {
                match self.keys[s].load(Acquire) {
                    k if k == key => return Ok(s),
                    0 => return Err(Some(s)),
                    _ => s = (s + 1) & (SLOTS - 1),
                }
            }
            Err(None)
        };
        let s = match find() {
            Ok(s) => s,
            Err(_) if !make => return None,
            Err(_) => {
                let _g = self.grow.lock().unwrap_or_else(|e| e.into_inner());
                match find() {
                    Ok(s) => s,
                    Err(None) => {
                        self.lost.fetch_add(1, Relaxed);
                        return None;
                    }
                    Err(Some(s)) => {
                        let c = Box::into_raw(Box::new(Chunk::new(self.units_per_chunk())));
                        self.chunks[s].store(c, Release);
                        self.keys[s].store(key, Release);
                        s
                    }
                }
            }
        };
        // SAFETY: set before its key, freed only when the classifier goes.
        Some(unsafe { &*self.chunks[s].load(Acquire) })
    }

    pub(crate) fn tag(&self, flag: u16, at: usize, len: usize) {
        if len == 0 || flag == 0 {
            return;
        }
        if let Some(s) = &self.stream {
            s.access(flag, self.submissions.load(Relaxed), at, len);
        }
        self.units(flag, at, len);
    }

    /// [`Classifier::tag`] without the stream.
    pub(crate) fn units(&self, flag: u16, at: usize, len: usize) {
        if len == 0 || flag == 0 {
            return;
        }
        let shift = self.cfg.shift;
        let per = self.units_per_chunk();
        let mut u = at >> shift;
        let last = at.saturating_add(len - 1) >> shift;
        while u <= last {
            let hi = u >> (CHUNK_SHIFT - shift);
            let base = hi << (CHUNK_SHIFT - shift);
            let end = (base + per - 1).min(last);
            if let Some(chunk) = self.chunk(hi as u64, true) {
                for i in u - base..=end - base {
                    let a = &chunk.flags[i];
                    if a.load(Relaxed) & flag == 0 && a.fetch_or(flag, Relaxed) == 0 {
                        chunk.summary[i >> 12].fetch_or(1 << ((i >> 6) & 63), Relaxed);
                    }
                }
            }
            u = end + 1;
        }
    }

    pub(crate) fn first_in_frame(&self, key: usize) -> bool {
        self.seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key)
    }

    pub(crate) fn sync(&self, kind: Sync, at: usize, len: usize, us: u64) {
        if let Some(s) = &self.stream {
            s.sync(kind, at, len, us);
        }
        if kind == Sync::Up {
            self.units(f::COPIED_UP, at, len);
        }
        self.syncs.lock().unwrap_or_else(|e| e.into_inner()).push((kind, at, len, us));
    }

    /// A copy-back's spans `[at + start, at + end)` took `us`; `landed`:
    /// they reached title memory. Only bytes that landed are GPU writes and
    /// traffic; the time goes to the spans by bytes.
    pub(crate) fn copied_back(&self, at: usize, spans: &[(usize, usize)], landed: bool, us: u64) {
        if !landed {
            return;
        }
        let bytes: usize = spans.iter().map(|&(s, e)| e - s).sum();
        let mut given = 0;
        for (k, &(start, end)) in spans.iter().enumerate() {
            self.tag(f::GPU_W_BACK, at + start, end - start);
            let part = if k + 1 == spans.len() { us - given } else { us * (end - start) as u64 / bytes.max(1) as u64 };
            given += part;
            self.sync(Sync::Back, at + start, end - start, part);
        }
    }

    pub(crate) fn submit(&self, lists: usize) -> Submission<'_> {
        let seq = self.submissions.fetch_add(1, Relaxed) + 1;
        if let Some(s) = &self.stream {
            s.submit(seq, lists);
        }
        Submission(self)
    }

    pub(crate) fn point(&self, p: Point) {
        if let Some(s) = &self.stream {
            s.point(p, self.submissions.load(Relaxed));
        }
    }

    pub(crate) fn allocated(&self, at: usize, len: usize, watched: bool) {
        if let Some(s) = &self.stream {
            s.region(if watched { 1 } else { 0 }, [at, len, 0, 0]);
        }
        let mut r = self.regions.write().unwrap_or_else(|e| e.into_inner());
        let id = r.add(Meta { kind: Kind::Virtual { watched }, at, len });
        r.insert(at, at.saturating_add(len), id, 0);
    }

    pub(crate) fn section(&self, section: usize, bytes: usize) {
        if let Some(s) = &self.stream {
            s.region(2, [section, bytes, 0, 0]);
        }
        let mut r = self.regions.write().unwrap_or_else(|e| e.into_inner());
        // A handle's number given out again is a new section.
        let id = r.add(Meta { kind: Kind::Section, at: section, len: bytes });
        r.sections.insert(section, id);
    }

    pub(crate) fn view(&self, at: usize, len: usize, section: usize, offset: usize) {
        if let Some(s) = &self.stream {
            s.region(3, [at, len, section, offset]);
        }
        let mut r = self.regions.write().unwrap_or_else(|e| e.into_inner());
        let id = r.section_id(section);
        r.insert(at, at.saturating_add(len), id, offset);
    }

    pub(crate) fn remapped(&self, at: usize, len: usize) {
        self.tag(f::REMAP, at, len);
        // Views there are gone (a new one says so again); allocations stay.
        let mut r = self.regions.write().unwrap_or_else(|e| e.into_inner());
        r.cut(at, at.saturating_add(len), |m| m.kind == Kind::Section);
    }

    /// See [`finish`].
    pub(crate) fn finish(&self) {
        if let Some(s) = &self.stream {
            let frame = self.state.lock().unwrap_or_else(|e| e.into_inner()).frame;
            s.finish(frame);
        }
    }

    /// Folds the frame's units into their regions (and logs, by time).
    pub(crate) fn frame_done(&self, now: std::time::Instant) {
        let shift = self.cfg.shift;
        let per = self.units_per_chunk();
        let mut taken: Vec<(usize, u16)> = Vec::new();
        for s in 0..SLOTS {
            let key = self.keys[s].load(Acquire);
            if key == 0 {
                continue;
            }
            let hi = (key - 1) as usize;
            let Some(chunk) = self.chunk(hi as u64, false) else {
                continue;
            };
            for (w, word) in chunk.summary.iter().enumerate() {
                let mut bits = word.swap(0, AcqRel);
                while bits != 0 {
                    let b = w * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    for i in b * 64..(b * 64 + 64).min(per) {
                        let flags = chunk.flags[i].swap(0, AcqRel);
                        if flags != 0 {
                            taken.push((((hi << (CHUNK_SHIFT - shift)) + i) << shift, flags));
                        }
                    }
                }
            }
        }
        let syncs = std::mem::take(&mut *self.syncs.lock().unwrap_or_else(|e| e.into_inner()));
        self.seen.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut regions = self.regions.write().unwrap_or_else(|e| e.into_inner());
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.frame += 1;
        let frame = st.frame;
        // Units of the frame by region and unit there: views of a section
        // meet at one unit and count once.
        let mut this: HashMap<(u32, u64), u16> = HashMap::new();
        for (at, flags) in taken {
            let (r, off) = regions.place(at);
            *this.entry((r, (off >> shift) as u64)).or_default() |= flags;
        }
        let mut or: HashMap<u32, u16> = HashMap::new();
        for (&(r, u), &flags) in &this {
            let unit = st.units.entry((r, u)).or_insert(Unit { first: frame, ..Unit::default() });
            let before = if unit.last + 1 == frame { unit.prev } else { 0 };
            let mut flags = flags;
            if flags & f::COPIED_UP != 0 && unit.ever & f::COPIED_UP != 0 && (flags | before) & f::OURS == 0 {
                flags |= f::CPU_W_DIRTY;
            }
            unit.ever |= flags;
            unit.frames += 1;
            unit.last = frame;
            unit.prev = flags;
            *or.entry(r).or_default() |= flags;
            let stats = st.stats.entry(r).or_default();
            for bit in 0..16 {
                if flags & 1 << bit != 0 {
                    stats.hits[bit] += 1;
                }
            }
            for bit in 0..16 {
                if flags & 1 << bit != 0 {
                    st.interval.1[bit] += 1;
                }
            }
        }
        for (r, flags) in or {
            let stats = st.stats.entry(r).or_default();
            for bit in 0..16 {
                if flags & 1 << bit != 0 {
                    stats.frames_with[bit] += 1;
                }
            }
        }
        for (kind, at, len, us) in syncs {
            // Each region its bytes, and the time by bytes.
            let pieces = regions.split(at, len);
            let mut given = 0;
            for (k, &(r, bytes)) in pieces.iter().enumerate() {
                let part = if k + 1 == pieces.len() { us - given } else { us * bytes as u64 / len.max(1) as u64 };
                given += part;
                let s = &mut st.stats.entry(r).or_default().sync[kind as usize];
                s.0 += bytes as u64;
                s.1 += part;
            }
            let i = &mut st.interval.2[kind as usize];
            i.0 += len as u64;
            i.1 += us;
        }
        st.interval.0 += 1;
        if let Some(s) = &self.stream {
            s.frame(frame, self.counters());
        }
        let logged = *st.logged.get_or_insert(now);
        if now.duration_since(logged).as_secs() >= self.cfg.interval_s {
            tracing::info!(target: "victory_gpu::memclass", "{}", self.interval_line(&regions, &mut st, now - logged));
            st.logged = Some(now);
        }
        let tabled = *st.tabled.get_or_insert(now);
        if now.duration_since(tabled).as_secs() >= self.cfg.table_s {
            for line in self.table(&regions, &st) {
                tracing::info!(target: "victory_gpu::memclass", "{line}");
            }
            st.tabled = Some(now);
        }
    }

    /// Counters over the run kept elsewhere, for the stream's check:
    /// refreshes asked and guard faults (`crate::copystat`, diagnostics
    /// builds; 0 otherwise), bytes refreshes copied, bytes of copy steps.
    fn counters(&self) -> [u64; 4] {
        #[cfg(feature = "diagnostics")]
        let (refreshes, faults, copy_steps) = {
            use crate::copystat::{C, total};
            (total(C::Refreshes), total(C::GuardFaults), total(C::CopyBytes))
        };
        #[cfg(not(feature = "diagnostics"))]
        let (refreshes, faults, copy_steps) = (0, 0, 0);
        [refreshes, faults, self.copied.load(Relaxed), copy_steps]
    }

    fn interval_line(&self, regions: &Regions, st: &mut State, took: std::time::Duration) -> String {
        let (frames, units, sync) = std::mem::take(&mut st.interval);
        let n = f64::from(frames.max(1));
        let unit = (1u64 << self.cfg.shift) as f64;
        let mut line = format!(
            "memclass interval frame={} frames={frames} secs={:.1} unit={}",
            st.frame,
            took.as_secs_f64(),
            1u64 << self.cfg.shift
        );
        for (k, name) in SYNC_NAMES.iter().enumerate() {
            line += &format!(
                " sync_{name}_mb={:.2} sync_{name}_ms={:.2}",
                sync[k].0 as f64 / 1e6 / n,
                sync[k].1 as f64 / 1e3 / n
            );
        }
        for (bit, name) in FLAG_NAMES.iter().enumerate() {
            line += &format!(" {name}_mb={:.1}", units[bit] as f64 * unit / 1e6 / n);
        }
        // Classes so far under the proposed rule (the table's reader applies
        // others).
        let classes = classes_of(regions, st, unit as u64, Rule::PROPOSED);
        for (class, name) in [
            (Class::GpuOnly, "gpu_only"),
            (Class::CpuOnly, "cpu_only"),
            (Class::Shared, "shared"),
            (Class::Unknown, "unknown"),
        ] {
            line += &format!(" {name}_mb={:.1}", classes.get(&class).copied().unwrap_or(0) as f64 / 1e6);
        }
        line += &format!(" lost={}", self.lost.load(Relaxed));
        if let Some(s) = &self.stream {
            // Size of the stream so far, and per minute since it began.
            let bytes = s.written.load(Relaxed);
            let minutes = s.began.elapsed().as_secs_f64() / 60.0;
            line += &format!(
                " stream_mb={:.1} stream_mb_per_min={:.1} stream_on={}",
                bytes as f64 / 1e6,
                bytes as f64 / 1e6 / minutes.max(1e-9),
                s.on.load(Relaxed)
            );
        }
        line
    }

    /// The table so far: a line per region, a line per region and flag
    /// combination (units with exactly those flags over the run).
    fn table(&self, regions: &Regions, st: &State) -> Vec<String> {
        let unit = 1u64 << self.cfg.shift;
        let mut combos: BTreeMap<(u32, u16), u64> = BTreeMap::new();
        for (&(r, _), u) in &st.units {
            *combos.entry((r, u.ever)).or_default() += 1;
        }
        let mut lines = vec![format!(
            "memclass table begin frame={} unit={unit} regions={}",
            st.frame,
            st.stats.len()
        )];
        let mut ids: Vec<u32> = st.stats.keys().copied().collect();
        ids.sort_unstable();
        for r in ids {
            let m = &regions.list[r as usize];
            let s = &st.stats[&r];
            let mut line = format!(
                "memclass region id={r} kind={} at={:#x} len={}",
                m.kind.name(),
                m.at,
                m.len
            );
            for (bit, name) in FLAG_NAMES.iter().enumerate() {
                line += &format!(" f_{name}={}", s.frames_with[bit]);
            }
            for (k, name) in SYNC_NAMES.iter().enumerate() {
                line += &format!(" sync_{name}_bytes={} sync_{name}_us={}", s.sync[k].0, s.sync[k].1);
            }
            lines.push(line);
        }
        for ((r, ever), n) in combos {
            lines.push(format!("memclass units region={r} ever={ever:#06x} units={n} bytes={}", n * unit));
        }
        lines.push(format!("memclass table end frame={}", st.frame));
        lines
    }

    /// Bytes per class under `rule` so far.
    #[cfg(test)]
    pub(crate) fn classes(&self, rule: Rule) -> HashMap<Class, u64> {
        let regions = self.regions.read().unwrap_or_else(|e| e.into_inner());
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        classes_of(&regions, &st, 1u64 << self.cfg.shift, rule)
    }
}

/// Bytes per class under `rule` of the units in `st`.
fn classes_of(regions: &Regions, st: &State, unit: u64, rule: Rule) -> HashMap<Class, u64> {
    let mut out = HashMap::new();
    for (&(r, _), u) in &st.units {
        *out.entry(rule.class(u.ever, regions.list[r as usize].kind)).or_insert(0) += unit;
    }
    out
}

/// The event stream (`docs/memclass-format.md`): records gathered under a
/// lock, written to the file at each frame.
struct Stream {
    buf: Mutex<(Vec<u8>, usize, u32)>,
    file: Mutex<Option<std::fs::File>>,
    on: AtomicBool,
    /// Bytes written to the file, and when it was made.
    written: AtomicU64,
    began: std::time::Instant,
}

pub(crate) const STREAM_MAGIC: &[u8; 4] = b"VMCS";
pub(crate) const STREAM_VERSION: u16 = 1;

pub(crate) mod rec {
    pub(crate) const ACCESS: u8 = 1;
    pub(crate) const SUBMIT: u8 = 2;
    pub(crate) const POINT: u8 = 3;
    pub(crate) const SYNC: u8 = 4;
    pub(crate) const FRAME: u8 = 5;
    pub(crate) const REGION: u8 = 6;
    pub(crate) const END: u8 = 7;
    pub(crate) const COUNTERS: u8 = 8;
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn thread() -> u64 {
    // SAFETY: no arguments.
    u64::from(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() })
}

impl Stream {
    fn open(path: &std::path::Path, cfg: &Config) -> Option<Stream> {
        let mut file = match std::fs::File::create(path) {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(target: "victory_gpu::memclass", ?path, %error, "no memclass stream");
                return None;
            }
        };
        let mut head = STREAM_MAGIC.to_vec();
        head.extend_from_slice(&STREAM_VERSION.to_le_bytes());
        head.push(cfg.shift as u8);
        head.push(0);
        file.write_all(&head).ok()?;
        // Counters of others start with the stream (`counters`).
        #[cfg(feature = "diagnostics")]
        crate::copystat::keep_totals();
        Some(Stream {
            buf: Mutex::new((Vec::with_capacity(1 << 20), 0, cfg.stream_frames)),
            file: Mutex::new(Some(file)),
            on: AtomicBool::new(true),
            written: AtomicU64::new(head.len() as u64),
            began: std::time::Instant::now(),
        })
    }

    fn with(&self, record: impl FnOnce(&mut Vec<u8>, &mut usize)) {
        if !self.on.load(Relaxed) {
            return;
        }
        let mut b = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        let (buf, last, _) = &mut *b;
        record(buf, last);
    }

    fn access(&self, flag: u16, seq: u64, at: usize, len: usize) {
        let tid = thread();
        self.with(|b, last| {
            b.push(rec::ACCESS);
            varint(b, u64::from(flag));
            varint(b, tid);
            varint(b, seq);
            varint(b, zigzag(at as i64 - *last as i64));
            varint(b, len as u64);
            *last = at;
        });
    }

    fn sync(&self, kind: Sync, at: usize, len: usize, us: u64) {
        self.with(|b, _| {
            b.push(rec::SYNC);
            b.push(kind as u8);
            varint(b, at as u64);
            varint(b, len as u64);
            varint(b, us);
        });
    }

    fn submit(&self, seq: u64, lists: usize) {
        self.with(|b, _| {
            b.push(rec::SUBMIT);
            varint(b, seq);
            varint(b, lists as u64);
        });
    }

    fn point(&self, p: Point, seq: u64) {
        self.with(|b, _| {
            b.push(rec::POINT);
            b.push(p as u8);
            varint(b, seq);
        });
    }

    fn region(&self, kind: u8, v: [usize; 4]) {
        self.with(|b, _| {
            b.push(rec::REGION);
            b.push(kind);
            for x in v {
                varint(b, x as u64);
            }
        });
    }

    /// Ends frame `frame`: its records go to the file; past the frames
    /// asked for, the stream ends.
    /// Ends the stream after frame `frame` (once: an ended stream stays
    /// as it is).
    fn finish(&self, frame: u32) {
        if !self.on.swap(false, Relaxed) {
            return;
        }
        let out = {
            let mut b = self.buf.lock().unwrap_or_else(|e| e.into_inner());
            b.0.push(rec::END);
            varint(&mut b.0, u64::from(frame));
            std::mem::take(&mut b.0)
        };
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut fh) = file.take() {
            match fh.write_all(&out).and_then(|()| fh.flush()) {
                Ok(()) => {
                    self.written.fetch_add(out.len() as u64, Relaxed);
                }
                Err(error) => {
                    tracing::warn!(target: "victory_gpu::memclass", %error, "memclass stream: END not written");
                }
            }
        }
    }

    fn frame(&self, frame: u32, counters: [u64; 4]) {
        if !self.on.load(Relaxed) {
            return;
        }
        let (out, end) = {
            let mut b = self.buf.lock().unwrap_or_else(|e| e.into_inner());
            b.0.push(rec::FRAME);
            varint(&mut b.0, u64::from(frame));
            b.0.push(rec::COUNTERS);
            for c in counters {
                varint(&mut b.0, c);
            }
            // 0: no end.
            let end = b.2 == 1;
            b.2 = b.2.saturating_sub(1);
            if end {
                b.0.push(rec::END);
                varint(&mut b.0, u64::from(frame));
                self.on.store(false, Relaxed);
            }
            (std::mem::replace(&mut b.0, Vec::with_capacity(1 << 20)), end)
        };
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if file.is_some() {
            self.written.fetch_add(out.len() as u64, Relaxed);
        }
        if let Some(fh) = file.as_mut()
            && let Err(error) = fh.write_all(&out)
        {
            tracing::warn!(target: "victory_gpu::memclass", %error, "memclass stream write failed: ended");
            self.on.store(false, Relaxed);
            *file = None;
            return;
        }
        if end {
            *file = None;
        }
    }
}

#[cfg(test)]
#[path = "memclass_tests.rs"]
pub(crate) mod tests;
