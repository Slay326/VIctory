//! Diagnostics: where a frame's time goes (`VICTORY_GPU_PROFILE=1`, off by
//! default), one line per presented frame and a summary with the
//! frame-rate line.
//!
//! - GPU time per pipeline: within each host list a timestamp wherever the
//!   work changes to another pipeline (by its name, its shaders' hashes) or
//!   to a command of the title's that binds none (texture copies, resolves,
//!   read-backs, clears, barriers); segment `i` runs from timestamp
//!   `i` to `i + 1`, the last one written when the list closes. Timestamps
//!   are written as the work before them completes, so the segments of a
//!   list add up to its time on the GPU exactly.
//! - Per submission: timestamps before and after its lists (the GPU's busy
//!   time for it), placed on the CPU's clock with the queue's clock
//!   calibration: from the lists going to the queue to the GPU starting,
//!   from the GPU finishing to the CPU waking, and the GPU's busy time
//!   outside the CPU's wait for the fence (host and GPU overlapping).
//! - CPU time of the submitting thread by part: waiting for the submission
//!   lock, the parts of preparing (giving up textures, heap sync with its
//!   own lock wait, uploads, deferred steps, card copies), handing the lists
//!   to the queue, waiting for the fence, the parts of copying back; and the
//!   frame's time outside submissions. Waits for the guard lock (on the
//!   submitting thread and on others) and title threads waiting for the
//!   view heaps' lock while a heap sync holds it.
//!
//! Off, every hook is a load of a set-once flag and a branch: no clock, no
//! lock, no allocation, no timestamp.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the profile is on (`VICTORY_GPU_PROFILE=1`; a test sets its own
/// on its thread).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_GPU_PROFILE").is_ok_and(|v| v.trim() == "1");
        ARMED.store(on, Relaxed);
        on
    })
}

/// Set once [`enabled`] found the profile on: hooks that must not allocate
/// (the guard lock, which the fault handler takes) look only at this. They
/// time nothing before the first submission asked.
static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether hooks that must not read the environment time their waits.
pub(crate) fn armed() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::FORCED.with(std::cell::Cell::get) {
        return on;
    }
    ARMED.load(Relaxed)
}

/// The guard lock was waited for since `from` (from [`armed`]): on the
/// submitting thread or another.
pub(crate) fn guard_waited(from: Option<std::time::Instant>) {
    let Some(from) = from else {
        return;
    };
    let part = if crate::host::in_submission() { Part::GuardWaitSubmitter } else { Part::GuardWaitOther };
    FRAME[part as usize].fetch_add(from.elapsed().as_micros() as u64, Relaxed);
}

/// CPU time (microseconds) and counts summed per frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Part {
    /// Submissions (count).
    Submissions = 0,
    /// The submitting thread waiting for the submission lock.
    LockWait,
    /// Preparing: textures given up (`evict_unused`).
    Evict,
    /// Preparing: descriptors the title copied into its heaps (all of
    /// `sync_heaps`, with its lock wait).
    SyncHeaps,
    /// Of it: waiting for the view heaps' lock.
    SyncHeapsLockWait,
    /// Preparing: texture uploads by a list of ours.
    Uploads,
    /// Preparing: each list's deferred steps (title memory copies, texture
    /// refreshes) and preludes, and the batch flush.
    Deferred,
    /// Preparing: copies in the GPU's memory up and the marks list.
    Card,
    /// Preparing, in all (from holding the lock to the lists going to the
    /// queue: the frame line's "preparing").
    Prep,
    /// Handing the lists to the queue and signalling the fence.
    Queue,
    /// Waiting for the fence.
    FenceWait,
    /// Copying back: the checks of lists using title memory and the ranges
    /// to go back.
    BackChecks,
    /// Copying back: copies in the GPU's memory down, marked pages back.
    BackMarks,
    /// Copying back: declared ranges compared and copied back.
    BackCopy,
    /// Copying back: the lists' work after it and their deferred writes.
    BackAfter,
    /// Copying back, in all (the frame line's "copying back").
    Back,
    /// Holding the lock in all (the submission from lock to release).
    Execute,
    /// The GPU's busy time between a submission's timestamps.
    GpuBusy,
    /// Submissions with both timestamps read (count).
    Stamped,
    /// From the lists going to the queue to the GPU starting them.
    Launch,
    /// From the GPU finishing to the waiting CPU waking.
    Wake,
    /// The GPU busy while the submitting thread was not waiting for it.
    Overlap,
    /// Submissions placed on the CPU's clock (count).
    Placed,
    /// The guard lock waited for, on the submitting thread.
    GuardWaitSubmitter,
    /// The guard lock waited for, on other threads.
    GuardWaitOther,
    /// Title threads waiting for the view heaps' lock (held by heap syncs).
    MirrorWait,
}
pub(crate) const PARTS: usize = 26;
const _: () = assert!(Part::MirrorWait as usize + 1 == PARTS);

/// Per-frame sums (taken by [`frame_done`]).
static FRAME: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
/// Sums since the last summary (taken by [`summary`]).
static SUMMARY: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
/// Wall time of the frames since the last summary (microseconds).
static SUMMARY_WALL: AtomicU64 = AtomicU64::new(0);
/// Frames since the last summary.
static SUMMARY_FRAMES: AtomicU64 = AtomicU64::new(0);

/// The time now, if the profile is on (for [`since`]).
pub(crate) fn now() -> Option<std::time::Instant> {
    enabled().then(std::time::Instant::now)
}

/// Adds the time since `from` (from [`now`]) to `part`; returns it in
/// microseconds (0 off).
pub(crate) fn since(part: Part, from: Option<std::time::Instant>) -> u64 {
    match from {
        Some(from) => {
            let us = from.elapsed().as_micros() as u64;
            FRAME[part as usize].fetch_add(us, Relaxed);
            us
        }
        None => 0,
    }
}

/// Adds `n` to `part` (microseconds or a count), if the profile is on.
pub(crate) fn add(part: Part, n: u64) {
    if enabled() {
        FRAME[part as usize].fetch_add(n, Relaxed);
    }
}

/// Keys of segments that are no pipeline: commands of the title's that
/// bind none, and a list past its timestamps.
pub(crate) const COPY: u32 = 0;
pub(crate) const RESOLVE: u32 = 1;
pub(crate) const READ_BACK: u32 = 2;
pub(crate) const CLEAR: u32 = 3;
pub(crate) const BARRIER: u32 = 4;
pub(crate) const OVERFLOW: u32 = 5;
const SPECIAL: [&str; 6] = [
    "(title texture copies)",
    "(title resolves)",
    "(title read-backs)",
    "(title clears)",
    "(title barriers)",
    "(overflow: list past its timestamps)",
];

/// Pipeline names by key (the specials first) and keys by name.
struct Names {
    names: Vec<String>,
    keys: std::collections::HashMap<String, u32>,
}

static NAMES: std::sync::Mutex<Option<Names>> = std::sync::Mutex::new(None);

fn with_names<R>(f: impl FnOnce(&mut Names) -> R) -> R {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let names = names.get_or_insert_with(|| Names {
        names: SPECIAL.iter().map(|s| (*s).to_owned()).collect(),
        keys: std::collections::HashMap::new(),
    });
    f(names)
}

/// The key of the pipeline named `name`.
pub(crate) fn key(name: &str) -> u32 {
    with_names(|n| {
        if let Some(&k) = n.keys.get(name) {
            return k;
        }
        let k = n.names.len() as u32;
        n.names.push(name.to_owned());
        n.keys.insert(name.to_owned(), k);
        k
    })
}

fn name(key: u32) -> String {
    with_names(|n| n.names.get(key as usize).cloned().unwrap_or_else(|| format!("(key {key})")))
}

/// GPU microseconds and segments per key, since the last summary.
static PIPELINES: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());
/// GPU microseconds in list segments this frame (for the frame line).
static FRAME_SEGMENTS: AtomicU64 = AtomicU64::new(0);

/// Timestamps per list; work past them goes to [`OVERFLOW`].
pub(crate) const SLOTS: u32 = 8192;

/// A list's segments, in recording order: segment `i` runs from timestamp
/// `i` to timestamp `i + 1` (the last one written when the list closes).
#[derive(Default)]
pub(crate) struct Segments {
    keys: Vec<u32>,
    /// The host pipeline object last looked up, and its key (no name
    /// lookup while a list keeps drawing with one pipeline).
    cached: Option<(usize, u32)>,
    /// The list was closed with its last timestamp at this slot.
    closed: Option<u32>,
}

impl Segments {
    /// The host pipeline object `pso` named `name` does the work from now
    /// on: the slot to write a timestamp to, if the work changes.
    pub(crate) fn pipeline(&mut self, pso: usize, name: &str) -> Option<u32> {
        if !enabled() || self.closed.is_some() {
            return None;
        }
        let key = match self.cached {
            Some((p, k)) if p == pso => k,
            _ => {
                let k = key(name);
                self.cached = Some((pso, k));
                k
            }
        };
        self.push(key)
    }

    /// A command of the title's binding no pipeline (a special key) from
    /// now on: the slot to write a timestamp to, if the work changes.
    pub(crate) fn other(&mut self, key: u32) -> Option<u32> {
        if !enabled() || self.closed.is_some() {
            return None;
        }
        self.push(key)
    }

    fn push(&mut self, key: u32) -> Option<u32> {
        let last = self.keys.last().copied();
        if last == Some(key) || last == Some(OVERFLOW) {
            return None;
        }
        // The last slot before the closing one: everything after it is
        // overflow.
        let key = if self.keys.len() as u32 + 2 >= SLOTS { OVERFLOW } else { key };
        self.keys.push(key);
        Some(self.keys.len() as u32 - 1)
    }

    /// The list closes: the slot of its last timestamp, if it has segments
    /// (timestamps `0..=slot` are then resolved).
    pub(crate) fn close(&mut self) -> Option<u32> {
        if self.keys.is_empty() || self.closed.is_some() {
            return None;
        }
        let slot = self.keys.len() as u32;
        self.closed = Some(slot);
        Some(slot)
    }

    /// How many timestamps a closed list resolved.
    pub(crate) fn resolved(&self) -> Option<usize> {
        self.closed.map(|slot| slot as usize + 1)
    }

    /// The list's GPU microseconds per segment key from its timestamps
    /// (ticks at `frequency` per second): segment `i` goes to `keys[i]`.
    pub(crate) fn per_key(&self, stamps: &[u64], frequency: u64) -> Vec<(u32, u64)> {
        let mut out = Vec::with_capacity(self.keys.len());
        for (i, &key) in self.keys.iter().enumerate() {
            let (Some(&a), Some(&b)) = (stamps.get(i), stamps.get(i + 1)) else {
                break;
            };
            out.push((key, ticks_us(b.saturating_sub(a), frequency)));
        }
        out
    }

    /// Adds the list's GPU time per key to the sums.
    pub(crate) fn attribute(&self, stamps: &[u64], frequency: u64) {
        let per = self.per_key(stamps, frequency);
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        for (key, us) in per {
            let k = key as usize;
            if pipelines.len() <= k {
                pipelines.resize(k + 1, (0, 0));
            }
            pipelines[k].0 += us;
            pipelines[k].1 += 1;
            FRAME_SEGMENTS.fetch_add(us, Relaxed);
        }
    }
}

/// Timestamp ticks at `frequency` per second, in microseconds.
pub(crate) fn ticks_us(ticks: u64, frequency: u64) -> u64 {
    (u128::from(ticks) * 1_000_000 / u128::from(frequency.max(1))) as u64
}

/// A submission on the CPU's clock (microseconds on one time line): the
/// lists handed to the queue, the CPU starting and ending its wait, and the
/// GPU's first and last timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placed {
    pub(crate) queued: f64,
    pub(crate) wait_from: f64,
    pub(crate) wait_to: f64,
    pub(crate) gpu_from: f64,
    pub(crate) gpu_to: f64,
}

impl Placed {
    /// (launch, wake, overlap) in microseconds: from queued to the GPU
    /// starting, from the GPU ending to the CPU waking, and the GPU's busy
    /// time outside the CPU's wait.
    pub(crate) fn split(&self) -> (u64, u64, u64) {
        let launch = (self.gpu_from - self.queued).max(0.0);
        let wake = (self.wait_to - self.gpu_to).max(0.0);
        let busy = (self.gpu_to - self.gpu_from).max(0.0);
        let inside = (self.gpu_to.min(self.wait_to) - self.gpu_from.max(self.wait_from)).max(0.0);
        (launch as u64, wake as u64, (busy - inside).max(0.0) as u64)
    }
}

/// A GPU timestamp on the CPU's clock in microseconds, from a calibration
/// pair (GPU ticks at `gpu_frequency`, QPC ticks at `cpu_frequency`).
pub(crate) fn gpu_to_cpu_us(stamp: u64, calibration: (u64, u64), gpu_frequency: u64, cpu_frequency: u64) -> f64 {
    let (gpu, cpu) = calibration;
    let cpu_us = cpu as f64 * 1e6 / cpu_frequency.max(1) as f64;
    cpu_us + (stamp as f64 - gpu as f64) * 1e6 / gpu_frequency.max(1) as f64
}

/// A submission placed on the CPU's clock: its launch, wake and overlap
/// added.
pub(crate) fn placed(p: &Placed) {
    if !enabled() {
        return;
    }
    let (launch, wake, overlap) = p.split();
    add(Part::Launch, launch);
    add(Part::Wake, wake);
    add(Part::Overlap, overlap);
    add(Part::Placed, 1);
}

/// One frame's sums as microseconds per part and the wall time.
pub(crate) struct Frame {
    pub(crate) parts: [u64; PARTS],
    pub(crate) wall: u64,
    pub(crate) segments: u64,
}

impl Frame {
    fn ms(&self, p: Part) -> f64 {
        self.parts[p as usize] as f64 / 1e3
    }

    /// The frame's time named: outside submissions plus the lock wait, the
    /// preparing, the queue, the fence wait and the copying back
    /// (milliseconds), and what of the wall time is left (in submissions,
    /// none of those).
    pub(crate) fn accounting(&self) -> (f64, f64) {
        let wall = self.wall as f64 / 1e3;
        let inside = self.ms(Part::Execute) + self.ms(Part::LockWait);
        let outside = (wall - inside).max(0.0);
        let named = outside
            + self.ms(Part::LockWait)
            + self.ms(Part::Prep)
            + self.ms(Part::Queue)
            + self.ms(Part::FenceWait)
            + self.ms(Part::Back);
        (named, wall - named)
    }

    pub(crate) fn line(&self, n: u64, vram: Option<[(f64, f64); 2]>) -> String {
        let ms = |p: Part| self.ms(p);
        let wall = self.wall as f64 / 1e3;
        let inside = ms(Part::Execute) + ms(Part::LockWait);
        let (named, left) = self.accounting();
        let prep_named = [Part::Evict, Part::SyncHeaps, Part::Uploads, Part::Deferred, Part::Card]
            .iter()
            .map(|&p| ms(p))
            .sum::<f64>();
        let back_named = [Part::BackChecks, Part::BackMarks, Part::BackCopy, Part::BackAfter]
            .iter()
            .map(|&p| ms(p))
            .sum::<f64>();
        let vram = vram.map_or("vram n/a".to_owned(), |[(local, budget), (shared, _)]| {
            format!("vram dedicated {local:.0} of {budget:.0} MB, shared {shared:.0} MB")
        });
        format!(
            "gpu profile frame {n}: wall {wall:.1} ms; submissions {}, held {:.1} ms; outside submissions {:.1}; \
             lock wait {:.1}; prep {:.1} (evict {:.1}, heap sync {:.1} of it lock wait {:.1}, uploads {:.1}, \
             deferred {:.1}, card {:.1}, other {:.1}); queue {:.1}; fence wait {:.1}; back {:.1} (checks {:.1}, \
             marks {:.1}, copy {:.1}, after {:.1}, other {:.1}); named {named:.1}, left {left:.1}; \
             gpu busy {:.1} ({} stamped), in list segments {:.1}, launch {:.1}, wake {:.1}, overlap {:.1} \
             ({} placed); guard wait submitter {:.1} others {:.1}; mirror wait {:.1}; {vram}",
            self.parts[Part::Submissions as usize],
            inside,
            (wall - inside).max(0.0),
            ms(Part::LockWait),
            ms(Part::Prep),
            ms(Part::Evict),
            ms(Part::SyncHeaps),
            ms(Part::SyncHeapsLockWait),
            ms(Part::Uploads),
            ms(Part::Deferred),
            ms(Part::Card),
            ms(Part::Prep) - prep_named,
            ms(Part::Queue),
            ms(Part::FenceWait),
            ms(Part::Back),
            ms(Part::BackChecks),
            ms(Part::BackMarks),
            ms(Part::BackCopy),
            ms(Part::BackAfter),
            ms(Part::Back) - back_named,
            ms(Part::GpuBusy),
            self.parts[Part::Stamped as usize],
            self.segments as f64 / 1e3,
            ms(Part::Launch),
            ms(Part::Wake),
            ms(Part::Overlap),
            self.parts[Part::Placed as usize],
            ms(Part::GuardWaitSubmitter),
            ms(Part::GuardWaitOther),
            ms(Part::MirrorWait),
        )
    }
}

/// A frame was presented: its line (with the video memory, local and
/// shared, as DXGI tells it), and its sums go to the summary.
pub(crate) fn frame_done(vram: impl FnOnce() -> Option<[(f64, f64); 2]>) {
    if !enabled() {
        return;
    }
    static LAST: std::sync::Mutex<Option<(std::time::Instant, u64)>> = std::sync::Mutex::new(None);
    let now = std::time::Instant::now();
    let (wall, n) = {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let (wall, n) = match *last {
            Some((at, n)) => ((now - at).as_micros() as u64, n + 1),
            None => (0, 0),
        };
        *last = Some((now, n));
        (wall, n)
    };
    let parts: [u64; PARTS] = std::array::from_fn(|i| FRAME[i].swap(0, Relaxed));
    let segments = FRAME_SEGMENTS.swap(0, Relaxed);
    // The first frame has no start: its sums are dropped.
    if n == 0 {
        return;
    }
    for (i, v) in parts.iter().enumerate() {
        SUMMARY[i].fetch_add(*v, Relaxed);
    }
    SUMMARY_WALL.fetch_add(wall, Relaxed);
    SUMMARY_FRAMES.fetch_add(1, Relaxed);
    let frame = Frame { parts, wall, segments };
    tracing::info!("{}", frame.line(n, vram()));
}

/// How many of the costliest keys hold `share` of `total` (sorted
/// descending).
pub(crate) fn keys_for_share(sorted: &[u64], share: f64) -> usize {
    let total: u64 = sorted.iter().sum();
    if total == 0 {
        return 0;
    }
    let mut sum = 0u64;
    for (i, us) in sorted.iter().enumerate() {
        sum += us;
        if sum as f64 >= share * total as f64 {
            return i + 1;
        }
    }
    sorted.len()
}

/// The summary over the frames since the last one: the parts per frame and
/// every key's GPU time (costliest first).
pub(crate) fn summary() {
    if !enabled() {
        return;
    }
    let frames = SUMMARY_FRAMES.swap(0, Relaxed) as f64;
    if frames <= 0.0 {
        return;
    }
    let parts: [u64; PARTS] = std::array::from_fn(|i| SUMMARY[i].swap(0, Relaxed));
    let wall = SUMMARY_WALL.swap(0, Relaxed);
    let pipelines = std::mem::take(&mut *PIPELINES.lock().unwrap_or_else(|e| e.into_inner()));
    let (head, list) = summary_lines(&parts, wall, &pipelines, frames, name);
    tracing::info!("{head}");
    tracing::info!("{list}");
}

/// The summary's two lines: parts per frame, and the keys.
fn summary_lines(
    parts: &[u64; PARTS],
    wall: u64,
    pipelines: &[(u64, u64)],
    frames: f64,
    name: impl Fn(u32) -> String,
) -> (String, String) {
    let ms = |us: u64| us as f64 / 1e3 / frames;
    let mut keyed: Vec<(u32, u64, u64)> = pipelines
        .iter()
        .enumerate()
        .filter(|(_, (us, n))| *us > 0 || *n > 0)
        .map(|(k, &(us, n))| (k as u32, us, n))
        .collect();
    keyed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let sorted: Vec<u64> = keyed.iter().map(|k| k.1).collect();
    let total: u64 = sorted.iter().sum();
    let head = format!(
        "gpu profile summary over {frames:.0} frames: wall {:.1} ms/frame, held {:.1}, lock wait {:.1}, prep {:.1}, \
         queue {:.1}, fence wait {:.1}, back {:.1}, gpu busy {:.1}, in list segments {:.1} in {} keys \
         (50% in {}, 80% in {}, 90% in {}), launch {:.1}, wake {:.1}, overlap {:.1}, guard wait submitter {:.1} \
         others {:.1}, heap sync {:.1} (lock wait {:.1}), mirror wait {:.1}",
        ms(wall),
        ms(parts[Part::Execute as usize]),
        ms(parts[Part::LockWait as usize]),
        ms(parts[Part::Prep as usize]),
        ms(parts[Part::Queue as usize]),
        ms(parts[Part::FenceWait as usize]),
        ms(parts[Part::Back as usize]),
        ms(parts[Part::GpuBusy as usize]),
        ms(total),
        keyed.len(),
        keys_for_share(&sorted, 0.5),
        keys_for_share(&sorted, 0.8),
        keys_for_share(&sorted, 0.9),
        ms(parts[Part::Launch as usize]),
        ms(parts[Part::Wake as usize]),
        ms(parts[Part::Overlap as usize]),
        ms(parts[Part::GuardWaitSubmitter as usize]),
        ms(parts[Part::GuardWaitOther as usize]),
        ms(parts[Part::SyncHeaps as usize]),
        ms(parts[Part::SyncHeapsLockWait as usize]),
        ms(parts[Part::MirrorWait as usize]),
    );
    // Every key, raw: microseconds and segments over the summary's frames.
    let mut list = format!("gpu profile keys over {frames:.0} frames (us, segments, name):");
    for (k, us, n) in &keyed {
        list.push_str(&format!(" {us} {n} {} ;;", name(*k)));
    }
    (head, list)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// The profile on or off for this thread's test, whatever the
        /// environment says.
        pub(crate) static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    /// The sums are shared by every test thread: the tests that change them
    /// run one at a time.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sums() -> (Vec<u64>, Vec<(u64, u64)>, u64) {
        (
            FRAME.iter().chain(SUMMARY.iter()).map(|a| a.load(Relaxed)).collect(),
            PIPELINES.lock().unwrap().clone(),
            FRAME_SEGMENTS.load(Relaxed),
        )
    }

    fn forced<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|c| c.set(Some(on)));
        let r = f();
        FORCED.with(|c| c.set(None));
        r
    }

    #[test]
    fn off_it_records_nothing_and_stamps_nothing() {
        forced(false, || {
            let before = sums();
            let mut s = Segments::default();
            for i in 0..1000 {
                assert_eq!(s.pipeline(i, "[a]"), None, "off: no timestamp");
                assert_eq!(s.other(COPY), None, "off: no timestamp");
                assert!(now().is_none(), "off: no clock");
                assert!(!armed(), "off: lock waits untimed");
                guard_waited(armed().then(std::time::Instant::now));
                add(Part::FenceWait, 5);
                since(Part::Prep, now());
                placed(&Placed { queued: 0.0, wait_from: 1.0, wait_to: 9.0, gpu_from: 2.0, gpu_to: 8.0 });
                frame_done(|| panic!("off: no video memory query"));
                summary();
            }
            assert_eq!(s.close(), None, "off: no closing timestamp");
            assert_eq!(s.resolved(), None);
            assert_eq!(sums(), before, "off: nothing recorded");
        });
    }

    #[test]
    fn segments_go_to_the_work_between_their_timestamps() {
        forced(true, || {
            let mut s = Segments::default();
            // Pipeline a draws twice, a copy, a again, b, a barrier, b.
            assert_eq!(s.pipeline(0x10, "[a]"), Some(0));
            assert_eq!(s.pipeline(0x10, "[a]"), None, "same pipeline: no new timestamp");
            assert_eq!(s.other(COPY), Some(1));
            assert_eq!(s.other(COPY), None, "copies in a row: one segment");
            assert_eq!(s.pipeline(0x10, "[a]"), Some(2), "back to a after a copy: a new segment");
            assert_eq!(s.pipeline(0x20, "[b]"), Some(3));
            assert_eq!(s.other(BARRIER), Some(4));
            assert_eq!(s.pipeline(0x20, "[b]"), Some(5));
            // Another host object with the same shaders is the same work.
            assert_eq!(s.pipeline(0x30, "[b]"), None);
            assert_eq!(s.close(), Some(6));
            assert_eq!(s.resolved(), Some(7));
            assert_eq!(s.pipeline(0x40, "[c]"), None, "closed: nothing more");
            let (a, b) = (key("[a]"), key("[b]"));
            assert!(a as usize >= SPECIAL.len() && b as usize >= SPECIAL.len() && a != b);
            // 1 MHz ticks: a 100, copy 7, a 20, b 1000, barrier 3, b 500.
            let stamps = [0, 100, 107, 127, 1127, 1130, 1630];
            assert_eq!(
                s.per_key(&stamps, 1_000_000),
                vec![(a, 100), (COPY, 7), (a, 20), (b, 1000), (BARRIER, 3), (b, 500)]
            );
            // The segments add up to the list's time between its first and
            // last timestamp.
            let sum: u64 = s.per_key(&stamps, 1_000_000).iter().map(|x| x.1).sum();
            assert_eq!(sum, 1630);
            // Missing stamps: only whole segments count.
            assert_eq!(s.per_key(&stamps[..3], 1_000_000), vec![(a, 100), (COPY, 7)]);
            // Summed per key.
            let _ = std::mem::take(&mut *PIPELINES.lock().unwrap());
            FRAME_SEGMENTS.store(0, Relaxed);
            s.attribute(&stamps, 1_000_000);
            let p = PIPELINES.lock().unwrap().clone();
            assert_eq!(p[a as usize], (120, 2));
            assert_eq!(p[b as usize], (1500, 2));
            assert_eq!(p[COPY as usize], (7, 1));
            assert_eq!(p[BARRIER as usize], (3, 1));
            assert_eq!(FRAME_SEGMENTS.swap(0, Relaxed), 1630);
            let _ = std::mem::take(&mut *PIPELINES.lock().unwrap());
        });
    }

    #[test]
    fn a_list_past_its_timestamps_ends_in_overflow() {
        forced(true, || {
            let mut s = Segments::default();
            for i in 0..SLOTS as usize + 10 {
                let _ = s.pipeline(i, if i % 2 == 0 { "[x]" } else { "[y]" });
            }
            let close = s.close().unwrap();
            assert!(close < SLOTS, "every timestamp fits the heap: {close}");
            assert_eq!(s.keys.last(), Some(&OVERFLOW));
            assert_eq!(s.keys.iter().filter(|&&k| k == OVERFLOW).count(), 1);
        });
    }

    #[test]
    fn placing_a_submission_on_the_cpus_clock() {
        // GPU at 10 MHz, QPC at 1 MHz; the calibration pair says GPU tick
        // 1000 was QPC 500 (500 us).
        assert_eq!(gpu_to_cpu_us(1000, (1000, 500), 10_000_000, 1_000_000), 500.0);
        assert_eq!(gpu_to_cpu_us(1100, (1000, 500), 10_000_000, 1_000_000), 510.0);
        // Queued at 0, waits from 2 to 100, the GPU busy 5..90: launch 5,
        // wake 10, busy before the wait 0 (it starts after).
        let p = Placed { queued: 0.0, wait_from: 2.0, wait_to: 100.0, gpu_from: 5.0, gpu_to: 90.0 };
        assert_eq!(p.split(), (5, 10, 0));
        // The GPU starting before the CPU waits: 1..2 overlaps the host.
        let p = Placed { queued: 0.0, wait_from: 2.0, wait_to: 100.0, gpu_from: 1.0, gpu_to: 90.0 };
        assert_eq!(p.split(), (1, 10, 1));
        // Still busy after the CPU woke (a calibration off): counted too.
        let p = Placed { queued: 0.0, wait_from: 0.0, wait_to: 50.0, gpu_from: 10.0, gpu_to: 60.0 };
        assert_eq!(p.split(), (10, 0, 10));
    }

    #[test]
    fn the_frame_line_names_its_wall_time() {
        let mut parts = [0u64; PARTS];
        parts[Part::Submissions as usize] = 3;
        parts[Part::LockWait as usize] = 1_000;
        parts[Part::Prep as usize] = 100_000;
        parts[Part::SyncHeaps as usize] = 15_000;
        parts[Part::Queue as usize] = 500;
        parts[Part::FenceWait as usize] = 180_000;
        parts[Part::Back as usize] = 45_000;
        // Held: prep + queue + wait + back + 2 ms the parts do not name.
        parts[Part::Execute as usize] = 100_000 + 500 + 180_000 + 45_000 + 2_000;
        let frame = Frame { parts, wall: 340_000, segments: 170_000 };
        let (named, left) = frame.accounting();
        // Held 327.5 + lock wait 1 = 328.5 in submissions; outside 11.5;
        // named 11.5 + 1 + 100 + 0.5 + 180 + 45 = 338, left the 2 unnamed.
        assert!((named - 338.0).abs() < 1e-9, "{named}");
        assert!((left - 2.0).abs() < 1e-9, "{left}");
        let line = frame.line(7, Some([(8300.0, 7500.0), (9000.0, 16000.0)]));
        assert!(line.starts_with("gpu profile frame 7: wall 340.0 ms; submissions 3, held 328.5 ms; outside submissions 11.5;"), "{line}");
        assert!(line.contains("prep 100.0 (evict 0.0, heap sync 15.0 of it lock wait 0.0,"), "{line}");
        assert!(line.contains("other 85.0); queue 0.5; fence wait 180.0; back 45.0"), "{line}");
        assert!(line.contains("named 338.0, left 2.0;"), "{line}");
        assert!(line.contains("in list segments 170.0"), "{line}");
        assert!(line.ends_with("vram dedicated 8300 of 7500 MB, shared 9000 MB"), "{line}");
    }

    #[test]
    fn the_summary_ranks_keys_and_tells_their_concentration() {
        assert_eq!(keys_for_share(&[50, 30, 10, 10], 0.5), 1);
        assert_eq!(keys_for_share(&[50, 30, 10, 10], 0.8), 2);
        assert_eq!(keys_for_share(&[50, 30, 10, 10], 0.9), 3);
        assert_eq!(keys_for_share(&[25, 25, 25, 25], 0.8), 4);
        assert_eq!(keys_for_share(&[], 0.5), 0);
        let mut parts = [0u64; PARTS];
        parts[Part::GpuBusy as usize] = 2_000_000;
        let pipelines = vec![(0, 0), (10_000, 4), (0, 0), (0, 0), (0, 0), (0, 0), (1_000_000, 10), (600_000, 3)];
        let names = |k: u32| format!("n{k}");
        let (head, list) = summary_lines(&parts, 4_000_000, &pipelines, 10.0, names);
        assert!(head.contains("wall 400.0 ms/frame"), "{head}");
        assert!(head.contains("gpu busy 200.0, in list segments 161.0 in 3 keys (50% in 1, 80% in 2, 90% in 2)"), "{head}");
        assert_eq!(list, "gpu profile keys over 10 frames (us, segments, name): 1000000 10 n6 ;; 600000 3 n7 ;; 10000 4 n1 ;;");
    }

    #[test]
    fn ticks() {
        assert_eq!(ticks_us(1_000_000_000, 1_000_000_000), 1_000_000);
        assert_eq!(ticks_us(5, 0), 5_000_000, "a zero frequency counts as 1");
    }
}
