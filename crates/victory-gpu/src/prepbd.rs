//! Diagnostics: where the time preparing a submission goes, part by part
//! (`VICTORY_PREP_BREAKDOWN=1`, off by default), logged as per-frame sums
//! with the frame-rate line. The frame line's "preparing" minus its copies
//! and textures left a rest no counter named (with the BVH builder on,
//! +188 ms a frame on a 3070); this names the parts of `Host::execute`
//! before the lists run, of `crate::arena::provide`, and of the steps that
//! were timed only as a whole.
//!
//! Off, every call is a load of a set-once flag and a branch: no clock, no
//! lock, no allocation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The parts timed (microseconds) or counted.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// `Host::execute`: from taking the submission lock to the lists going
    /// to the queue (what the frame line calls preparing), and the wait for
    /// the lock before it (not part of preparing).
    Total,
    LockWait,
    Evict,
    SyncHeaps,
    Provide,
    Uploads,
    /// Each list's submission-time steps and preludes, then the last batch
    /// of refreshes, then the copies in the GPU's memory and their bitmaps.
    Deferred,
    LastFlush,
    Card,
    /// `crate::arena::provide`: regions and the reservation; the builder's
    /// pools (`around`); its input buffers scanned; the pages found there and
    /// the pages asked for, sorted (the scan included); slots given; evictions and unwatched pages sent again; the
    /// pages filled going up; the title's writes going up.
    ArenaRegions,
    ArenaAround,
    ArenaScan,
    ArenaSort,
    ArenaGive,
    ArenaReads,
    ArenaFill,
    ArenaRefresh,
    /// Of the refresh: the write-watch queries.
    ArenaWatch,
    /// In the arena's uploads: reading title memory into the stage buffer,
    /// and waiting for each batch to run on the GPU (counted too).
    UploadRead,
    UploadWait,
    UploadWaits,
    /// Bytes the scan read.
    ScanBytes,
    /// `crate::memory::prepare`: page tables written again, and the walk
    /// over their copies.
    MemoryRewrite,
    MemoryWalk,
    /// Steps marking ranges shaders write (after their batch flush, which
    /// the copies count).
    GpuWriteMark,
}
const PARTS: usize = 25;
const _: () = assert!(Part::GpuWriteMark as usize + 1 == PARTS);
/// Per submission group (`crate::costsplit`; all in one while that is off).
static SUM: [[AtomicU64; PARTS]; crate::costsplit::GROUPS] =
    [const { [const { AtomicU64::new(0) }; PARTS] }; crate::costsplit::GROUPS];

/// Whether the breakdown is on (`VICTORY_PREP_BREAKDOWN=1`; a test sets
/// its own on its thread).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PREP_BREAKDOWN").is_ok_and(|v| v.trim() == "1"))
}

/// The time now, if the breakdown is on.
pub(crate) fn now() -> Option<std::time::Instant> {
    enabled().then(std::time::Instant::now)
}

/// Adds the time since `from` (from [`now`]) to `part`.
pub(crate) fn since(part: Part, from: Option<std::time::Instant>) {
    if let Some(from) = from {
        SUM[crate::costsplit::current()][part as usize].fetch_add(from.elapsed().as_micros() as u64, Relaxed);
    }
}

/// Adds `n` to `part` (a count or bytes), if the breakdown is on.
pub(crate) fn add(part: Part, n: u64) {
    if enabled() {
        SUM[crate::costsplit::current()][part as usize].fetch_add(n, Relaxed);
    }
}

/// Times its scope into a part (nothing when the breakdown is off).
pub(crate) struct Span(Part, Option<std::time::Instant>);

impl Drop for Span {
    fn drop(&mut self) {
        since(self.0, self.1);
    }
}

/// A [`Span`] over the rest of the caller's scope.
pub(crate) fn span(part: Part) -> Span {
    Span(part, now())
}

/// Takes the sums since the last call, per group.
fn take_groups() -> [[u64; PARTS]; crate::costsplit::GROUPS] {
    std::array::from_fn(|g| std::array::from_fn(|i| SUM[g][i].swap(0, Relaxed)))
}

/// Takes the sums since the last call, all groups together.
#[cfg(test)]
fn take() -> [u64; PARTS] {
    sum_groups(&take_groups())
}

fn sum_groups(groups: &[[u64; PARTS]; crate::costsplit::GROUPS]) -> [u64; PARTS] {
    std::array::from_fn(|i| groups.iter().map(|g| g[i]).sum())
}

/// Logs the sums per frame over `frames` frames, if the breakdown is on.
pub(crate) fn log(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let groups = take_groups();
    log_one("", &sum_groups(&groups), frames);
    // With `VICTORY_COST_SPLIT`: the same per submission group.
    if crate::costsplit::enabled() {
        for (g, name) in crate::costsplit::GROUP_NAMES.iter().enumerate() {
            log_one(&format!(" ({name} submissions)"), &groups[g], frames);
        }
    }
}

fn log_one(group: &str, s: &[u64; PARTS], frames: f64) {
    let ms = |p: Part| s[p as usize] as f64 / 1e3 / frames;
    let named = [Part::Evict, Part::SyncHeaps, Part::Provide, Part::Uploads, Part::Deferred, Part::LastFlush, Part::Card]
        .iter()
        .map(|&p| ms(p))
        .sum::<f64>();
    // (The sort includes the scan.)
    let arena = [
        Part::ArenaRegions,
        Part::ArenaAround,
        Part::ArenaSort,
        Part::ArenaGive,
        Part::ArenaReads,
        Part::ArenaFill,
        Part::ArenaRefresh,
    ]
    .iter()
    .map(|&p| ms(p))
    .sum::<f64>();
    tracing::info!(
        "prep breakdown per frame{group}: total {:.1} ms (lock wait before it {:.1}); evict {:.1}, heap sync {:.1}, \
         arena {:.1}, heap uploads {:.1}, deferred steps {:.1}, last flush {:.1}, card marks {:.1}, other {:.1}; \
         arena: regions {:.1}, around {:.1}, sort {:.1} (of it scan {:.1}, {:.1} MB), give {:.1}, reads {:.1}, fill {:.1}, \
         refresh {:.1} (watch {:.1}), other {:.1}; arena uploads: reading title memory {:.1}, GPU waits {:.1} ({:.1} waits); \
         memory prepare: rewrite {:.1}, walk {:.1}; GPU-write marks {:.1}",
        ms(Part::Total),
        ms(Part::LockWait),
        ms(Part::Evict),
        ms(Part::SyncHeaps),
        ms(Part::Provide),
        ms(Part::Uploads),
        ms(Part::Deferred),
        ms(Part::LastFlush),
        ms(Part::Card),
        ms(Part::Total) - named,
        ms(Part::ArenaRegions),
        ms(Part::ArenaAround),
        ms(Part::ArenaSort),
        ms(Part::ArenaScan),
        s[Part::ScanBytes as usize] as f64 / 1e6 / frames,
        ms(Part::ArenaGive),
        ms(Part::ArenaReads),
        ms(Part::ArenaFill),
        ms(Part::ArenaRefresh),
        ms(Part::ArenaWatch),
        ms(Part::Provide) - arena,
        ms(Part::UploadRead),
        ms(Part::UploadWait),
        s[Part::UploadWaits as usize] as f64 / frames,
        ms(Part::MemoryRewrite),
        ms(Part::MemoryWalk),
        ms(Part::GpuWriteMark),
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// The breakdown on or off for this thread's test, whatever the
        /// environment says.
        pub(crate) static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        /// Allocations made on this thread (counted by the test binary's
        /// allocator below).
        static ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// The test binary's allocator: the system's, counting each thread's
    /// allocations.
    struct Counting;

    // SAFETY: every call goes to the system allocator unchanged; only counted.
    unsafe impl std::alloc::GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
            // SAFETY: the caller's layout, passed on.
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
            // SAFETY: as above.
            unsafe { std::alloc::System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
            // SAFETY: a block this allocator handed out with this layout.
            unsafe { std::alloc::System.dealloc(p, layout) }
        }

        unsafe fn realloc(&self, p: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
            // SAFETY: as above.
            unsafe { std::alloc::System.realloc(p, layout, size) }
        }
    }

    #[global_allocator]
    static ALLOC: Counting = Counting;

    /// The sums are shared by every test thread: the tests that change them
    /// run one at a time.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sums() -> [u64; PARTS] {
        std::array::from_fn(|i| SUM.iter().map(|g| g[i].load(Relaxed)).sum())
    }

    /// Allocations made on this thread so far.
    pub(crate) fn allocations() -> u64 {
        ALLOCATIONS.with(std::cell::Cell::get)
    }

    #[test]
    fn off_it_records_nothing_allocates_nothing_and_reads_no_clock() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|f| f.set(Some(false)));
        let before = sums();
        let allocations = ALLOCATIONS.with(std::cell::Cell::get);
        for _ in 0..1000 {
            assert!(now().is_none(), "off: no clock read");
            let _span = span(Part::Provide);
            since(Part::ArenaScan, now());
            add(Part::ScanBytes, 4096);
            add(Part::UploadWaits, 1);
            log(1.0);
        }
        assert_eq!(ALLOCATIONS.with(std::cell::Cell::get), allocations, "off: no allocation");
        assert_eq!(sums(), before, "off: nothing recorded");
        FORCED.with(|f| f.set(None));
    }

    #[test]
    fn on_it_sums_parts_and_takes_them_per_line() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|f| f.set(Some(true)));
        let _ = take();
        {
            let _span = span(Part::ArenaScan);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        add(Part::ScanBytes, 4096);
        add(Part::UploadWaits, 3);
        let s = take();
        assert!(s[Part::ArenaScan as usize] >= 1500, "the span's time: {}", s[Part::ArenaScan as usize]);
        assert_eq!(s[Part::ScanBytes as usize], 4096);
        assert_eq!(s[Part::UploadWaits as usize], 3);
        assert_eq!(take(), [0; PARTS], "taken: the next line starts from zero");
        FORCED.with(|f| f.set(None));
    }
}
