//! Copy-backs on a thread of their own (`VICTORY_LANDING=1`, off by
//! default).
//!
//! Without the switch a submission's copy-back (the pages shaders marked,
//! the ranges they declared) and its bottom-of-pipe writes run on the
//! submitting thread before the next submission starts preparing. With it,
//! once the GPU is done and the bitmaps and copies in the GPU's memory have
//! come down (both use the queue, so they stay on the submitting thread),
//! what goes back into title memory "lands" here, followed by the
//! bottom-of-pipe writes and then the fence Signals queued behind the
//! submission ([`after_landing`]): the title sees a submission done only
//! once its writes are in title memory, as before.
//!
//! The next submission prepares meanwhile. What a copy-back changes is per
//! host copy (its pending ranges, snapshot segments, armed and saved pages,
//! the CPU side of a copy in the GPU's memory) and per title page (the
//! bytes, the guards, the pages showing the same memory elsewhere). None of
//! that is kept per submission, so it is not split: the next submission
//! waits for the landing at the first step that touches a copy the landing
//! writes or a title page it writes ([`touch`], the copy-back's [`Set`]),
//! and at any step that may write title memory ([`touch_all`]). Steps that
//! touch neither go ahead. Before its own copy-back a submission waits for
//! the one landing (one at a time, in order).
//!
//! A submission whose copy-back runs work of its lists (`defer_after`) or
//! withholds ranges lands on the submitting thread as before (counted): that
//! work may use the queue or the submission lock.
//!
//! The host's descriptor heaps: the next submission's heap sync runs once
//! the GPU is done with the last one (the submitting thread waited for its
//! fence) and a landing reads and writes no descriptors (texture views made
//! for reads, `crate::descriptor::make_read`, are made before it starts).

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::watch::PAGE;

/// Whether copy-backs land on their own thread (`VICTORY_LANDING=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_LANDING").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("copy-backs land on a thread of their own (VICTORY_LANDING)");
        }
        on
    })
}

/// What a landing writes: host copies (by address) and title memory
/// (disjoint page-aligned ranges in order, with the pages showing the same
/// memory elsewhere).
#[derive(Default)]
pub(crate) struct Set {
    copies: HashSet<usize>,
    ranges: Vec<(usize, usize)>,
}

impl Set {
    /// The set of a copy-back: marked pages (copy, page address) and
    /// declared ranges (copy, address, bytes).
    pub(crate) fn of(pages: &[(Arc<crate::shadow::Shadow>, usize)], declared: &[(Arc<crate::shadow::Shadow>, usize, usize)]) -> Set {
        let mut copies = HashSet::new();
        let mut numbers: Vec<usize> = Vec::new();
        for (copy, page) in pages {
            copies.insert(Arc::as_ptr(copy) as usize);
            numbers.push(page / PAGE);
        }
        for (copy, address, len) in declared {
            copies.insert(Arc::as_ptr(copy) as usize);
            if *len > 0 {
                numbers.extend(address / PAGE..(address + len).div_ceil(PAGE));
            }
        }
        numbers.sort_unstable();
        numbers.dedup();
        let elsewhere = crate::watch::shown_elsewhere(&numbers);
        numbers.extend(elsewhere.into_iter().flatten());
        Set { copies, ranges: Set::joined(numbers) }
    }

    /// Page numbers as joined address ranges.
    fn joined(mut numbers: Vec<usize>) -> Vec<(usize, usize)> {
        numbers.sort_unstable();
        numbers.dedup();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for n in numbers {
            match ranges.last_mut() {
                Some((_, end)) if *end == n * PAGE => *end += PAGE,
                _ => ranges.push((n * PAGE, (n + 1) * PAGE)),
            }
        }
        ranges
    }

    /// Whether a step touching `copy` (if any) or title memory `[from, to)`
    /// meets what the landing writes.
    pub(crate) fn meets(&self, copy: Option<usize>, from: usize, to: usize) -> bool {
        if copy.is_some_and(|c| self.copies.contains(&c)) {
            return true;
        }
        if from >= to {
            return false;
        }
        let i = self.ranges.partition_point(|&(_, end)| end <= from);
        self.ranges.get(i).is_some_and(|&(start, _)| start < to)
    }

    /// Whether the landing writes nothing.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.copies.is_empty() && self.ranges.is_empty()
    }
}

enum Job {
    /// A copy-back, and what it writes.
    Land(Box<dyn FnOnce() + Send>),
    /// Work behind it (fence Signals).
    Then(Box<dyn FnOnce() + Send>),
}

#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    /// The thread runs a job.
    busy: bool,
}

static STATE: Mutex<State> = Mutex::new(State { jobs: VecDeque::new(), busy: false });
static CHANGED: Condvar = Condvar::new();
/// The landing posted and not done yet: what it writes.
static CURRENT: Mutex<Option<Arc<Set>>> = Mutex::new(None);
/// Whether [`CURRENT`] holds one (read without its lock: most steps run
/// with no landing in flight).
static IN_FLIGHT: AtomicBool = AtomicBool::new(false);

thread_local! {
    static LANDER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Steps of this thread that met a landing (tests).
    #[cfg(test)]
    static MET_HERE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Counters since the last frame-rate line: landings on the thread, landings
/// on the submitting thread with the switch on, microseconds landing, steps
/// that met a landing and the microseconds they waited, microseconds
/// submissions waited for the landing before their own copy-back.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const LANDED: usize = 0;
const INLINE: usize = 1;
const LAND_US: usize = 2;
const MET: usize = 3;
const MET_US: usize = 4;
const SERIAL_US: usize = 5;

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Test hook: a landing starts this many milliseconds late.
#[cfg(test)]
pub(crate) static DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// Lands `work` (a copy-back writing `set`) on the thread. The caller holds
/// the submission lock and waited for the last landing ([`serial`]).
pub(crate) fn post(set: Set, work: Box<dyn FnOnce() + Send>) {
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let spawned = std::thread::Builder::new().name("victory-landing".into()).spawn(run);
        if let Err(e) = spawned {
            crate::host::fail(format_args!("no thread for copy-backs ({e})"));
        }
    });
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(set));
    IN_FLIGHT.store(true, Ordering::Release);
    STATS[LANDED].fetch_add(1, Ordering::Relaxed);
    let mut s = state();
    s.jobs.push_back(Job::Land(work));
    s.busy = true;
    CHANGED.notify_all();
}

fn run() {
    LANDER.with(|l| l.set(true));
    loop {
        let job = {
            let mut s = state();
            loop {
                if let Some(job) = s.jobs.pop_front() {
                    s.busy = true;
                    break job;
                }
                s.busy = false;
                CHANGED.notify_all();
                s = CHANGED.wait(s).unwrap_or_else(|e| e.into_inner());
            }
        };
        match job {
            Job::Land(work) => {
                #[cfg(test)]
                std::thread::sleep(std::time::Duration::from_millis(DELAY_MS.load(Ordering::Relaxed)));
                let started = std::time::Instant::now();
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err() {
                    tracing::error!("a copy-back on the landing thread panicked");
                }
                STATS[LAND_US].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = None;
                IN_FLIGHT.store(false, Ordering::Release);
            }
            Job::Then(work) => {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err() {
                    tracing::error!("work behind a copy-back panicked");
                }
            }
        }
    }
}

/// Runs `work` once the landing in flight and the work queued behind it are
/// done: now if there is none (the fence Signals behind a submission).
pub(crate) fn after_landing(work: impl FnOnce() + Send + 'static) {
    let mut s = state();
    if !s.busy && s.jobs.is_empty() {
        drop(s);
        work();
        return;
    }
    s.jobs.push_back(Job::Then(Box::new(work)));
    CHANGED.notify_all();
}

/// Waits until nothing is landing or queued behind a landing; how long.
pub(crate) fn wait() -> std::time::Duration {
    let started = std::time::Instant::now();
    if LANDER.with(std::cell::Cell::get) {
        return std::time::Duration::ZERO;
    }
    let mut s = state();
    while s.busy || !s.jobs.is_empty() {
        s = CHANGED.wait(s).unwrap_or_else(|e| e.into_inner());
    }
    started.elapsed()
}

/// Whether a landing is posted and not done.
#[cfg(test)]
pub(crate) fn in_flight() -> bool {
    IN_FLIGHT.load(Ordering::Acquire)
}

/// Before a submission's own copy-back: the last one has landed.
pub(crate) fn serial() {
    let waited = wait();
    STATS[SERIAL_US].fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
}

/// A step about to touch host copy `copy` (if any) and title memory
/// `[from, to)`: waits for the landing in flight if it writes either.
pub(crate) fn touch(copy: Option<usize>, from: usize, to: usize) {
    if !IN_FLIGHT.load(Ordering::Acquire) || LANDER.with(std::cell::Cell::get) {
        return;
    }
    let set = CURRENT.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if set.is_some_and(|s| s.meets(copy, from, to)) {
        met();
    }
}

/// A step that may write any title memory (a title copy, the title's own
/// work): waits for the landing in flight.
pub(crate) fn touch_all() {
    if IN_FLIGHT.load(Ordering::Acquire) && !LANDER.with(std::cell::Cell::get) {
        met();
    }
}

fn met() {
    #[cfg(test)]
    MET_HERE.with(|m| m.set(m.get() + 1));
    let waited = wait();
    STATS[MET].fetch_add(1, Ordering::Relaxed);
    STATS[MET_US].fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
}

/// A copy-back landed on the submitting thread although the switch is on.
pub(crate) fn note_inline() {
    STATS[INLINE].fetch_add(1, Ordering::Relaxed);
}

/// The counters per frame, logged every 5 s while the switch is on (called
/// with the frame-rate counts, `frames` since the last call).
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    static SINCE: Mutex<(f64, Option<std::time::Instant>)> = Mutex::new((0.0, None));
    let frames = {
        let mut since = SINCE.lock().unwrap_or_else(|e| e.into_inner());
        since.0 += frames;
        let started = *since.1.get_or_insert_with(std::time::Instant::now);
        if started.elapsed() < std::time::Duration::from_secs(5) {
            return;
        }
        let frames = since.0;
        *since = (0.0, Some(std::time::Instant::now()));
        frames
    };
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0)).collect();
    tracing::info!(
        "landing per frame: {:.1} copy-backs landed on their thread, {:.1} on the submitting thread; \
         {:.1} ms landing; next submission met a landing {:.1} times, waited {:.1} ms; \
         waited {:.1} ms for the last landing before its own copy-back",
        s[LANDED],
        s[INLINE],
        s[LAND_US] / 1e3,
        s[MET],
        s[MET_US] / 1e3,
        s[SERIAL_US] / 1e3,
    );
}

/// Steps of this thread that met a landing so far (tests; other tests'
/// submissions count on their threads).
#[cfg(test)]
pub(crate) fn met_count() -> u64 {
    MET_HERE.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_meets_its_copies_and_pages_and_their_aliases_only() {
        let a = Set { copies: [7usize].into_iter().collect(), ranges: Set::joined(vec![3, 4, 9]) };
        assert_eq!(a.ranges, vec![(3 * PAGE, 5 * PAGE), (9 * PAGE, 10 * PAGE)]);
        assert!(a.meets(Some(7), 0, 0), "its copy, anywhere");
        assert!(!a.meets(Some(8), 0, PAGE), "another copy, other pages");
        assert!(a.meets(None, 5 * PAGE - 1, 5 * PAGE), "the last byte of a range");
        assert!(!a.meets(None, 5 * PAGE, 9 * PAGE), "between the ranges");
        assert!(a.meets(None, 8 * PAGE, 9 * PAGE + 1));
        assert!(!a.meets(None, 10 * PAGE, 11 * PAGE));
        assert!(Set::default().is_empty());
    }

    /// The landing tests one at a time (they delay the landing thread).
    static ONE: Mutex<()> = Mutex::new(());

    /// A tracked title buffer of `pages` pages and its host copy, taken.
    fn buffer(pages: usize) -> Option<(usize, Arc<crate::shadow::Shadow>)> {
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return None; // no host GPU here
        };
        copy.update(p, len);
        Some((p, copy))
    }

    fn gone(p: usize) {
        crate::shadow::unregister(p, p as u64);
    }

    fn title(at: usize) -> u8 {
        // SAFETY: the test's title memory.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// A submission N whose shaders write byte 5 of a declared range (a
    /// host step after the snapshot stands for the GPU), landing late.
    fn submission(host: &crate::host::Host, p: usize, copy: &Arc<crate::shadow::Shadow>) -> crate::host::HostList {
        let list = host.new_list().unwrap();
        copy.gpu_writes(&list, p as u64, 256, false);
        let c = copy.clone();
        list.defer_host(move || c.write_host((p + 5) as u64, &[7]));
        list.close();
        list
    }

    /// (a) and (d): with the copy-back landing late, `execute` returns
    /// before title memory holds the GPU's byte, and the work behind it
    /// (the queue's fence Signal) runs only once it does; a title write to
    /// the same page while it lands stays, and the copy takes it.
    #[test]
    fn the_fence_follows_the_landing_and_title_writes_meanwhile_stay() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else { return };
        let Some((p, copy)) = buffer(1) else { return };
        let list = submission(host, p, &copy);
        DELAY_MS.store(300, Ordering::Relaxed);
        host.execute_with(&[&list], true);
        let seen = Arc::new(AtomicU64::new(u64::MAX));
        let s = seen.clone();
        after_landing(move || s.store(u64::from(title(p + 5)), Ordering::SeqCst));
        assert_eq!(title(p + 5), 0, "not landed yet");
        assert_eq!(seen.load(Ordering::SeqCst), u64::MAX, "the fence waits for the landing");
        // The title writes the page while the copy-back lands.
        // SAFETY: the test's title memory (a guard's fault opens it).
        unsafe { ((p + 100) as *mut u8).write_volatile(0x55) };
        wait();
        DELAY_MS.store(0, Ordering::Relaxed);
        assert_eq!(seen.load(Ordering::SeqCst), 7, "title memory held the GPU's byte when the fence was set");
        assert_eq!((title(p + 5), title(p + 100)), (7, 0x55), "the GPU's byte and the title's");
        let word = copy.read_host((p + 100) as u64, 1)[0];
        assert_eq!(word & 0xff, 0x55, "the copy took the title's byte");
        gone(p);
    }

    /// (b) and (c): the next submission's step that reads the copy the
    /// landing writes waits for it; a step reading another copy does not.
    #[test]
    fn the_next_submission_waits_only_where_the_landing_writes() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else { return };
        let Some((p, copy)) = buffer(1) else { return };
        let Some((q, other)) = buffer(1) else {
            gone(p);
            return;
        };
        // (c) Another copy: its refresh does not wait for the landing (no
        // step of this thread meets it; other tests' submissions may hold
        // the submission lock meanwhile, so the time is not asserted).
        let list = submission(host, p, &copy);
        DELAY_MS.store(400, Ordering::Relaxed);
        host.execute_with(&[&list], true);
        let met = met_count();
        let next = host.new_list().unwrap();
        other.refresh(&next, q as u64, 256);
        let flying = Arc::new(AtomicBool::new(false));
        next.close();
        host.execute_with(&[&next], true);
        assert_eq!(met_count(), met, "not held back by a landing it does not touch");
        wait();
        assert_eq!(title(p + 5), 7);
        // (b) The same copy: its refresh waits for the landing, and the
        // step after it runs once it has landed. Other tests' submissions
        // may take the submission lock first and wait for the landing
        // themselves (then this one finds it landed): up to 5 tries.
        let mut met_once = false;
        for _ in 0..5 {
            // SAFETY: the test's title memory.
            unsafe { ((p + 5) as *mut u8).write_volatile(0) };
            copy.update(p, PAGE);
            let list = submission(host, p, &copy);
            host.execute_with(&[&list], true);
            let met = met_count();
            let next = host.new_list().unwrap();
            copy.refresh(&next, p as u64, 256);
            let f = flying.clone();
            next.defer_host(move || f.store(in_flight(), Ordering::SeqCst));
            next.close();
            host.execute_with(&[&next], true);
            wait();
            assert!(!flying.load(Ordering::SeqCst), "the step after the refresh ran once it had landed");
            assert_eq!(title(p + 5), 7);
            if met_count() > met {
                met_once = true;
                break;
            }
        }
        DELAY_MS.store(0, Ordering::Relaxed);
        assert!(met_once, "the refresh met the landing");
        gone(p);
        gone(q);
    }
}
