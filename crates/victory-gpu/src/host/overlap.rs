//! Heap syncs while a submission runs on the GPU (`VICTORY_SYNC_OVERLAP`).
//!
//! The sync before each submission (`crate::descriptor::sync_heaps`) reads
//! only records the title wrote into its heaps from the CPU: it needs
//! neither the running submission's results nor their copy-back, and title
//! threads already write host slots while submissions run (their view
//! calls, draw tables). What the title wrote by the time a submission is on
//! the GPU can so be made while we wait for it, leaving the next sync the
//! pages written since. 0 (the default) syncs before submissions only; 1
//! syncs once more right before the wait; 2 syncs over and over on a thread
//! of ours until the submission has run.
//!
//! The thread takes no lock the waiting thread holds (`submitted`, the held
//! work's): the sync ran under both on the submitting thread all along, so
//! it never takes them. It stops before the wait returns to its caller, so
//! nothing after the wait (copy-back, views made for slots shaders read)
//! runs beside it.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, OnceLock};

/// `VICTORY_SYNC_OVERLAP`: 0, 1 or 2 (above).
pub(crate) fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = std::env::var("VICTORY_SYNC_OVERLAP")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(0)
            .min(2);
        if mode > 0 {
            tracing::info!(mode, "heap syncs also run while submissions run (VICTORY_SYNC_OVERLAP)");
        }
        mode
    })
}

/// Heap syncs run while submissions ran and their microseconds, since the
/// last frame-rate line.
pub(super) static OVERLAP: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// How long the thread rests between syncs (a stop wakes it at once):
/// each sync holds the heaps' lock, which title threads writing views wait on.
const REST: std::time::Duration = std::time::Duration::from_millis(1);

#[derive(Default)]
struct State {
    /// The sync to run while a submission runs.
    job: Option<fn()>,
    /// Whether a sync is running.
    busy: bool,
}

/// A thread running a sync over and over while asked to.
#[derive(Default)]
pub(super) struct Worker {
    state: Mutex<State>,
    changed: Condvar,
}

impl Worker {
    /// A worker with its thread.
    pub(super) fn start() -> &'static Worker {
        let worker: &'static Worker = Box::leak(Box::default());
        let spawned = std::thread::Builder::new()
            .name("victory heap sync".into())
            .spawn(move || worker.run());
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no thread for heap syncs while submissions run");
        }
        worker
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn run(&self) {
        let mut state = self.lock();
        loop {
            let Some(sync) = state.job else {
                state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            state.busy = true;
            drop(state);
            timed(sync);
            state = self.lock();
            state.busy = false;
            self.changed.notify_all();
            if state.job.is_some() {
                state = self.changed.wait_timeout(state, REST).unwrap_or_else(|e| e.into_inner()).0;
            }
        }
    }

    /// Runs `wait` while the thread runs `sync` over and over; returns
    /// once no sync runs any more.
    pub(super) fn during<R>(&self, sync: fn(), wait: impl FnOnce() -> R) -> R {
        {
            let mut state = self.lock();
            state.job = Some(sync);
            self.changed.notify_all();
        }
        let waited = wait();
        let mut state = self.lock();
        state.job = None;
        self.changed.notify_all();
        while state.busy {
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        waited
    }
}

/// Runs `sync` once, counted (a panic in it is logged, not passed on: the
/// thread must stay to let the waiting thread go).
fn timed(sync: fn()) {
    let started = std::time::Instant::now();
    if std::panic::catch_unwind(sync).is_err() {
        tracing::warn!("a heap sync while a submission ran panicked");
    }
    let us = started.elapsed().as_micros() as u64;
    OVERLAP[0].fetch_add(1, Relaxed);
    OVERLAP[1].fetch_add(us, Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::OverlapPasses, 1);
        crate::copystat::add(crate::copystat::C::OverlapUs, us);
    }
}

/// Runs `wait` (for a submission on the GPU) with heap syncs beside it as
/// `mode` says.
pub(crate) fn while_waiting<R>(mode: u8, sync: fn(), wait: impl FnOnce() -> R) -> R {
    match mode {
        1 => {
            timed(sync);
            wait()
        }
        2 => {
            static WORKER: OnceLock<&'static Worker> = OnceLock::new();
            WORKER.get_or_init(Worker::start).during(sync, wait)
        }
        _ => wait(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    static RUNNING: AtomicUsize = AtomicUsize::new(0);
    static RAN: AtomicUsize = AtomicUsize::new(0);
    static SEEN: AtomicBool = AtomicBool::new(false);
    static WRITTEN: AtomicBool = AtomicBool::new(false);

    fn slow() {
        RUNNING.fetch_add(1, Relaxed);
        SEEN.store(WRITTEN.load(Relaxed), Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(3));
        RAN.fetch_add(1, Relaxed);
        RUNNING.fetch_sub(1, Relaxed);
    }

    /// The thread syncs while the wait lasts, sees what was written during
    /// it, and no sync runs once the wait has returned.
    #[test]
    fn syncs_run_during_the_wait_and_stop_before_it_returns() {
        let worker = Worker::start();
        let ran = worker.during(slow, || {
            WRITTEN.store(true, Relaxed);
            let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !SEEN.load(Relaxed) && std::time::Instant::now() < until {
                std::thread::yield_now();
            }
            SEEN.load(Relaxed)
        });
        assert!(ran, "a write during the wait was seen by a sync during it");
        assert_eq!(RUNNING.load(Relaxed), 0, "no sync runs after the wait");
        let after = RAN.load(Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(RAN.load(Relaxed), after, "the thread rests between waits");
    }

    fn boom() {
        panic!("test");
    }

    /// A panicking sync does not keep the waiting thread.
    #[test]
    fn a_panicking_sync_lets_the_wait_return() {
        let worker = Worker::start();
        assert_eq!(worker.during(boom, || 7), 7);
        assert_eq!(worker.during(boom, || 8), 8);
    }
}
