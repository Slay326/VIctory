//! Access faults on title pages the GPU owns: a read measurement and the
//! safety net of the memory model.
//!
//! The memory model wants pages only the GPU uses to stay on the card,
//! with no mirror and no copy-back. Whether the title's CPU code ever
//! touches such a page cannot be proven ahead of time, so the page is made
//! inaccessible instead and the fault of the first access decides:
//!
//! - [`Mode::Measure`] changes nothing about what is synced. A page is
//!   armed after its copy-back, the first access is counted per 4 KiB
//!   block and the page gets back the protection it had. The counts say
//!   which pages the title never reads.
//! - [`Mode::Net`] arms the pages treated as GPU-only. The first access
//!   waits for the GPU work submitted so far, copies the GPU's bytes into
//!   title memory, demotes the page to shared for good and only then lets
//!   the thread run on. A wrong classification costs a sync, never a byte.
//!
//! The unit is the 64 KiB page of [`crate::watch`]. Everything the module
//! needs from the system and the engine is behind [`Protect`], so the state
//! machine runs in tests without either.
//!
//! Invariants (tests in `pagenet_tests.rs`):
//! - I2: a faulting thread runs on only after the GPU writes submitted so
//!   far are in title memory (fence, copy, demote, open, in that order).
//! - I3: a demoted page is never armed or GPU-only again.
//! - I5: several threads faulting on one page cause one sync, and all of
//!   them run on only after it.
//! - I6: a fault on a page that is not armed changes nothing.
//! - I7: measuring changes no byte and no sync decision.
//! - I8: an access by a host thread is never counted as the title's.
//! - I9: the addresses showing one physical page are GPU-only together or
//!   not at all, and leave the class together, by one sync: a second sync
//!   for another address would put the GPU's bytes over what the CPU wrote
//!   through the first.
//!
//! Locks: `arm`, `arm_group`, `join`, `host_access`, `disarm_for_free` and
//! the measuring branch of a fault change protections and are called with
//! the backend's exclusion held by the caller (on real memory the write
//! guard's lock). A sync is not: it waits for the GPU and copies, and takes
//! the exclusion itself for its last step ([`Protect::guarded`]), always
//! before the net's own lock.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

/// The tracked and protected unit (as `crate::watch::PAGE`).
pub const PAGE: usize = 64 << 10;
/// The counted unit inside a page: false sharing shows here.
pub const BLOCK: usize = 4 << 10;

/// What the module does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Nothing: no page is armed, no fault is taken.
    Off,
    /// Count the first access after each copy-back.
    Measure,
    /// Sync and demote a GPU-only page on its first access.
    Net,
}

/// The protection of a page, as far as the module tells them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prot {
    /// Open.
    ReadWrite,
    /// The write guard of `crate::watch`: taken, not written since.
    ReadOnly,
    /// Armed, or not mapped.
    NoAccess,
}

/// Why a page is armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Measure: right after the page's copy-back.
    AfterCopyBack,
    /// Net: the page is treated as GPU-only.
    GpuOnly,
}

/// Whose fault it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Ours: the access can be retried.
    Handled,
    /// Not ours: the next handler decides.
    NotOurs,
}

/// Where a page stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Not armed (never, or opened again by a measured access).
    Shared,
    /// Armed to count the next access.
    ArmedMeasure,
    /// Armed as GPU-only.
    GpuOnlyArmed,
    /// A faulting thread is bringing the GPU's bytes into title memory.
    Syncing,
    /// Was GPU-only and got touched: shared for good.
    Demoted,
}

/// What touched a GPU-only page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoteReason {
    /// A read by the title's CPU code (or the host's, in net mode).
    CpuRead,
    /// A write.
    CpuWrite,
    /// The engine itself needed the page's bytes in title memory, or was
    /// about to write it there: another copy or a texture reading it, a
    /// copy step or a kernel write into it.
    Host,
    /// One of the page's addresses was unmapped, or mapped anew in a way
    /// that could not be closed.
    Remap,
    /// What shaders wrote there was to be dropped (withheld writes), which
    /// a page with no bytes in title memory cannot do: the GPU's bytes
    /// were kept instead.
    Withheld,
}

/// What [`PageNet::arm_group`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Armed {
    /// The pages are GPU-only from now on.
    Now,
    /// They were already.
    Already,
    /// Not armed: a page of them is demoted, or cannot be closed.
    Refused,
}

/// One page leaving the GPU-only class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Demotion {
    /// The page number.
    pub page: usize,
    /// The access that did it.
    pub reason: DemoteReason,
    /// The frame it happened in.
    pub frame: u64,
    /// How many addresses showed the page's memory and left with it.
    pub pages: usize,
}

/// What the faults showed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Title reads that hit an armed page, by (page number, 4 KiB block).
    pub reads: BTreeMap<(usize, usize), u64>,
    /// Title writes that hit an armed page, by (page number, 4 KiB block).
    pub writes: BTreeMap<(usize, usize), u64>,
    /// `arm` calls ignored because the page is demoted (I3).
    pub arm_noops: u64,
    /// Syncs done by the net (fence and forced copy-back).
    pub syncs: u64,
}

/// What did not go as the plain path expects; not part of [`Stats`] so a
/// clean run compares equal to an idle one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Oddities {
    /// Accesses of the host to armed pages (opened, not counted, I8).
    pub host_faults: u64,
    /// Armed pages whose protection someone else replaced before the
    /// fault or the opening: the arming was lost, nothing was counted.
    pub overridden: u64,
    /// `arm` calls that found the page's protection changing under them.
    pub arm_lost: u64,
}

/// What the module needs from the system and the engine.
pub trait Protect: Send + Sync + 'static {
    /// The protection of page `page` (a page number); `NoAccess` when it is
    /// not mapped.
    fn get(&self, page: usize) -> Prot;
    /// Sets `page` to `to` if and only if it is `from` now, as one step
    /// against everyone else who protects title pages. True if it did.
    ///
    /// The one-step rule is what keeps the write guard intact: a page is
    /// only ever given back the protection it was armed from, and only
    /// while nobody else has protected it since.
    fn swap(&self, page: usize, from: Prot, to: Prot) -> bool;
    /// Returns once every GPU write submitted so far has completed (I2).
    fn fence_wait(&self);
    /// The GPU's bytes of `page` into title memory, without the "written
    /// since taken" check of the ordinary copy-back: an armed page cannot
    /// have been written by the title. Must not touch the title's own view
    /// of the page, which stays closed until the copy is done.
    ///
    /// Called once for a group of addresses showing the same memory, with
    /// the page the group was armed by.
    fn copy_back_forced(&self, page: usize);
    /// The current frame, for demotion records.
    fn frame(&self) -> u64;
    /// Runs `f`, which changes protections ([`Protect::swap`]), as one step
    /// against everyone else who protects title pages. Called by a sync
    /// with no lock of the net held; `f` takes the net's.
    fn guarded<R>(&self, f: impl FnOnce() -> R) -> R
    where
        Self: Sized,
    {
        f()
    }
}

/// One page the module has armed at some point.
#[derive(Clone, Copy)]
struct Entry {
    state: State,
    /// The protection the page was armed from, given back on opening.
    from: Prot,
    /// The page its group was armed by (itself, alone): the addresses
    /// showing the same memory are GPU-only together (I9).
    group: usize,
}

#[derive(Default)]
struct Inner {
    pages: BTreeMap<usize, Entry>,
    stats: Stats,
    oddities: Oddities,
    demotions: Vec<Demotion>,
    host: BTreeSet<u32>,
    /// How often each page was armed.
    arms: BTreeMap<usize, u64>,
    /// The GPU-only groups of more than one page, by the page they were
    /// armed by: every page of the group, that one first.
    groups: BTreeMap<usize, Vec<usize>>,
}

/// The fault bookkeeping of title pages (see the module).
pub struct PageNet<B: Protect> {
    mode: Mode,
    backend: B,
    inner: Mutex<Inner>,
    /// Signalled when a sync ends.
    synced: Condvar,
    /// Pages armed now: lets the paths that only matter for armed pages
    /// leave without the lock.
    armed: AtomicUsize,
}

impl<B: Protect> PageNet<B> {
    /// A net in `mode` over `backend`, with nothing armed.
    pub fn new(mode: Mode, backend: B) -> Self {
        Self {
            mode,
            backend,
            inner: Mutex::new(Inner::default()),
            synced: Condvar::new(),
            armed: AtomicUsize::new(0),
        }
    }

    /// The mode it was made with.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The backend it was made with.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    // A panic while holding the lock leaves the maps consistent (every
    // change is a single insert or counter), and a fault handler has no
    // way to report a poisoned lock but to crash the title.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Arms `page`: the next access faults into [`Self::on_fault`].
    ///
    /// Nothing happens when the mode is off, when the reason is not the
    /// mode's (measure arms after copy-backs, the net arms GPU-only pages),
    /// when the page is armed or syncing already, or when it is demoted
    /// (counted, I3).
    pub fn arm(&self, page: usize, reason: Reason) {
        let state = match (self.mode, reason) {
            (Mode::Measure, Reason::AfterCopyBack) => State::ArmedMeasure,
            (Mode::Net, Reason::GpuOnly) => {
                self.arm_group(&[page]);
                return;
            }
            _ => return,
        };
        let mut inner = self.lock();
        match inner.pages.get(&page).map(|e| e.state) {
            Some(State::Demoted) => {
                inner.stats.arm_noops += 1;
                return;
            }
            // Someone else's protection replaced ours since: the arming
            // is gone, and the page is armed afresh.
            Some(State::ArmedMeasure) if self.backend.get(page) != Prot::NoAccess => {
                inner.oddities.overridden += 1;
                self.armed.fetch_sub(1, Ordering::Relaxed);
            }
            Some(State::ArmedMeasure | State::GpuOnlyArmed | State::Syncing) => return,
            Some(State::Shared) | None => {}
        }
        let from = self.backend.get(page);
        if from == Prot::NoAccess {
            // Not mapped, or closed by someone else: not ours to open later.
            return;
        }
        if !self.backend.swap(page, from, Prot::NoAccess) {
            inner.oddities.arm_lost += 1;
            return;
        }
        inner.pages.insert(
            page,
            Entry {
                state,
                from,
                group: page,
            },
        );
        *inner.arms.entry(page).or_default() += 1;
        self.armed.fetch_add(1, Ordering::Relaxed);
    }

    /// Makes `pages` GPU-only together (the net's mode only): the addresses
    /// showing one physical page, the one whose copy holds the GPU's bytes
    /// first. All of them are closed, or none (I9): a page that is demoted
    /// (counted, I3), not mapped, closed by someone else or whose
    /// protection changes under us refuses the whole group, and the pages
    /// closed so far get their protection back.
    pub fn arm_group(&self, pages: &[usize]) -> Armed {
        if self.mode != Mode::Net || pages.is_empty() {
            return Armed::Refused;
        }
        let mut inner = self.lock();
        let states: Vec<Option<State>> = pages
            .iter()
            .map(|p| inner.pages.get(p).map(|e| e.state))
            .collect();
        if states.contains(&Some(State::Demoted)) {
            inner.stats.arm_noops += 1;
            return Armed::Refused;
        }
        let closed = |s: &Option<State>| matches!(s, Some(State::GpuOnlyArmed | State::Syncing));
        if states.iter().all(closed) {
            return Armed::Already;
        }
        if states.iter().any(closed) {
            // Part of another group, or of this one with pages missing:
            // not widened here (`join` does that, one page at a time).
            return Armed::Refused;
        }
        let mut done: Vec<(usize, Prot)> = Vec::with_capacity(pages.len());
        for &page in pages {
            let from = self.backend.get(page);
            if from == Prot::NoAccess
                || done.iter().any(|&(p, _)| p == page)
                || !self.backend.swap(page, from, Prot::NoAccess)
            {
                for &(page, from) in &done {
                    if !self.backend.swap(page, Prot::NoAccess, from) {
                        inner.oddities.overridden += 1;
                    }
                }
                inner.oddities.arm_lost += 1;
                return Armed::Refused;
            }
            done.push((page, from));
        }
        for &(page, from) in &done {
            inner.pages.insert(
                page,
                Entry {
                    state: State::GpuOnlyArmed,
                    from,
                    group: pages[0],
                },
            );
            *inner.arms.entry(page).or_default() += 1;
        }
        if pages.len() > 1 {
            inner.groups.insert(pages[0], pages.to_vec());
        }
        self.armed.fetch_add(pages.len(), Ordering::Relaxed);
        Armed::Now
    }

    /// `page` shows the memory of `of` from now on (a new mapping): if `of`
    /// is GPU-only, `page` is closed with it and leaves with it. False if
    /// `of` is GPU-only and `page` could not be closed: the caller must
    /// take the group out of the class ([`Self::demote`]) before the new
    /// address is used.
    pub fn join(&self, of: usize, page: usize) -> bool {
        if self.mode != Mode::Net {
            return true;
        }
        let mut inner = self.lock();
        let Some(entry) = inner.pages.get(&of).copied() else {
            return true;
        };
        if !matches!(entry.state, State::GpuOnlyArmed | State::Syncing) {
            return true;
        }
        match inner.pages.get(&page).map(|e| e.state) {
            Some(State::GpuOnlyArmed | State::Syncing) => return true,
            Some(State::Demoted) => return false,
            _ => {}
        }
        let from = self.backend.get(page);
        if from == Prot::NoAccess || !self.backend.swap(page, from, Prot::NoAccess) {
            inner.oddities.arm_lost += 1;
            return false;
        }
        inner.pages.insert(
            page,
            Entry {
                state: entry.state,
                from,
                group: entry.group,
            },
        );
        *inner.arms.entry(page).or_default() += 1;
        let group = entry.group;
        inner
            .groups
            .entry(group)
            .or_insert_with(|| vec![group])
            .push(page);
        self.armed.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Takes `page` and the pages closed with it out of the GPU-only class
    /// for good, for a reason that is not an access fault: the GPU's bytes
    /// go into title memory first, as for a fault (I2). True if the page
    /// was GPU-only (another thread's sync of it is waited for).
    pub fn demote(&self, page: usize, reason: DemoteReason) -> bool {
        if self.mode != Mode::Net {
            return false;
        }
        let mut inner = self.lock();
        loop {
            match inner.pages.get(&page).map(|e| e.state) {
                Some(State::GpuOnlyArmed) => {
                    self.sync(inner, page, reason);
                    return true;
                }
                Some(State::Syncing) => {
                    inner = self.synced.wait(inner).unwrap_or_else(|e| e.into_inner());
                    if inner.pages.get(&page).map(|e| e.state) != Some(State::Syncing) {
                        return true;
                    }
                }
                _ => return false,
            }
        }
    }

    /// The sync of a GPU-only page and the pages closed with it: fence,
    /// one copy of the GPU's bytes into title memory, then all of them
    /// demoted and opened together (I2, I9).
    fn sync(&self, mut inner: MutexGuard<'_, Inner>, page: usize, reason: DemoteReason) {
        let group = inner.pages.get(&page).map_or(page, |e| e.group);
        let members = inner
            .groups
            .get(&group)
            .cloned()
            .unwrap_or_else(|| vec![page]);
        for member in &members {
            if let Some(entry) = inner.pages.get_mut(member) {
                entry.state = State::Syncing;
            }
        }
        // The sync runs without the lock: it waits for the GPU, and other
        // threads must be able to find the pages syncing meanwhile.
        drop(inner);
        self.backend.fence_wait();
        self.backend.copy_back_forced(group);
        let frame = self.backend.frame();
        self.backend.guarded(|| {
            let mut inner = self.lock();
            // Read again: an address may have joined meanwhile.
            let members = inner.groups.remove(&group).unwrap_or(members);
            for &member in &members {
                let Some(entry) = inner.pages.get_mut(&member) else {
                    continue;
                };
                entry.state = State::Demoted;
                let from = entry.from;
                if !self.backend.swap(member, Prot::NoAccess, from) {
                    inner.oddities.overridden += 1;
                }
            }
            // Counted as closed until here: a host access meanwhile must
            // find them and wait.
            self.armed.fetch_sub(members.len(), Ordering::Relaxed);
            inner.stats.syncs += 1;
            inner.demotions.push(Demotion {
                page,
                reason,
                frame,
                pages: members.len(),
            });
        });
        self.synced.notify_all();
    }

    /// A thread `tid` faulted reading (or writing, `is_write`) `addr`: the
    /// host's if `tid` was named by [`Self::host_thread`].
    pub fn on_fault(&self, addr: usize, is_write: bool, tid: u32) -> Fault {
        let host = self.lock().host.contains(&tid);
        self.on_fault_from(addr, is_write, host)
    }

    /// A thread faulted reading (or writing, `is_write`) `addr`, in the
    /// host's own code (`host`) or the title's.
    ///
    /// `Handled` means the access can be retried: the page was armed and is
    /// open now, or it was opened by another thread in the meantime.
    pub fn on_fault_from(&self, addr: usize, is_write: bool, host: bool) -> Fault {
        if self.mode == Mode::Off {
            return Fault::NotOurs;
        }
        let page = addr / PAGE;
        let block = (addr % PAGE) / BLOCK;
        let mut inner = self.lock();
        let Some(entry) = inner.pages.get(&page).copied() else {
            return Fault::NotOurs;
        };
        match entry.state {
            State::ArmedMeasure => {
                self.armed.fetch_sub(1, Ordering::Relaxed);
                inner.pages.insert(
                    page,
                    Entry {
                        state: State::Shared,
                        ..entry
                    },
                );
                if !self.backend.swap(page, Prot::NoAccess, entry.from) {
                    // Someone protected the page after us: their
                    // protection stands, and so does their fault.
                    inner.oddities.overridden += 1;
                    return self.late(page, is_write);
                }
                if host {
                    inner.oddities.host_faults += 1;
                } else {
                    let counts = if is_write {
                        &mut inner.stats.writes
                    } else {
                        &mut inner.stats.reads
                    };
                    *counts.entry((page, block)).or_default() += 1;
                }
                Fault::Handled
            }
            State::GpuOnlyArmed => {
                if host {
                    inner.oddities.host_faults += 1;
                } else {
                    let counts = if is_write {
                        &mut inner.stats.writes
                    } else {
                        &mut inner.stats.reads
                    };
                    *counts.entry((page, block)).or_default() += 1;
                }
                let reason = match (host, is_write) {
                    (true, _) => DemoteReason::Host,
                    (false, true) => DemoteReason::CpuWrite,
                    (false, false) => DemoteReason::CpuRead,
                };
                self.sync(inner, page, reason);
                Fault::Handled
            }
            State::Syncing => {
                // I5: wait for the other thread's sync, then retry the
                // access: the page is open as far as the net goes (a
                // write may fault once more, into the write guard).
                while inner
                    .pages
                    .get(&page)
                    .is_some_and(|e| e.state == State::Syncing)
                {
                    inner = self.synced.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
                Fault::Handled
            }
            State::Shared | State::Demoted => self.late(page, is_write),
        }
    }

    /// A fault on a page that was armed once and is not now: a thread that
    /// faulted before another one opened the page. Retried if the page
    /// allows the access by now; otherwise it is someone else's fault (a
    /// write to the write guard, a real violation).
    fn late(&self, page: usize, is_write: bool) -> Fault {
        match (self.backend.get(page), is_write) {
            (Prot::ReadWrite, _) | (Prot::ReadOnly, false) => Fault::Handled,
            _ => Fault::NotOurs,
        }
    }

    /// The host is about to read or write `page` itself, by a path that
    /// fails on a closed page instead of faulting (a copy in the kernel):
    /// a page armed for measuring is opened first, not counted (I8). True
    /// if it was armed.
    ///
    /// A page armed as GPU-only is left alone: there the host's access has
    /// to sync like the title's, which is the fault path's business.
    pub fn host_access(&self, page: usize) -> bool {
        if !self.any_armed() {
            return false;
        }
        let mut inner = self.lock();
        let Some(entry) = inner.pages.get(&page).copied() else {
            return false;
        };
        if entry.state != State::ArmedMeasure {
            return false;
        }
        self.armed.fetch_sub(1, Ordering::Relaxed);
        inner.pages.insert(
            page,
            Entry {
                state: State::Shared,
                ..entry
            },
        );
        if self.backend.swap(page, Prot::NoAccess, entry.from) {
            inner.oddities.host_faults += 1;
        } else {
            inner.oddities.overridden += 1;
        }
        true
    }

    /// Whether any page is armed now (no lock taken).
    pub fn any_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed) != 0
    }

    /// How often each page was armed so far.
    pub fn armed_counts(&self) -> BTreeMap<usize, u64> {
        self.lock().arms.clone()
    }

    /// Whether `page` is treated as GPU-only right now.
    pub fn is_gpu_only(&self, page: usize) -> bool {
        self.state(page) == State::GpuOnlyArmed
    }

    /// The pages of `[first, last)` that are closed as GPU-only now (or
    /// being synced): title memory there is not what the GPU holds.
    pub fn closed_in(&self, first: usize, last: usize) -> Vec<usize> {
        if !self.any_armed() {
            return Vec::new();
        }
        self.lock()
            .pages
            .range(first..last)
            .filter(|(_, e)| matches!(e.state, State::GpuOnlyArmed | State::Syncing))
            .map(|(&p, _)| p)
            .collect()
    }

    /// The page `page`'s group was armed by, while it is closed as
    /// GPU-only (or being synced).
    pub fn group_of(&self, page: usize) -> Option<usize> {
        self.lock()
            .pages
            .get(&page)
            .filter(|e| matches!(e.state, State::GpuOnlyArmed | State::Syncing))
            .map(|e| e.group)
    }

    /// Where `page` stands.
    pub fn state(&self, page: usize) -> State {
        self.lock()
            .pages
            .get(&page)
            .map_or(State::Shared, |e| e.state)
    }

    /// `page` is about to be freed or unmapped: opens it if it is armed
    /// (after a sync in progress) and forgets it. A demoted page stays
    /// demoted, also for whatever is mapped there next (I3).
    pub fn disarm_for_free(&self, page: usize) {
        let mut inner = self.lock();
        loop {
            let Some(entry) = inner.pages.get(&page).copied() else {
                return;
            };
            match entry.state {
                State::Syncing => {
                    inner = self.synced.wait(inner).unwrap_or_else(|e| e.into_inner())
                }
                State::Demoted => return,
                State::ArmedMeasure | State::GpuOnlyArmed => {
                    self.armed.fetch_sub(1, Ordering::Relaxed);
                    if !self.backend.swap(page, Prot::NoAccess, entry.from) {
                        inner.oddities.overridden += 1;
                    }
                    inner.pages.remove(&page);
                    return;
                }
                State::Shared => {
                    inner.pages.remove(&page);
                    return;
                }
            }
        }
    }

    /// The counts so far.
    pub fn stats(&self) -> Stats {
        self.lock().stats.clone()
    }

    /// What did not go as the plain path expects.
    pub fn oddities(&self) -> Oddities {
        self.lock().oddities
    }

    /// The demotions not taken yet ([`Self::drain_demotions`]), in order.
    pub fn demotions(&self) -> Vec<Demotion> {
        self.lock().demotions.clone()
    }

    /// The demotions since the last call, in order.
    pub fn drain_demotions(&self) -> Vec<Demotion> {
        std::mem::take(&mut self.lock().demotions)
    }

    /// Thread `tid` is the host's: its faults open the page like any other
    /// (it must run on) but never count as title accesses (I8).
    pub fn host_thread(&self, tid: u32) {
        self.lock().host.insert(tid);
    }
}
