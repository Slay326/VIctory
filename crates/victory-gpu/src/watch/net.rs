//! [`crate::pagenet`] on real title memory: which pages the title's CPU
//! code touches after the GPU wrote them (`VICTORY_PAGENET=measure`; off
//! unless asked for), and the memory model's safety net
//! (`VICTORY_MEMMODEL=1`; off unless asked for), further down.
//!
//! Measuring:
//!
//! A page is armed (no access) when a copy-back has put the GPU's bytes
//! into it and the copy has taken it again, so only a page under the write
//! guard is ever armed, and opening it gives the guard's read-only back.
//! Every way back arms ([`Way`]): pages written at computed addresses,
//! whole pages of declared ranges, and marked pages, whether written at the
//! title's address or through our own view of the section (the title's
//! guard never opened; our view is another address and stays open).
//! The first access faults into the write guard's handler, which asks here
//! first ([`fault`]): the access is counted per 4 KiB block and the page is
//! as it was before arming, so a write faults once more, into the guard.
//!
//! Measuring must not change what is synced:
//! - Every protection change here is made holding the guard lock, as the
//!   write guard's own are, so neither undoes the other's. A thread that
//!   holds the lock already (the guard's code reading a page it is about to
//!   open) is let through without it: nobody else changes guards meanwhile.
//! - The engine's own copies run in the kernel, which fails on a closed
//!   page instead of faulting: they open their pages first
//!   ([`host_touch`]), uncounted.
//! - The write guard opening or guarding a page replaces our protection;
//!   the arming is lost and noted, nothing else.
//! - Pages showing the same memory elsewhere are armed together, or not at
//!   all: a page whose other addresses stay open would look unread.
//!
//! shortcut: a system call of the title's reading from an armed page fails
//! (as one writing into a guarded page does, see `crate::watch`); and the
//! title sees the closed page if it asks for the protection. Both are noted
//! nowhere: a measuring run that misbehaves is compared with one without.
//!
//! A fault is counted as the title's whoever's code took it: the runtime's
//! code reading a buffer for the title is a CPU read all the same, and code
//! of ours that should not have counted errs towards "shared", which costs
//! speed, never a byte. Where the faulting code is, in our module or not,
//! is reported beside.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use super::{GUARD_OWNER, Guards, PAGE, Tracked, guards, own, this_thread};
use crate::pagenet::{Armed, BLOCK, DemoteReason, Fault, Mode, PageNet, Prot, Protect, Reason, State};
use crate::shadow::Shadow;

/// Title memory's protections, for [`PageNet`]. Only called with the guard
/// lock held (by [`locked`], or by this thread further up).
pub(crate) struct Win;

/// The protection of the whole of page `page`, if it is committed and has
/// one protection throughout.
fn protection(page: usize) -> Option<u32> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
    (n != 0 && info.State == MEM_COMMIT && info.RegionSize >= PAGE).then_some(info.Protect)
}

impl Protect for Win {
    fn get(&self, page: usize) -> Prot {
        use windows_sys::Win32::System::Memory::{
            PAGE_EXECUTE_READ, PAGE_READONLY, PAGE_READWRITE,
        };
        match protection(page) {
            Some(PAGE_READWRITE) => Prot::ReadWrite,
            Some(PAGE_READONLY | PAGE_EXECUTE_READ) => Prot::ReadOnly,
            _ => Prot::NoAccess,
        }
    }

    /// Only between the write guard's read-only and no access: a page that
    /// is open is not armed (see the module).
    fn swap(&self, page: usize, from: Prot, to: Prot) -> bool {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, VirtualProtect};
        let (from, to) = match (from, to) {
            (Prot::ReadOnly, Prot::NoAccess) => (PAGE_READONLY, PAGE_NOACCESS),
            (Prot::NoAccess, Prot::ReadOnly) => (PAGE_NOACCESS, PAGE_READONLY),
            _ => return false,
        };
        if protection(page) != Some(from) {
            return false;
        }
        let mut old = 0;
        // SAFETY: title memory; only its protection changes.
        unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, to, &mut old) != 0 }
    }

    /// Nothing to wait for: a sync runs inside [`idle`], where no
    /// submission is under way and the last one's copy-backs are done.
    fn fence_wait(&self) {}

    /// By the copy that closed the page ([`close`]), through our own view
    /// of the page's section: the title's stays closed.
    fn copy_back_forced(&self, page: usize) {
        let owner = owners().get(&page).cloned();
        if !owner.is_some_and(|copy| copy.copy_back_forced(page * PAGE)) {
            // The page opens with title memory's older bytes: never
            // silent (`report`).
            FORCED_FAILED.fetch_add(1, Relaxed);
        }
    }

    fn frame(&self) -> u64 {
        FRAME.load(Relaxed)
    }

    /// Under the guard lock, taken here unless this thread holds it.
    fn guarded<R>(&self, f: impl FnOnce() -> R) -> R {
        locked(|_| f())
    }
}

/// Presented frames so far.
static FRAME: AtomicU64 = AtomicU64::new(0);

/// What is counted beside [`PageNet`]'s own counts.
#[derive(Default)]
struct Beside {
    /// Armed pages that show another armed page's memory: page to the page
    /// whose copy-back armed it.
    alias_of: BTreeMap<usize, usize>,
    /// Faults by the address of the faulting code (a module offset if it is
    /// in our module), the first [`CODE_PLACES`] places seen.
    code: BTreeMap<(bool, usize), u64>,
}

const CODE_PLACES: usize = 64;

static BESIDE: Mutex<Beside> = Mutex::new(Beside {
    alias_of: BTreeMap::new(),
    code: BTreeMap::new(),
});
/// Copy-backs whose page was not armed: not under the guard, or a page
/// showing its memory elsewhere is not.
static NOT_ARMED: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones whose own page was under the guard (a page showing
/// its memory elsewhere was not).
static NOT_ARMED_ELSEWHERE: AtomicU64 = AtomicU64::new(0);

/// The way a page's bytes went back (`crate::shadow`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Way {
    /// A page shaders wrote at an address they computed.
    Computed,
    /// A whole page of a range a list declared.
    Declared,
    /// A page marked through views.
    Marked,
}

/// Copy-backs that armed their page, and those that did not, by [`Way`].
static BY_WAY: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];
/// Faults counted in our module's code, and in any other.
static IN_OURS: AtomicU64 = AtomicU64::new(0);
static ELSEWHERE: AtomicU64 = AtomicU64::new(0);

/// What `VICTORY_PAGENET` (its value) and `VICTORY_MEMMODEL` (on or not)
/// ask for: measuring, or the memory model with its net, which wins where
/// both are set (a page is closed for one reason at a time).
fn mode(pagenet: Option<&str>, memmodel: bool) -> Option<Mode> {
    match pagenet {
        _ if memmodel => Some(Mode::Net),
        Some("measure") => Some(Mode::Measure),
        _ => None,
    }
}

#[cfg(test)]
thread_local! {
    /// Tests switch measuring on for their own thread: the crate's other
    /// tests run as without it.
    pub(crate) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests switch the memory model on for their own thread (it wins over
    /// measuring, as in `mode`).
    pub(crate) static TEST_MODEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static NET: OnceLock<Option<PageNet<Win>>> = OnceLock::new();
/// The tests' net in the model's mode, beside the measuring one.
#[cfg(test)]
static MODEL: OnceLock<PageNet<Win>> = OnceLock::new();

/// The net, if it was made: the fault handler's way to it, which must not
/// read the environment or log.
fn net_made() -> Option<&'static PageNet<Win>> {
    #[cfg(test)]
    {
        if TEST_MODEL.get() {
            return MODEL.get();
        }
        if !TEST_ON.get() {
            return None;
        }
    }
    NET.get().and_then(Option::as_ref)
}

/// The net, if measuring or the memory model is on.
fn net() -> Option<&'static PageNet<Win>> {
    #[cfg(test)]
    {
        if TEST_MODEL.get() {
            return Some(MODEL.get_or_init(|| PageNet::new(Mode::Net, Win)));
        }
        if !TEST_ON.get() {
            return None;
        }
        NET.get_or_init(|| Some(PageNet::new(Mode::Measure, Win)))
            .as_ref()
    }
    #[cfg(not(test))]
    NET.get_or_init(|| {
        let asked = std::env::var("VICTORY_PAGENET").ok();
        let mode = mode(asked.as_deref(), false);
        match (&asked, mode) {
            (_, Some(Mode::Net)) => {
                ours();
                let section = section_wanted();
                tracing::info!(
                    section,
                    "memory model on (VICTORY_MEMMODEL): pages only the GPU uses stay in its copy, closed to the title"
                );
                if section == 0 {
                    tracing::warn!(
                        "VICTORY_MEMMODEL: no section named (VICTORY_MEMMODEL_SECTION, its size in bytes): nothing is closed"
                    );
                }
                if asked.is_some() {
                    tracing::warn!("VICTORY_PAGENET is ignored with VICTORY_MEMMODEL");
                }
            }
            (Some(_), Some(mode)) => {
                ours();
                tracing::info!(
                    ?mode,
                    "page net on (VICTORY_PAGENET): title pages are closed after copy-backs"
                );
            }
            (Some(asked), None) if asked != "0" && !asked.is_empty() => {
                tracing::warn!(asked, "VICTORY_PAGENET: only `measure` is known; left off");
            }
            _ => {}
        }
        mode.map(|mode| PageNet::new(mode, Win))
    })
    .as_ref()
}

/// Our module's address range (found when the net is made: never in the
/// fault handler).
fn ours() -> Option<(usize, usize)> {
    static OURS: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *OURS.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleHandleExW,
        };
        use windows_sys::Win32::System::ProcessStatus::{K32GetModuleInformation, MODULEINFO};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut module = std::ptr::null_mut();
        let flags =
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        // SAFETY: an address inside our own code, and a valid out pointer.
        if unsafe { GetModuleHandleExW(flags, protection as *const () as *const u16, &mut module) }
            == 0
        {
            return None;
        }
        // SAFETY: plain data filled in by the call.
        let mut info: MODULEINFO = unsafe { std::mem::zeroed() };
        // SAFETY: our module's handle and a valid out pointer of its size.
        let ok = unsafe {
            K32GetModuleInformation(
                GetCurrentProcess(),
                module,
                &mut info,
                size_of_val(&info) as u32,
            )
        };
        (ok != 0).then_some((info.lpBaseOfDll as usize, info.SizeOfImage as usize))
    })
}

/// Runs `f` with the guard lock held: taken here (and handed to `f`), or
/// held by this thread further up already (`None`: its guard set is in use
/// there and must not be touched).
fn locked<R>(f: impl FnOnce(Option<&mut Guards>) -> R) -> R {
    if GUARD_OWNER.load(Relaxed) == this_thread() {
        return f(None);
    }
    let mut g = guards();
    f(Some(&mut g))
}

/// A copy-back on `way` put the GPU's bytes into page `page` (a page
/// number) and the copy has the page taken: armed, with the pages showing
/// the same memory, if all of them are under the write guard.
pub(crate) fn arm(page: usize, way: Way) {
    let Some(net) = net() else { return };
    if net.mode() != Mode::Measure {
        return;
    }
    locked(|g| {
        // Under the lock already: a copy-back does not run there.
        let Some(g) = g else { return };
        let others = g.alias_of.get(&page).cloned().unwrap_or_default();
        let own = g.protected.contains(&page);
        if !own || others.iter().any(|p| !g.protected.contains(p)) {
            NOT_ARMED.fetch_add(1, Relaxed);
            if own {
                NOT_ARMED_ELSEWHERE.fetch_add(1, Relaxed);
            }
            BY_WAY[way as usize][1].fetch_add(1, Relaxed);
            return;
        }
        BY_WAY[way as usize][0].fetch_add(1, Relaxed);
        net.arm(page, Reason::AfterCopyBack);
        if others.is_empty() {
            return;
        }
        let mut beside = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
        for other in others {
            net.arm(other, Reason::AfterCopyBack);
            beside.alias_of.insert(other, page);
        }
    });
}

/// Whether page `page` (a page number) is armed. A copy-back asks before it
/// reads the page (which opens it, [`host_touch`]): finding nothing to put
/// back, it arms the page again, or every copy-back that changes nothing
/// would leave what the GPU wrote earlier open to uncounted reads.
pub(crate) fn armed(page: usize) -> bool {
    net().is_some_and(|net| net.any_armed() && net.state(page) == crate::pagenet::State::ArmedMeasure)
}

/// An access violation reading (or writing, `write`) `at`, taken by the
/// code at `code`: true if the page was armed and the access can be
/// retried. Called by the write guard's handler before it looks itself.
pub(crate) fn fault(at: usize, write: bool, code: usize) -> bool {
    let Some(net) = net_made() else { return false };
    if net.mode() == Mode::Net {
        return fault_of_closed(net, at, write);
    }
    let before = net.any_armed();
    let handled = locked(|_| net.on_fault_from(at, write, false)) == Fault::Handled;
    if handled && before {
        let place = match ours() {
            Some((base, len)) if (base..base + len).contains(&code) => (true, code - base),
            _ => (false, code),
        };
        if place.0 { &IN_OURS } else { &ELSEWHERE }.fetch_add(1, Relaxed);
        // Never waited for: the handler must not hang on a report.
        if let Ok(mut beside) = BESIDE.try_lock() {
            let places = beside.code.len();
            if let Some(n) = beside.code.get_mut(&place) {
                *n += 1;
            } else if places < CODE_PLACES {
                beside.code.insert(place, 1);
            }
        }
    }
    handled
}

/// The engine is about to copy `len` bytes at `at` in the kernel, from or
/// to title memory (or its own): armed pages of the range are opened first,
/// uncounted.
pub(crate) fn host_touch(at: usize, len: usize) {
    let Some(net) = net() else { return };
    if len == 0 || !net.any_armed() {
        return;
    }
    if net.mode() == Mode::Net {
        // Title memory there is older than the GPU's copy: synced first.
        let (first, last) = (at / PAGE, at.saturating_add(len - 1) / PAGE + 1);
        demote_closed(net, first, last, DemoteReason::Host);
        return;
    }
    locked(|_| {
        for page in at / PAGE..=at.saturating_add(len - 1) / PAGE {
            net.host_access(page);
        }
    });
}

/// Title memory `[first, last)` (page numbers) was mapped or unmapped: its
/// pages are forgotten, so a fault on what is mapped there next is not
/// taken for an armed page's.
pub(crate) fn remapped(first: usize, last: usize) {
    let Some(net) = net() else { return };
    if net.mode() == Mode::Net && net.any_armed() {
        // Every page of the title views the range touches: our own address
        // of such a view is forgotten as a whole after this, and a closed
        // page of it could not be synced any more.
        let (first, last) = own().reach(first, last);
        // While our own view still knows the address: the GPU's bytes go
        // into the pages before the title's view of them is gone, and the
        // other addresses showing them open with it.
        demote_closed(net, first, last, DemoteReason::Remap);
    }
    let known: Vec<usize> = net
        .armed_counts()
        .range(first..last)
        .map(|(&p, _)| p)
        .collect();
    if known.is_empty() {
        return;
    }
    locked(|_| {
        for page in known {
            net.disarm_for_free(page);
        }
    });
}

/// How many frames lie between two reports.
const REPORT_FRAMES: u64 = 600;

/// A frame was presented: counted, and every [`REPORT_FRAMES`] frames the
/// counts so far are logged (and written to the file `VICTORY_PAGENET_OUT`
/// names, whole each time).
pub(crate) fn frame() {
    let frame = FRAME.fetch_add(1, Relaxed) + 1;
    if !frame.is_multiple_of(REPORT_FRAMES) {
        return;
    }
    let Some(net) = net() else { return };
    if net.mode() == Mode::Net {
        return;
    }
    let report = report(net, frame);
    tracing::info!(target: "victory_gpu::pagenet", "{}", report.lines().next().unwrap_or_default());
    if let Some(path) = std::env::var_os("VICTORY_PAGENET_OUT")
        && let Err(e) = std::fs::write(&path, &report)
    {
        tracing::warn!(error = %e, "VICTORY_PAGENET_OUT: not written");
    }
}

/// The counts so far: a summary line (after the first counts: copy-backs
/// that armed their page and that did not, by way back, and how many of
/// the latter had only another view of the page open), then one tab-separated line per
/// armed page (`P`, address, times armed, the page it was armed with, title
/// reads, title writes), per 4 KiB block with a fault (`B`, address, reads,
/// writes) and per place of faulting code (`C`, `ours` or `other`, address
/// or module offset, faults).
fn report(net: &PageNet<Win>, frame: u64) -> String {
    use std::fmt::Write;
    let (stats, odd, arms) = (net.stats(), net.oddities(), net.armed_counts());
    let beside = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
    let mut per_page: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    for (&(page, _), &n) in &stats.reads {
        per_page.entry(page).or_default().0 += n;
    }
    for (&(page, _), &n) in &stats.writes {
        per_page.entry(page).or_default().1 += n;
    }
    let read = per_page.values().filter(|c| c.0 != 0).count();
    let written_only = per_page.values().filter(|c| c.0 == 0 && c.1 != 0).count();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "pagenet frame={frame} pages_armed={} arms={} pages_read={read} pages_written_only={written_only} \
         pages_untouched={} read_faults={} write_faults={} host_opens={} overridden={} arm_refused={} \
         copy_backs_not_armed={} faults_in_ours={} faults_elsewhere={} \
         armed_computed={} not_armed_computed={} armed_declared={} not_armed_declared={} \
         armed_marked={} not_armed_marked={} not_armed_other_view_open={}",
        arms.len(),
        arms.values().sum::<u64>(),
        arms.len() - arms.keys().filter(|p| per_page.contains_key(p)).count(),
        stats.reads.values().sum::<u64>(),
        stats.writes.values().sum::<u64>(),
        odd.host_faults,
        odd.overridden,
        odd.arm_lost,
        NOT_ARMED.load(Relaxed),
        IN_OURS.load(Relaxed),
        ELSEWHERE.load(Relaxed),
        BY_WAY[Way::Computed as usize][0].load(Relaxed),
        BY_WAY[Way::Computed as usize][1].load(Relaxed),
        BY_WAY[Way::Declared as usize][0].load(Relaxed),
        BY_WAY[Way::Declared as usize][1].load(Relaxed),
        BY_WAY[Way::Marked as usize][0].load(Relaxed),
        BY_WAY[Way::Marked as usize][1].load(Relaxed),
        NOT_ARMED_ELSEWHERE.load(Relaxed),
    );
    for (&page, &n) in &arms {
        let with = beside.alias_of.get(&page).map_or(0, |p| p * PAGE);
        let (r, w) = per_page.get(&page).copied().unwrap_or_default();
        let _ = writeln!(out, "P\t{:#x}\t{n}\t{with:#x}\t{r}\t{w}", page * PAGE);
    }
    let mut blocks: BTreeMap<(usize, usize), (u64, u64)> = BTreeMap::new();
    for (&at, &n) in &stats.reads {
        blocks.entry(at).or_default().0 += n;
    }
    for (&at, &n) in &stats.writes {
        blocks.entry(at).or_default().1 += n;
    }
    for ((page, block), (r, w)) in blocks {
        let _ = writeln!(out, "B\t{:#x}\t{r}\t{w}", page * PAGE + block * BLOCK);
    }
    for (&(in_ours, at), &n) in &beside.code {
        let _ = writeln!(
            out,
            "C\t{}\t{at:#x}\t{n}",
            if in_ours { "ours" } else { "other" }
        );
    }
    out
}

// ------------------------------------------------------------------------
// The memory model's net.
//
// A page only the GPU uses stays in its host copy: once a copy-back has
// made title memory and the copy equal, the page is closed at every address
// showing it ([`close`]) and its later copy-backs are skipped ([`skips`]).
// Title memory there grows old. It is brought up to date, and the page
// leaves the class for good, at the first of:
// - an access by any thread at any of its addresses (the fault);
// - the engine itself reading or writing it in title memory
//   ([`host_touch`]): another copy or a texture's staging taking the page, a
//   copy step, a kernel write the runtime announces;
// - one of its addresses going away ([`remapped`]), or a new one that
//   cannot be closed ([`aliased`]);
// - shaders' writes to it being withheld ([`withheld`]).
//
// Which pages: only where exactly one tracker, the closing copy's, holds the
// physical page over all its addresses ([`eligible`]). Another host copy of
// the page gets what the GPU wrote only by way of title memory (the
// copy-back marks it changed, and it reads the page again): with the
// copy-back skipped it would go stale without any CPU access, which no
// fault catches. A tracker that comes later starts with every page changed
// and reads title memory before it uses a byte, which syncs.
//
// Locks, in order: the runtime's page lock (it calls `remapped` and
// `aliased` holding it); the submission lock and `SYNC` ([`idle`]); `OWN`
// (the forced copy-back); the guard lock; the net's own. A sync never
// starts on a thread that holds the guard lock or is syncing already
// ([`may_sync`]): the paths that take the guard lock sync what they are
// about to touch before they take it.
//
// shortcut: as for measuring, a system call of the title's reading from a
// closed page fails instead of faulting; with the model on that is a title
// read that sees an error where it should see the GPU's bytes. Not counted:
// a run with the model is compared with one without for new failing calls.

thread_local! {
    /// This thread is syncing: what it touches meanwhile is the sync's own.
    static IN_SYNC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One sync at a time, also among the threads the submission lock lets
/// through together (the submitting thread and its copy-back workers).
static SYNC: Mutex<()> = Mutex::new(());

/// Runs `f` with no submission under way, the last one's copy-backs done,
/// and no other sync running: what [`Protect::fence_wait`] waits for.
fn idle<R>(f: impl FnOnce() -> R) -> R {
    crate::host::submission_idle(|| {
        let _one = SYNC.lock().unwrap_or_else(|e| e.into_inner());
        IN_SYNC.set(true);
        let done = f();
        IN_SYNC.set(false);
        done
    })
}

/// Why a page was not closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Not in the section the model was given, or in no view we have one
    /// of our own of.
    Section,
    /// Another tracker holds the page, at this or another address.
    Holders,
    /// An address showing it is not under the write guard.
    Unguarded,
    /// The title wrote it since the copy took it.
    Changed,
    /// The net refused: demoted before, or an address could not be closed.
    Net,
}

const REFUSALS: usize = 5;

/// Pages closed by [`Way`], copy-backs skipped by [`Way`], pages not closed
/// by [`Refusal`].
static CLOSED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static SKIPPED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static REFUSED: [AtomicU64; REFUSALS] = [const { AtomicU64::new(0) }; REFUSALS];
/// Syncs whose copy did not land: the page opened with older bytes.
static FORCED_FAILED: AtomicU64 = AtomicU64::new(0);
/// Closed pages met on a thread that could not sync ([`may_sync`]).
static CANNOT_SYNC: AtomicU64 = AtomicU64::new(0);
/// Closed pages the write guard was about to open, left closed.
static OPEN_REFUSED: AtomicU64 = AtomicU64::new(0);
/// The size in bytes of the section whose pages may be closed (0: none;
/// `usize::MAX`: any).
static SECTION: AtomicUsize = AtomicUsize::new(if cfg!(test) { usize::MAX } else { 0 });

fn section_wanted() -> usize {
    SECTION.load(Relaxed)
}

/// The copies holding the GPU's bytes of closed pages, by the page each
/// closed (its group's page): kept alive while the page is closed, since
/// nothing else has those bytes.
static OWNERS: Mutex<BTreeMap<usize, Arc<Shadow>>> = Mutex::new(BTreeMap::new());
/// Owners of pages demoted since, let go at the next frame (not in a fault
/// handler).
static RETIRED: Mutex<Vec<Arc<Shadow>>> = Mutex::new(Vec::new());

fn owners() -> std::sync::MutexGuard<'static, BTreeMap<usize, Arc<Shadow>>> {
    OWNERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The group of `group` was synced: its owner is not needed any more.
fn retire(net: &PageNet<Win>, group: Option<usize>) {
    let Some(group) = group else { return };
    if net.group_of(group).is_some() {
        return;
    }
    let owner = owners().remove(&group);
    if let Some(owner) = owner {
        RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(owner);
    }
}

/// Whether this thread can sync a closed page now: not while it holds the
/// guard lock (the submission lock comes first) and not inside a sync.
/// Counted where it cannot: the access then sees the closed page (a kernel
/// copy fails, a fault goes on to the next handler).
fn may_sync() -> bool {
    if IN_SYNC.get() {
        return false;
    }
    if GUARD_OWNER.load(Relaxed) == this_thread() {
        CANNOT_SYNC.fetch_add(1, Relaxed);
        return false;
    }
    true
}

/// A fault at `at` with the model on: a closed page is synced and demoted
/// with the pages closed with it, then the access is retried.
fn fault_of_closed(net: &PageNet<Win>, at: usize, write: bool) -> bool {
    let page = at / PAGE;
    match net.state(page) {
        State::GpuOnlyArmed | State::Syncing => {}
        // Opened by another thread since this one faulted.
        State::Demoted => return net.on_fault_from(at, write, false) == Fault::Handled,
        _ => return false,
    }
    if !may_sync() {
        return false;
    }
    let group = net.group_of(page);
    let handled = idle(|| net.on_fault_from(at, write, false)) == Fault::Handled;
    retire(net, group);
    handled
}

/// The closed pages of `[first, last)` are synced and demoted.
fn demote_closed(net: &PageNet<Win>, first: usize, last: usize, reason: DemoteReason) {
    let closed = net.closed_in(first, last);
    if closed.is_empty() || !may_sync() {
        return;
    }
    for page in closed {
        let group = net.group_of(page);
        idle(|| net.demote(page, reason));
        retire(net, group);
    }
}

/// The pages to close with `page` (itself first), if the copy tracked by
/// `t` may keep it to itself: its tracker is the only one over every
/// address showing the page, all of them are under the write guard, and the
/// title has not written the page since the copy took it (title memory and
/// the copy are equal).
fn eligible(g: &Guards, page: usize, t: &Tracked) -> Result<Vec<usize>, Refusal> {
    let mut others = g.alias_of.get(&page).cloned().unwrap_or_default();
    others.retain(|&o| o != page);
    others.sort_unstable();
    others.dedup();
    let alone = g.holders.get(&page).is_some_and(|h| h.len() == 1 && Arc::ptr_eq(&h[0], &t.changed));
    if !alone || others.iter().any(|o| g.holders.contains_key(o)) {
        return Err(Refusal::Holders);
    }
    if !g.protected.contains(&page) || others.iter().any(|o| !g.protected.contains(o)) {
        return Err(Refusal::Unguarded);
    }
    if t.changed.0.lock().unwrap_or_else(|e| e.into_inner()).contains(&page) {
        return Err(Refusal::Changed);
    }
    Ok(std::iter::once(page).chain(others).collect())
}

/// A copy-back on `way` made title memory and `copy` equal in the whole
/// page `page` (a page number), which `copy` has taken: from now on the
/// page is GPU-only, closed at every address showing it, if it may be
/// ([`eligible`], and in the model's section).
pub(crate) fn close(page: usize, way: Way, copy: &Shadow) {
    let Some(net) = net() else { return };
    if net.mode() != Mode::Net {
        return;
    }
    let wanted = section_wanted();
    let (Some(t), Some(me)) = (copy.tracker(), copy.shared()) else {
        return;
    };
    if wanted == 0 || net.is_gpu_only(page) {
        return;
    }
    let section = own().section_of(page * PAGE, PAGE);
    if !section.is_some_and(|bytes| wanted == usize::MAX || bytes == wanted) {
        REFUSED[Refusal::Section as usize].fetch_add(1, Relaxed);
        return;
    }
    // The owner is known before the page closes: a fault right after finds
    // the copy to sync from.
    let fresh = {
        let mut owners = owners();
        let fresh = !owners.contains_key(&page);
        if fresh {
            owners.insert(page, me);
        }
        fresh
    };
    let armed = locked(|g| {
        // Under the lock already: a copy-back does not run there.
        let g = g?;
        match eligible(g, page, t) {
            Ok(group) => Some(net.arm_group(&group)),
            Err(why) => {
                REFUSED[why as usize].fetch_add(1, Relaxed);
                None
            }
        }
    });
    match armed {
        Some(Armed::Now) => {
            CLOSED[way as usize].fetch_add(1, Relaxed);
        }
        Some(Armed::Refused) => {
            REFUSED[Refusal::Net as usize].fetch_add(1, Relaxed);
        }
        Some(Armed::Already) | None => {}
    }
    if fresh && armed != Some(Armed::Now) {
        let owner = owners().remove(&page);
        drop(owner);
    }
}

/// Whether the copy-back on `way` of page `page` (a page number) of `copy`
/// is skipped: the page is GPU-only and `copy` is the one keeping it. Asked
/// before the copy-back touches title memory.
pub(crate) fn skips(page: usize, way: Way, copy: &Shadow) -> bool {
    let ours = kept(page, copy);
    if ours {
        SKIPPED[way as usize].fetch_add(1, Relaxed);
    }
    ours
}

/// Whether page `page` (a page number) is GPU-only and `copy` is the one
/// keeping it.
pub(crate) fn kept(page: usize, copy: &Shadow) -> bool {
    let Some(net) = net() else { return false };
    if net.mode() != Mode::Net || !net.any_armed() || !net.is_gpu_only(page) {
        return false;
    }
    owners().get(&page).is_some_and(|owner| std::ptr::eq(Arc::as_ptr(owner), copy))
}

/// What shaders wrote into pages `[first, last)` is to be dropped from the
/// copies (`super::retake`): closed pages there have no older bytes to go
/// back to, so they keep the GPU's and leave the class.
pub(super) fn withheld(first: usize, last: usize) {
    let Some(net) = net() else { return };
    if net.mode() == Mode::Net && net.any_armed() {
        demote_closed(net, first, last, DemoteReason::Withheld);
    }
}

/// The runtime mapped page `page` where `others` show it already (called
/// with the guard lock held, `page` guarded if they are): it is closed with
/// them if they are closed. False if it could not be: the caller demotes
/// them once the guard lock is free ([`unjoined`]).
pub(super) fn aliased(page: usize, others: &[usize]) -> bool {
    let Some(net) = net_made() else { return true };
    if net.mode() != Mode::Net || !net.any_armed() {
        return true;
    }
    others.iter().all(|&other| net.join(other, page))
}

/// After [`aliased`] failed, with the guard lock free again.
pub(super) fn unjoined(others: &[usize]) {
    let Some(net) = net() else { return };
    for &other in others {
        demote_closed(net, other, other + 1, DemoteReason::Remap);
    }
}

/// Whether a page of `[first, last)` is closed as GPU-only now (called with
/// the guard lock held: none is closed while the caller holds it).
pub(super) fn closed(first: usize, last: usize) -> bool {
    let Some(net) = net_made() else { return false };
    net.mode() == Mode::Net && !net.closed_in(first, last).is_empty()
}

/// A kernel copy between `a` and `b` (`len` bytes each) failed: true if a
/// page of either was closed as GPU-only meanwhile (closed after
/// [`host_touch`] looked) and is synced now, so the copy is worth trying
/// once more. A copy that failed for another reason is not retried.
pub(crate) fn touched_again(a: usize, b: usize, len: usize) -> bool {
    let Some(net) = net() else { return false };
    if len == 0 || net.mode() != Mode::Net || !net.any_armed() {
        return false;
    }
    let pages = |at: usize| (at / PAGE, at.saturating_add(len - 1) / PAGE + 1);
    let was = [a, b].iter().any(|&at| !net.closed_in(pages(at).0, pages(at).1).is_empty());
    if was {
        host_touch(a, len);
        host_touch(b, len);
    }
    was
}

/// Whether the write guard must leave page `page` closed (called with the
/// guard lock held, before it opens the page): it is GPU-only, and opening
/// it here would show the title old bytes. The page stays as it is; the
/// next access faults and syncs.
pub(super) fn keeps_closed(page: usize) -> bool {
    let Some(net) = net_made() else { return false };
    if net.mode() != Mode::Net || !net.any_armed() || net.group_of(page).is_none() {
        return false;
    }
    OPEN_REFUSED.fetch_add(1, Relaxed);
    true
}

#[cfg(test)]
pub(crate) mod tests {
    /// For the engine's tests: (title reads, title writes) counted at the
    /// 4 KiB block of `at`, and whether the page of `at` is closed.
    pub(crate) fn seen(at: usize) -> ((u64, u64), bool) {
        (counts(at), protection(at / PAGE) == Some(PAGE_NOACCESS))
    }

    /// The model's net (made by the first test that switches it on).
    fn model() -> &'static PageNet<Win> {
        MODEL.get_or_init(|| PageNet::new(Mode::Net, Win))
    }

    /// Where page `page` stands in the model's net.
    pub(crate) fn state(page: usize) -> State {
        model().state(page)
    }

    /// The demotion recorded for page `page` (the page whose access did
    /// it): its reason and how many addresses left together.
    pub(crate) fn demotion(page: usize) -> Option<(DemoteReason, usize)> {
        model().demotions().iter().find(|d| d.page == page).map(|d| (d.reason, d.pages))
    }

    /// How many demotions are recorded for `pages`.
    pub(crate) fn demotions_of(pages: &[usize]) -> usize {
        model().demotions().iter().filter(|d| pages.contains(&d.page)).count()
    }

    /// Of every test running, only ever more: copy-backs skipped, pages
    /// closed, syncs, pages refused for `why`.
    pub(crate) fn skipped() -> u64 {
        SKIPPED.iter().map(|n| n.load(Relaxed)).sum()
    }

    pub(crate) fn closed() -> u64 {
        CLOSED.iter().map(|n| n.load(Relaxed)).sum()
    }

    pub(crate) fn syncs() -> u64 {
        model().stats().syncs
    }

    pub(crate) fn refused(why: Refusal) -> u64 {
        REFUSED[why as usize].load(Relaxed)
    }

    /// Copy-backs on `way` so far that armed their page, and that did not
    /// (of every test running: only ever more).
    pub(crate) fn by_way(way: Way) -> (u64, u64) {
        let [armed, not] = &BY_WAY[way as usize];
        (armed.load(Relaxed), not.load(Relaxed))
    }

    /// The report's summary line.
    pub(crate) fn summary() -> String {
        report(net().expect("on"), 0).lines().next().unwrap_or_default().to_owned()
    }

    fn arm(page: usize) {
        super::arm(page, Way::Computed);
    }

    use super::super::tests::{free, pool};
    use super::super::{track, writing};
    use super::*;
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};

    #[test]
    fn only_measure_switches_it_on() {
        assert_eq!(mode(Some("measure"), false), Some(Mode::Measure));
        assert_eq!(mode(None, true), Some(Mode::Net));
        assert_eq!(mode(Some("measure"), true), Some(Mode::Net), "the model wins");
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("1"),
            Some("net"),
            Some("Measure"),
        ] {
            assert_eq!(mode(off, false), None, "{off:?}");
        }
    }

    /// A pool of `pages` pages holding `fill`, tracked and taken (guarded),
    /// with measuring on for this thread.
    fn taken(pages: usize, fill: u8) -> (usize, crate::watch::Tracked) {
        TEST_ON.set(true);
        let p = pool(pages);
        // SAFETY: a fresh allocation of that size.
        unsafe { (p as *mut u8).write_bytes(fill, pages * PAGE) };
        let t = track(p, pages * PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + pages * PAGE), [(p, p + pages * PAGE)]);
        (p, t)
    }

    fn counts(at: usize) -> (u64, u64) {
        let stats = net().expect("on").stats();
        let key = (at / PAGE, at % PAGE / BLOCK);
        (
            stats.reads.get(&key).copied().unwrap_or(0),
            stats.writes.get(&key).copied().unwrap_or(0),
        )
    }

    /// A title read of an armed page is counted at its 4 KiB block and the
    /// page is under the write guard again; the write after it is the
    /// guard's (the tracker sees it), not counted.
    #[test]
    fn a_read_of_an_armed_page_is_counted_and_the_guard_is_back() {
        let (p, t) = taken(2, 0x5a);
        arm(p / PAGE + 1);
        assert_eq!(
            protection(p / PAGE),
            Some(PAGE_READONLY),
            "only the page asked for"
        );
        assert_eq!(protection(p / PAGE + 1), Some(PAGE_NOACCESS), "armed");
        let at = p + PAGE + 3 * BLOCK + 7;
        // SAFETY: a committed page of ours, closed: the handler opens it.
        let read = unsafe { (at as *const u8).read_volatile() };
        assert_eq!(read, 0x5a);
        assert_eq!(counts(at), (1, 0));
        assert_eq!(
            protection(p / PAGE + 1),
            Some(PAGE_READONLY),
            "the guard is back"
        );
        assert!(t.guarded(p / PAGE + 1) && !t.changed(p / PAGE + 1));
        // SAFETY: as above; read-only now: the write guard's fault.
        unsafe { (at as *mut u8).write_volatile(1) };
        assert!(
            t.changed(p / PAGE + 1),
            "the write is tracked as without the net"
        );
        assert_eq!(counts(at), (1, 0), "and is the guard's fault, not ours");
        assert_eq!(protection(p / PAGE + 1), Some(PAGE_READWRITE));
        drop(t);
        free(p);
    }

    /// A title write to an armed page reaches the net before the write
    /// guard: counted as a write, then tracked.
    #[test]
    fn a_write_to_an_armed_page_is_counted_then_tracked() {
        let (p, t) = taken(1, 0);
        arm(p / PAGE);
        let at = p + 5 * BLOCK;
        // SAFETY: a committed page of ours, closed: the handler opens it,
        // and the write guard's fault follows.
        unsafe { (at as *mut u8).write_volatile(9) };
        assert_eq!(counts(at), (0, 1));
        assert!(t.changed(p / PAGE), "tracked");
        // SAFETY: open now.
        assert_eq!(unsafe { (at as *const u8).read_volatile() }, 9);
        drop(t);
        free(p);
    }

    /// The engine's kernel copy from an armed page works (it opens the page
    /// first) and is no title read; a page that is not under the guard is
    /// not armed at all.
    #[test]
    fn the_engines_copy_opens_an_armed_page_uncounted() {
        let (p, t) = taken(2, 0x77);
        arm(p / PAGE);
        assert_eq!(protection(p / PAGE), Some(PAGE_NOACCESS));
        let mut bytes = [0u8; 16];
        assert!(
            crate::host::read_into((p + 100) as *const u8, &mut bytes),
            "readable for the engine"
        );
        assert_eq!(bytes, [0x77; 16]);
        assert_eq!(counts(p + 100), (0, 0), "not the title's read");
        assert_eq!(protection(p / PAGE), Some(PAGE_READONLY));
        assert!(t.guarded(p / PAGE));

        // SAFETY: a guarded page of ours: the guard opens it.
        unsafe { ((p + PAGE) as *mut u8).write_volatile(1) };
        let not_armed = NOT_ARMED.load(Relaxed);
        arm(p / PAGE + 1);
        assert_eq!(
            protection(p / PAGE + 1),
            Some(PAGE_READWRITE),
            "open: not armed"
        );
        assert!(NOT_ARMED.load(Relaxed) > not_armed);
        drop(t);
        free(p);
    }

    /// A kernel write the runtime announces opens an armed page as it does
    /// a guarded one (the write guard's protection replaces ours), also
    /// when the guard's code reads the closed page while holding the guard
    /// lock (it saves an armed tracker's bytes before the write): that read
    /// is let through without the lock, where it would wait for itself.
    #[test]
    fn an_announced_kernel_write_opens_an_armed_page() {
        let (p, t) = taken(1, 0x11);
        t.arm(p, p + PAGE);
        arm(p / PAGE);
        assert_eq!(protection(p / PAGE), Some(PAGE_NOACCESS));
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            TEST_ON.set(true);
            drop(writing(p + 8, 4));
            tx.send(()).ok();
        });
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("the write's announcement returns (no wait for its own lock)");
        worker.join().unwrap();
        assert_eq!(
            protection(p / PAGE),
            Some(PAGE_READWRITE),
            "open for the kernel"
        );
        assert!(t.changed(p / PAGE));
        assert_eq!(
            t.saved(p / PAGE).as_deref().map(|b| b[8]),
            Some(0x11),
            "saved from the closed page"
        );
        drop(t);
        free(p);
    }

    /// The write guard never opens a page closed as GPU-only, whoever asks
    /// it to (title memory there is old): the page stays closed and under
    /// the guard, and its next access syncs.
    #[test]
    fn the_write_guard_leaves_a_gpu_only_page_closed() {
        let (p, t) = taken(1, 0x33);
        TEST_MODEL.set(true);
        let net = net().expect("on");
        assert_eq!(net.mode(), Mode::Net);
        assert_eq!(locked(|_| net.arm_group(&[p / PAGE])), Armed::Now);
        assert_eq!(protection(p / PAGE), Some(PAGE_NOACCESS));
        {
            let mut g = guards();
            assert!(!super::super::open(&mut g, p / PAGE), "refused");
            super::super::changed(&mut g, p / PAGE, p / PAGE + 1, false);
            assert!(g.protected.contains(&(p / PAGE)), "still the guard's");
        }
        assert_eq!(protection(p / PAGE), Some(PAGE_NOACCESS), "still closed");
        assert_eq!(net.state(p / PAGE), State::GpuOnlyArmed);
        // The next access syncs (no copy keeps this page: nothing lands,
        // which is counted) and opens it for good.
        let failed = FORCED_FAILED.load(Relaxed);
        // SAFETY: a committed page of ours, closed: the handler opens it.
        assert_eq!(unsafe { (p as *const u8).read_volatile() }, 0x33);
        assert_eq!(net.state(p / PAGE), State::Demoted);
        assert!(FORCED_FAILED.load(Relaxed) > failed, "a sync with nothing to copy from is never silent");
        TEST_MODEL.set(false);
        drop(t);
        free(p);
    }

    /// Unmapped or remapped memory is forgotten: a fault there is not ours.
    #[test]
    fn remapped_pages_are_forgotten() {
        let (p, t) = taken(1, 0);
        arm(p / PAGE);
        assert_eq!(protection(p / PAGE), Some(PAGE_NOACCESS));
        crate::watch::title_memory_remapped(p, PAGE);
        assert_eq!(
            net().expect("on").state(p / PAGE),
            crate::pagenet::State::Shared
        );
        assert!(!fault(p, false, 0), "not ours any more");
        drop(t);
        free(p);
    }
}
