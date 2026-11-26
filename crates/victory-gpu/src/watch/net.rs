//! [`crate::pagenet`] on real title memory: which pages the title's CPU
//! code touches after the GPU wrote them (`VICTORY_PAGENET=measure`; off
//! unless asked for).
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
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use super::{GUARD_OWNER, Guards, PAGE, guards, this_thread};
use crate::pagenet::{BLOCK, Fault, Mode, PageNet, Prot, Protect, Reason};

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

    // Measuring never syncs; the net's mode is not offered yet (`mode`).
    fn fence_wait(&self) {}

    fn copy_back_forced(&self, _page: usize) {}

    fn frame(&self) -> u64 {
        FRAME.load(Relaxed)
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

/// What `VICTORY_PAGENET` asks for. Only measuring: the net needs the
/// engine's side of the sync, which is not there yet, and arming without it
/// would hand the title stale pages.
fn mode(value: Option<&str>) -> Option<Mode> {
    match value {
        Some("measure") => Some(Mode::Measure),
        _ => None,
    }
}

#[cfg(test)]
thread_local! {
    /// Tests switch measuring on for their own thread: the crate's other
    /// tests run as without it.
    pub(crate) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static NET: OnceLock<Option<PageNet<Win>>> = OnceLock::new();

/// The net, if it was made: the fault handler's way to it, which must not
/// read the environment or log.
fn net_made() -> Option<&'static PageNet<Win>> {
    #[cfg(test)]
    if !TEST_ON.get() {
        return None;
    }
    NET.get().and_then(Option::as_ref)
}

/// The net, if measuring is on.
fn net() -> Option<&'static PageNet<Win>> {
    #[cfg(test)]
    {
        if !TEST_ON.get() {
            return None;
        }
        NET.get_or_init(|| Some(PageNet::new(Mode::Measure, Win)))
            .as_ref()
    }
    #[cfg(not(test))]
    NET.get_or_init(|| {
        let asked = std::env::var("VICTORY_PAGENET").ok();
        let mode = mode(asked.as_deref());
        match (&asked, mode) {
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

#[cfg(test)]
pub(crate) mod tests {
    /// For the engine's tests: (title reads, title writes) counted at the
    /// 4 KiB block of `at`, and whether the page of `at` is closed.
    pub(crate) fn seen(at: usize) -> ((u64, u64), bool) {
        (counts(at), protection(at / PAGE) == Some(PAGE_NOACCESS))
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
        assert_eq!(mode(Some("measure")), Some(Mode::Measure));
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("1"),
            Some("net"),
            Some("Measure"),
        ] {
            assert_eq!(mode(off), None, "{off:?}");
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
