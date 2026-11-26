//! Tests t1-t8 for [`crate::pagenet`] (read measurement and the memory
//! model's safety net), against a model of the system and the engine.
//!
//! The test model ([`Mock`]) stands for the OS and the engine: per 64 KiB
//! page a protection (the baseline is the write guard of `crate::watch`:
//! read-only once taken), title memory, the host copy (what the GPU sees and
//! writes), the existing write tracker's "changed since taken" set, a fence
//! with GPU writes that land only when their submission completes, and an
//! ordered event log. A title access ([`access`]) behaves as on hardware:
//! no access faults into `PageNet::on_fault`; a write to a read-only page is
//! the existing guard (marks changed, opens, no `PageNet` call).
//!
//! What the model cannot show is the engine's side of the hooks: that
//! `copy_back_forced` really keeps every GPU byte without opening the
//! title's view, and that the fault reaches the net before the write guard's
//! handler. Those are tested where they are implemented.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::pagenet::{
    BLOCK, DemoteReason, Demotion, Fault, Mode, Oddities, PAGE, PageNet, Prot, Protect, Reason,
    State, Stats,
};

// -------------------------------------------------------- test model ---

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ev {
    Protect(usize, Prot),
    FenceWait,
    FenceDone,
    CopyStart(usize),
    CopyDone(usize),
    Refresh(usize),
}

struct MockState {
    prot: BTreeMap<usize, Prot>,
    title: BTreeMap<usize, Vec<u8>>,
    /// The host copy: what the GPU reads and writes.
    copy: BTreeMap<usize, Vec<u8>>,
    /// The existing tracker: pages written since taken.
    changed: BTreeSet<usize>,
    /// GPU writes of the submission in flight: land at `complete`.
    pending: Vec<(usize, u8)>,
    submitted: u64,
    completed: u64,
    frame: u64,
    /// While false, `copy_back_forced` blocks after logging `CopyStart` (t4).
    gate_open: bool,
    log: Vec<Ev>,
}

struct Mock {
    s: Mutex<MockState>,
    cv: Condvar,
}

fn pattern(at: usize) -> u8 {
    (at * 7 + 1) as u8
}

impl Mock {
    /// Pages `pages` (page numbers) mapped, title = copy = pattern, taken:
    /// read-only (the baseline guard), nothing changed, nothing in flight.
    fn new(pages: &[usize]) -> Arc<Mock> {
        let mut s = MockState {
            prot: BTreeMap::new(),
            title: BTreeMap::new(),
            copy: BTreeMap::new(),
            changed: BTreeSet::new(),
            pending: Vec::new(),
            submitted: 0,
            completed: 0,
            frame: 1,
            gate_open: true,
            log: Vec::new(),
        };
        for &p in pages {
            let bytes: Vec<u8> = (0..PAGE).map(|i| pattern(p * PAGE + i)).collect();
            s.prot.insert(p, Prot::ReadOnly);
            s.title.insert(p, bytes.clone());
            s.copy.insert(p, bytes);
        }
        Arc::new(Mock {
            s: Mutex::new(s),
            cv: Condvar::new(),
        })
    }

    fn st(&self) -> std::sync::MutexGuard<'_, MockState> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A GPU write of `value` at `addr` in a new submission, still running.
    fn gpu_write_pending(&self, addr: usize, value: u8) {
        let mut s = self.st();
        s.submitted += 1;
        s.pending.push((addr, value));
    }

    /// The submission in flight completes: its GPU writes land in the copy.
    fn complete(&self) {
        let mut s = self.st();
        for (addr, v) in std::mem::take(&mut s.pending) {
            s.copy.get_mut(&(addr / PAGE)).expect("mapped")[addr % PAGE] = v;
        }
        s.completed = s.submitted;
        drop(s);
        self.cv.notify_all();
    }

    /// A finished GPU write (submit + complete).
    fn gpu_write(&self, addr: usize, value: u8) {
        self.gpu_write_pending(addr, value);
        self.complete();
    }

    /// The engine's own copy-back of a shared page after a submission
    /// (baseline path, no discard here: pages in the tests are not written
    /// by the title meanwhile), then the page is taken (read-only).
    fn engine_copy_back(&self, page: usize) {
        let mut s = self.st();
        let bytes = s.copy[&page].clone();
        s.title.insert(page, bytes);
        s.changed.remove(&page);
        s.prot.insert(page, Prot::ReadOnly);
        s.log.push(Ev::Protect(page, Prot::ReadOnly));
    }

    /// The refresh before the next submission: changed pages go to the
    /// copy (what makes a dirty page visible to the next GPU read, I4).
    fn refresh(&self) {
        let mut s = self.st();
        for page in std::mem::take(&mut s.changed) {
            let bytes = s.title[&page].clone();
            s.copy.insert(page, bytes);
            if s.prot[&page] == Prot::ReadWrite {
                s.prot.insert(page, Prot::ReadOnly);
            }
            s.log.push(Ev::Refresh(page));
        }
    }

    fn next_frame(&self) {
        self.st().frame += 1;
    }

    fn close_gate(&self) {
        self.st().gate_open = false;
    }

    fn open_gate(&self) {
        self.st().gate_open = true;
        self.cv.notify_all();
    }

    fn title(&self, addr: usize) -> u8 {
        self.st().title[&(addr / PAGE)][addr % PAGE]
    }

    fn copy(&self, addr: usize) -> u8 {
        self.st().copy[&(addr / PAGE)][addr % PAGE]
    }

    fn prot(&self, page: usize) -> Prot {
        self.st().prot[&page]
    }

    fn changed(&self, page: usize) -> bool {
        self.st().changed.contains(&page)
    }

    fn log(&self) -> Vec<Ev> {
        self.st().log.clone()
    }

    fn count(&self, ev: &Ev) -> usize {
        self.st().log.iter().filter(|e| *e == ev).count()
    }

    fn index(&self, ev: &Ev) -> Option<usize> {
        self.st().log.iter().position(|e| e == ev)
    }

    /// Waits up to 2 s until `ev` is logged.
    fn wait_for(&self, ev: &Ev) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let mut s = self.st();
        while !s.log.contains(ev) {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return false;
            }
            s = self
                .cv
                .wait_timeout(s, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

impl Protect for Arc<Mock> {
    fn swap(&self, page: usize, from: Prot, to: Prot) -> bool {
        let mut s = self.st();
        if s.prot.get(&page) != Some(&from) {
            return false;
        }
        s.prot.insert(page, to);
        s.log.push(Ev::Protect(page, to));
        true
    }

    fn get(&self, page: usize) -> Prot {
        self.st().prot.get(&page).copied().unwrap_or(Prot::NoAccess)
    }

    fn fence_wait(&self) {
        let mut s = self.st();
        s.log.push(Ev::FenceWait);
        while s.completed < s.submitted {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.log.push(Ev::FenceDone);
    }

    fn copy_back_forced(&self, page: usize) {
        let mut s = self.st();
        s.log.push(Ev::CopyStart(page));
        self.cv.notify_all();
        while !s.gate_open {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        let bytes = s.copy[&page].clone();
        s.title.insert(page, bytes);
        s.log.push(Ev::CopyDone(page));
        self.cv.notify_all();
    }

    fn frame(&self) -> u64 {
        self.st().frame
    }
}

type Net = PageNet<Arc<Mock>>;

/// A title thread `tid` reads (`write == None`) or writes `addr`, as the
/// hardware would run it: no access faults into the net and is retried
/// once after `Handled`; a write to a read-only page is the existing guard
/// (changed, opened). Returns the byte read or written.
fn access(net: &Net, mock: &Arc<Mock>, addr: usize, write: Option<u8>, tid: u32) -> u8 {
    let page = addr / PAGE;
    for attempt in 0..2 {
        let prot = mock.prot(page);
        if prot == Prot::NoAccess {
            assert!(
                attempt == 0,
                "{addr:#x} still no access after Handled: would fault forever"
            );
            assert_eq!(
                net.on_fault(addr, write.is_some(), tid),
                Fault::Handled,
                "an armed page's fault is the net's (else the crash reporter)"
            );
            continue;
        }
        let mut s = mock.st();
        return match write {
            None => s.title[&page][addr % PAGE],
            Some(v) => {
                if prot == Prot::ReadOnly {
                    // The existing guard: marks changed, opens.
                    s.changed.insert(page);
                    s.prot.insert(page, Prot::ReadWrite);
                }
                s.title.get_mut(&page).expect("mapped")[addr % PAGE] = v;
                v
            }
        };
    }
    unreachable!()
}

/// Address of 4 KiB block `block`, byte `byte`, of page `page`.
fn at(page: usize, block: usize, byte: usize) -> usize {
    page * PAGE + block * BLOCK + byte
}

const TITLE_TID: u32 = 100;
const TITLE_TID_2: u32 = 101;
const HOST_TID: u32 = 7;

// ------------------------------------------------------------- tests ---

/// t1 (I2): a read fault on a GPU-only page syncs before the thread runs on.
///
/// Input: page P = 0x10 (pattern, taken, read-only); Net; `arm(P, GpuOnly)`;
/// the GPU writes X = 0xa5 at block 2 byte 8 in a submission still running.
/// Run: title thread 100 reads that byte (on its own thread). After 100 ms
/// it has not returned and no copy started (the sync waits for the fence);
/// then the submission completes.
/// Expected: the read returns X; the log orders FenceDone < CopyStart(P) <
/// CopyDone(P) < first protection change of P after arming (open); state
/// Demoted, `is_gpu_only(P)` false, `stats().syncs == 1`; `drain_demotions()`
/// is exactly `[Demotion { page: P, reason: CpuRead, frame: 1 }]`, a second
/// drain is empty (one log entry).
#[test]
fn t1_net_fault_syncs_before_resume() {
    const P: usize = 0x10;
    const X: u8 = 0xa5;
    let mock = Mock::new(&[P]);
    let net = Arc::new(PageNet::new(Mode::Net, mock.clone()));
    net.arm(P, Reason::GpuOnly);
    assert_eq!(mock.prot(P), Prot::NoAccess, "armed: no access");
    assert!(net.is_gpu_only(P));
    let armed_at = mock.log().len();
    let addr = at(P, 2, 8);
    mock.gpu_write_pending(addr, X);

    let (tx, rx) = mpsc::channel();
    let (n, m) = (net.clone(), mock.clone());
    let reader =
        std::thread::spawn(move || tx.send(access(&n, &m, addr, None, TITLE_TID)).unwrap());
    assert!(
        rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "I2: the faulting thread waits for the submission in flight"
    );
    assert_eq!(
        mock.count(&Ev::CopyStart(P)),
        0,
        "I2: no copy before the fence"
    );
    mock.complete();
    let read = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("the read returns after the fence");
    reader.join().unwrap();

    assert_eq!(read, X, "the CPU reads what the GPU wrote");
    let fence = mock.index(&Ev::FenceDone).expect("fence waited");
    let start = mock.index(&Ev::CopyStart(P)).expect("copy started");
    let done = mock.index(&Ev::CopyDone(P)).expect("copy done");
    let open = mock.log()[armed_at..]
        .iter()
        .position(|e| matches!(e, Ev::Protect(p, prot) if *p == P && *prot != Prot::NoAccess))
        .map(|i| i + armed_at)
        .expect("page opened");
    assert!(
        fence < start && start < done && done < open,
        "Fence -> copy -> open: {:?}",
        mock.log()
    );
    assert_eq!(net.state(P), State::Demoted);
    assert!(!net.is_gpu_only(P));
    assert_eq!(net.stats().syncs, 1);
    assert_eq!(
        net.drain_demotions(),
        vec![Demotion {
            page: P,
            reason: DemoteReason::CpuRead,
            frame: 1
        }]
    );
    assert!(net.drain_demotions().is_empty(), "one log entry");
}

/// t2 (I3): a demoted page stays demoted.
///
/// Input: page P = 0x11, Net, `arm(P, GpuOnly)`, title read at block 0
/// (fault -> demoted, as t1, no submission in flight).
/// Run: 5 frames, each `next_frame()` then `arm(P, GpuOnly)`; then a title
/// read and write of P.
/// Expected: after each arm `is_gpu_only(P)` false and state Demoted; no
/// `Protect(P, NoAccess)` in the log after the demotion; `stats().arm_noops
/// == 5`, `stats().syncs == 1`; the later accesses do not fault (no
/// `on_fault` call: `access` would panic on a second NoAccess); exactly one
/// demotion drained.
#[test]
fn t2_demoted_is_sticky() {
    const P: usize = 0x11;
    let mock = Mock::new(&[P]);
    let net = PageNet::new(Mode::Net, mock.clone());
    net.arm(P, Reason::GpuOnly);
    access(&net, &mock, at(P, 0, 0), None, TITLE_TID);
    assert_eq!(net.state(P), State::Demoted);
    let demoted_at = mock.log().len();
    for _ in 0..5 {
        mock.next_frame();
        net.arm(P, Reason::GpuOnly);
        assert!(!net.is_gpu_only(P), "I3: never GPU-only again");
        assert_eq!(net.state(P), State::Demoted);
    }
    assert!(
        !mock.log()[demoted_at..].contains(&Ev::Protect(P, Prot::NoAccess)),
        "I3: never armed again: {:?}",
        &mock.log()[demoted_at..]
    );
    assert_ne!(mock.prot(P), Prot::NoAccess);
    access(&net, &mock, at(P, 3, 0), None, TITLE_TID);
    access(&net, &mock, at(P, 3, 1), Some(0x33), TITLE_TID);
    let stats = net.stats();
    assert_eq!(stats.arm_noops, 5);
    assert_eq!(stats.syncs, 1);
    assert_eq!(net.drain_demotions().len(), 1);
}

/// t3 (I4, H-02): a write fault in another 4 KiB block keeps both writers.
///
/// Input: page P = 0x12, Net, `arm(P, GpuOnly)`, then the GPU writes
/// X = 0x8e at block 0 byte 0x88 (submission completed).
/// Run: title thread writes Y = 0xb1 at block 1 byte 0 (faults), then the
/// refresh before the next submission.
/// Expected: title memory holds X at block 0 and Y at block 1; P is changed
/// in the existing tracker (dirty) right after the write; after the refresh
/// the copy holds both too (the next GPU read sees Y, I4); one demotion
/// with reason CpuWrite; `stats().syncs == 1`; title bytes the GPU did not
/// write are unchanged (pattern at block 0 byte 0x87, block 1 byte 1).
///
/// This holds the net to the order (sync with the forced copy, then open to
/// the write guard, so the write is dirty). That the engine's forced copy
/// keeps the GPU's bytes is not shown here: the model's copy cannot discard.
///
/// Red against today: the same sequence through today's
/// `Shadow::copy_back_page` (title write lands, page changed, nothing
/// saved -> page kept as the title has it, GPU bytes dropped) is test
/// `a_gpu_write_at_a_computed_address_survives_a_title_write_elsewhere_in_its_page`
/// (`shadow_invariants.rs`, senior 62, fork h02/copyback-test 1fbadba,
/// shown red there), not
/// doubled here. Here the order differs (fault BEFORE the write lands), so
/// the net must sync with `copy_back_forced`, never with the discarding
/// path; and watch's handler must not see the fault first (it would open
/// the page and mark it changed, giving today's discard).
#[test]
fn t3_write_fault_keeps_both() {
    const P: usize = 0x12;
    const X: u8 = 0x8e;
    const Y: u8 = 0xb1;
    let mock = Mock::new(&[P]);
    let net = PageNet::new(Mode::Net, mock.clone());
    net.arm(P, Reason::GpuOnly);
    let (gpu_at, cpu_at) = (at(P, 0, 0x88), at(P, 1, 0));
    mock.gpu_write(gpu_at, X);
    access(&net, &mock, cpu_at, Some(Y), TITLE_TID);

    assert_eq!(
        mock.title(gpu_at),
        X,
        "the GPU's byte (block 0) is in title memory"
    );
    assert_eq!(
        mock.title(cpu_at),
        Y,
        "the title's byte (block 1) is in title memory"
    );
    assert_eq!(
        mock.title(at(P, 0, 0x87)),
        pattern(at(P, 0, 0x87)),
        "nothing else changed"
    );
    assert_eq!(
        mock.title(at(P, 1, 1)),
        pattern(at(P, 1, 1)),
        "nothing else changed"
    );
    assert!(
        mock.changed(P),
        "I4: the first CPU write after the demotion is dirty"
    );
    mock.refresh();
    assert_eq!(
        mock.copy(cpu_at),
        Y,
        "I4: the next GPU read sees the title's write"
    );
    assert_eq!(mock.copy(gpu_at), X, "and still the GPU's");
    assert_eq!(net.stats().syncs, 1);
    assert_eq!(
        net.drain_demotions(),
        vec![Demotion {
            page: P,
            reason: DemoteReason::CpuWrite,
            frame: 1
        }]
    );
}

/// t4 (I5): two threads fault on the same page: one sync, both after it.
///
/// Input: page P = 0x13, Net, `arm(P, GpuOnly)`, GPU wrote X = 0x5a at
/// block 0 byte 0 (completed); the mock's copy gate closed.
/// Run (simulated order): thread A (tid 100) reads block 0 byte 0 and
/// blocks inside `copy_back_forced` (CopyStart(P) logged); then thread B
/// (tid 101) writes Z = 0x77 at block 3 byte 0 and faults on the same page.
/// After 100 ms neither has returned; then the gate opens.
/// Expected: A reads X, B's write landed (title block 3 byte 0 == Z) and is
/// dirty; exactly one CopyStart(P) and one FenceDone in the log; state
/// Demoted; `stats().syncs == 1`; exactly one demotion (reason of the
/// first fault, CpuRead). Then a late fault for P (as a third thread that
/// faulted before the page was opened and is dispatched only now):
/// `on_fault(at(P,5,0), false, 102)` -> Handled, still one copy, still no
/// second demotion.
#[test]
fn t4_concurrent_faults_single_sync() {
    const P: usize = 0x13;
    const X: u8 = 0x5a;
    const Z: u8 = 0x77;
    let mock = Mock::new(&[P]);
    let net = Arc::new(PageNet::new(Mode::Net, mock.clone()));
    net.arm(P, Reason::GpuOnly);
    mock.gpu_write(at(P, 0, 0), X);
    mock.close_gate();

    let (tx_a, rx_a) = mpsc::channel();
    let (n, m) = (net.clone(), mock.clone());
    let a = std::thread::spawn(move || {
        tx_a.send(access(&n, &m, at(P, 0, 0), None, TITLE_TID))
            .unwrap()
    });
    assert!(mock.wait_for(&Ev::CopyStart(P)), "A is syncing");
    assert_eq!(net.state(P), State::Syncing);

    let (tx_b, rx_b) = mpsc::channel();
    let (n, m) = (net.clone(), mock.clone());
    let b = std::thread::spawn(move || {
        tx_b.send(access(&n, &m, at(P, 3, 0), Some(Z), TITLE_TID_2))
            .unwrap()
    });
    assert!(
        rx_b.recv_timeout(Duration::from_millis(100)).is_err(),
        "I5: B waits for A's sync"
    );
    assert!(rx_a.try_recv().is_err(), "A is still in the sync");
    mock.open_gate();
    assert_eq!(
        rx_a.recv_timeout(Duration::from_secs(2))
            .expect("A returns"),
        X
    );
    assert_eq!(
        rx_b.recv_timeout(Duration::from_secs(2))
            .expect("B returns"),
        Z
    );
    a.join().unwrap();
    b.join().unwrap();

    assert_eq!(mock.title(at(P, 3, 0)), Z, "B's write landed");
    assert!(mock.changed(P), "and is dirty");
    assert_eq!(mock.count(&Ev::CopyStart(P)), 1, "I5: one sync");
    assert_eq!(mock.count(&Ev::FenceDone), 1);
    assert_eq!(net.state(P), State::Demoted);
    assert_eq!(net.stats().syncs, 1);
    let demotions = net.drain_demotions();
    assert_eq!(
        demotions,
        vec![Demotion {
            page: P,
            reason: DemoteReason::CpuRead,
            frame: 1
        }]
    );

    assert_eq!(
        net.on_fault(at(P, 5, 0), false, 102),
        Fault::Handled,
        "a late fault of the page is retried"
    );
    assert_eq!(mock.count(&Ev::CopyStart(P)), 1, "no second sync");
    assert!(net.drain_demotions().is_empty(), "no second demotion");
}

/// The t5 sequence on pages 0x20 (read), 0x21 (GPU write, copy-back, read)
/// and 0x22 (write) in `mode`; returns title and copy bytes of all three,
/// the changed set, the sync events (fence and copies) and the net.
#[allow(clippy::type_complexity)]
fn t5_sequence(mode: Mode) -> (Vec<u8>, Vec<u8>, BTreeSet<usize>, Vec<Ev>, Net, Arc<Mock>) {
    let pages = [0x20, 0x21, 0x22];
    let mock = Mock::new(&pages);
    let net = PageNet::new(mode, mock.clone());
    // Pages 0x20 and 0x22 were just copied back: armed for measuring.
    net.arm(0x20, Reason::AfterCopyBack);
    net.arm(0x22, Reason::AfterCopyBack);
    access(&net, &mock, at(0x20, 2, 4), None, TITLE_TID);
    access(&net, &mock, at(0x22, 5, 0), Some(0x55), TITLE_TID);
    // Page 0x21: a submission writes it, the engine copies it back, the
    // page is armed again, then the title reads block 0.
    mock.gpu_write(at(0x21, 9, 0), 0x66);
    mock.engine_copy_back(0x21);
    net.arm(0x21, Reason::AfterCopyBack);
    if mode == Mode::Measure {
        assert_eq!(
            mock.prot(0x21),
            Prot::NoAccess,
            "armed again after the copy-back"
        );
    }
    access(&net, &mock, at(0x21, 0, 0), None, TITLE_TID);
    mock.refresh();
    let s = mock.st();
    let bytes =
        |m: &BTreeMap<usize, Vec<u8>>| pages.iter().flat_map(|p| m[p].clone()).collect::<Vec<u8>>();
    let (title, copy) = (bytes(&s.title), bytes(&s.copy));
    let syncs: Vec<Ev> = s
        .log
        .iter()
        .filter(|e| {
            matches!(
                e,
                Ev::FenceWait | Ev::FenceDone | Ev::CopyStart(_) | Ev::CopyDone(_) | Ev::Refresh(_)
            )
        })
        .cloned()
        .collect();
    let changed = s.changed.clone();
    drop(s);
    (title, copy, changed, syncs, net, mock)
}

/// t5 (I7): measuring is passive.
///
/// Input/run: [`t5_sequence`] once with Off, once with Measure.
/// Expected: identical title bytes, copy bytes, changed sets and sync
/// events (fence, forced copies, refreshes) for both; Off never sets
/// NoAccess and its stats are empty; Measure's stats are exactly reads
/// {(0x20, 2): 1, (0x21, 0): 1} and writes {(0x22, 5): 1}, syncs 0, no
/// demotions; after the measured write page 0x22 is changed in the
/// existing tracker (write tracking active again after opening); page 0x21
/// was armed again after its copy-back (asserted in the sequence); each
/// measured page is no longer NoAccess at the end.
#[test]
fn t5_measure_is_passive() {
    let (title_off, copy_off, changed_off, syncs_off, net_off, mock_off) = t5_sequence(Mode::Off);
    let (title_m, copy_m, changed_m, syncs_m, net_m, mock_m) = t5_sequence(Mode::Measure);
    assert!(title_off == title_m, "I7: measure changes no title byte");
    assert!(copy_off == copy_m, "I7: nor a copy byte");
    assert_eq!(changed_off, changed_m, "I7: nor what the tracker saw");
    assert_eq!(syncs_off, syncs_m, "I7: nor a sync decision");
    assert!(
        changed_m.contains(&0x22) || syncs_m.contains(&Ev::Refresh(0x22)),
        "write tracking active after opening"
    );
    assert!(
        !mock_off
            .log()
            .iter()
            .any(|e| matches!(e, Ev::Protect(_, Prot::NoAccess))),
        "Off arms nothing"
    );
    assert_eq!(net_off.stats(), Stats::default());
    let stats = net_m.stats();
    assert_eq!(
        stats.reads,
        BTreeMap::from([((0x20, 2), 1), ((0x21, 0), 1)])
    );
    assert_eq!(stats.writes, BTreeMap::from([((0x22, 5), 1)]));
    assert_eq!(stats.syncs, 0);
    assert!(net_m.drain_demotions().is_empty());
    for p in [0x20, 0x21, 0x22] {
        assert_ne!(
            mock_m.prot(p),
            Prot::NoAccess,
            "page {p:#x} opened after its fault"
        );
        assert_eq!(net_m.state(p), State::Shared);
    }
    assert_eq!(net_m.oddities(), Oddities::default());
}

/// t6 (I6, I8): foreign addresses and host accesses.
///
/// Input: pages A = 0x30 (armed GpuOnly, Net), B = 0x31 (known to the mock,
/// never armed), C = 0x32 (armed then `disarm_for_free`); a second net in
/// Measure with page D = 0x33 armed, host thread 7 registered.
/// Run/expected:
/// - `on_fault` at 0x1000 (no page of ours), in B, in C: NotOurs, no event
///   logged, states unchanged (A GpuOnlyArmed, B and C Shared), stats
///   unchanged, no demotion (I6). C is no longer NoAccess after the disarm.
/// - Measure: host thread 7 faults on D (read): Handled (it must run on),
///   but `stats()` stays empty (I8). Net mode with a host fault is left to
///   Fable (sync and demote, or open without counting?). Note: today the
///   host reads title memory through `ReadProcessMemory` (`host::read`),
///   which fails on a NoAccess page instead of faulting, so in the engine
///   I8 holds by construction but such reads return false.
#[test]
fn t6_foreign_and_host_access() {
    let mock = Mock::new(&[0x30, 0x31, 0x32]);
    let net = PageNet::new(Mode::Net, mock.clone());
    net.arm(0x30, Reason::GpuOnly);
    net.arm(0x32, Reason::GpuOnly);
    net.disarm_for_free(0x32);
    assert_ne!(mock.prot(0x32), Prot::NoAccess, "disarmed for the free");
    let (log, stats) = (mock.log(), net.stats());
    for (addr, write) in [
        (0x1000, false),
        (at(0x31, 1, 0), true),
        (at(0x32, 0, 0), false),
    ] {
        assert_eq!(
            net.on_fault(addr, write, TITLE_TID),
            Fault::NotOurs,
            "I6: {addr:#x} is not ours"
        );
    }
    assert_eq!(mock.log(), log, "I6: no state change logged");
    assert_eq!(net.stats(), stats);
    assert_eq!(net.state(0x30), State::GpuOnlyArmed);
    assert_eq!(net.state(0x31), State::Shared);
    assert_eq!(net.state(0x32), State::Shared);
    assert!(net.drain_demotions().is_empty());

    let mock = Mock::new(&[0x33]);
    let net = PageNet::new(Mode::Measure, mock.clone());
    net.host_thread(HOST_TID);
    net.arm(0x33, Reason::AfterCopyBack);
    assert_eq!(
        net.on_fault(at(0x33, 0, 0), false, HOST_TID),
        Fault::Handled,
        "the host runs on"
    );
    assert_eq!(
        net.stats(),
        Stats::default(),
        "I8: a host access is no title read"
    );
}

/// t7 (I7): opening a measured page never undoes a guard set after arming.
///
/// Input: Measure; page A = 0x40 written by the title before (open, changed)
/// and armed; page B = 0x41 armed from the guard (read-only).
/// Run: the engine refreshes and takes A again (read-only: the arming is
/// gone, as a `VirtualProtect` of the write guard replaces ours), then the
/// title's write to A reaches the net first (handler order). The engine
/// opens B itself (a kernel write), then a title read of B that faulted
/// before that is dispatched.
/// Expected: A's fault is NotOurs and A is still read-only (the net giving
/// A back its old read-write protection would let the write through
/// untracked: a stale GPU copy); B's fault is Handled (retried, the page is
/// open) and B stays read-write; nothing counted, two overrides noted.
#[test]
fn t7_measure_never_undoes_a_later_guard() {
    let (a, b) = (0x40, 0x41);
    let mock = Mock::new(&[a, b]);
    let net = PageNet::new(Mode::Measure, mock.clone());
    access(&net, &mock, at(a, 0, 0), Some(1), TITLE_TID);
    assert_eq!(mock.prot(a), Prot::ReadWrite);
    net.arm(a, Reason::AfterCopyBack);
    net.arm(b, Reason::AfterCopyBack);
    assert_eq!(
        (mock.prot(a), mock.prot(b)),
        (Prot::NoAccess, Prot::NoAccess)
    );

    mock.refresh();
    mock.engine_copy_back(a);
    assert_eq!(
        net.on_fault(at(a, 0, 1), true, TITLE_TID),
        Fault::NotOurs,
        "the write guard's fault"
    );
    assert_eq!(
        mock.prot(a),
        Prot::ReadOnly,
        "I7: the guard set after arming stands"
    );

    mock.st().prot.insert(b, Prot::ReadWrite);
    assert_eq!(
        net.on_fault(at(b, 0, 0), false, TITLE_TID),
        Fault::Handled,
        "the page is open: retry"
    );
    assert_eq!(mock.prot(b), Prot::ReadWrite);

    assert_eq!(
        net.stats(),
        Stats::default(),
        "nothing measured: both armings were lost"
    );
    assert_eq!(
        net.oddities(),
        Oddities {
            overridden: 2,
            ..Oddities::default()
        }
    );
    assert_eq!((net.state(a), net.state(b)), (State::Shared, State::Shared));
}

/// t8 (I8, I3): in net mode a host thread's fault syncs and demotes like
/// the title's (the host must see the GPU's bytes too, and a page the host
/// reads through the title's view is not GPU-only), but is not counted as a
/// title access. An arm of a page that is closed already (not mapped, or
/// closed by someone else) arms nothing.
#[test]
fn t8_net_host_fault_and_unmapped_arm() {
    const P: usize = 0x50;
    let mock = Mock::new(&[P]);
    let net = PageNet::new(Mode::Net, mock.clone());
    net.host_thread(HOST_TID);
    net.arm(P, Reason::GpuOnly);
    mock.gpu_write(at(P, 1, 0), 0x42);
    assert_eq!(net.on_fault(at(P, 1, 0), false, HOST_TID), Fault::Handled);
    assert_eq!(
        mock.title(at(P, 1, 0)),
        0x42,
        "the host reads the GPU's byte"
    );
    assert_eq!(net.state(P), State::Demoted);
    let stats = net.stats();
    assert!(
        stats.reads.is_empty() && stats.writes.is_empty(),
        "I8: {stats:?}"
    );
    assert_eq!(stats.syncs, 1);
    assert_eq!(net.oddities().host_faults, 1);
    assert_eq!(net.drain_demotions().len(), 1);

    net.arm(0x51, Reason::GpuOnly);
    assert_eq!(net.state(0x51), State::Shared, "not mapped: nothing to arm");
    assert!(!net.is_gpu_only(0x51));
    let mock = Mock::new(&[0x52]);
    mock.st().prot.insert(0x52, Prot::NoAccess);
    let net = PageNet::new(Mode::Net, mock.clone());
    net.arm(0x52, Reason::GpuOnly);
    assert_eq!(
        net.state(0x52),
        State::Shared,
        "closed by someone else: not ours to open"
    );
    assert_eq!(
        net.on_fault(at(0x52, 0, 0), false, TITLE_TID),
        Fault::NotOurs
    );
}

/// t9 (I7, I8): the host's own access opens a measured page uncounted.
///
/// Input: Measure; pages A = 0x60 and B = 0x61 armed after a copy-back.
/// Run: `host_access(A)` (the host is about to copy A in the kernel, which
/// fails on a closed page instead of faulting); then a title read of A and
/// one of B.
/// Expected: `host_access` says A was armed and A is read-only again (what
/// it was armed from: the host's copy then runs as without the net); A's
/// title read faults nowhere and counts nothing; B's is counted; nothing is
/// armed afterwards (`any_armed`), each page was armed once; a second
/// `host_access(A)` and one of a page never armed do nothing.
#[test]
fn t9_host_access_opens_uncounted() {
    let (a, b) = (0x60, 0x61);
    let mock = Mock::new(&[a, b]);
    let net = PageNet::new(Mode::Measure, mock.clone());
    assert!(!net.any_armed());
    net.arm(a, Reason::AfterCopyBack);
    net.arm(b, Reason::AfterCopyBack);
    assert!(net.any_armed());
    assert!(net.host_access(a), "A was armed");
    assert_eq!(mock.prot(a), Prot::ReadOnly, "I7: as it was before arming");
    assert_eq!(net.state(a), State::Shared);
    access(&net, &mock, at(a, 1, 0), None, TITLE_TID);
    assert!(net.any_armed(), "B still is");
    access(&net, &mock, at(b, 2, 0), None, TITLE_TID);
    assert!(!net.any_armed());
    assert!(!net.host_access(a) && !net.host_access(0x62));
    let stats = net.stats();
    assert_eq!(
        stats.reads,
        BTreeMap::from([((b, 2), 1)]),
        "I8: only the title's read"
    );
    assert!(stats.writes.is_empty());
    assert_eq!(
        net.oddities(),
        Oddities {
            host_faults: 1,
            ..Oddities::default()
        }
    );
    assert_eq!(net.armed_counts(), BTreeMap::from([(a, 1), (b, 1)]));
}

/// t10 (measure): an arming lost to someone else's protection is renewed.
///
/// Input: Measure; page P = 0x70 armed; the engine opens it (a kernel
/// write) and takes it again (refresh, copy-back): read-only, no fault
/// came, the net still has it as armed.
/// Run: the next copy-back arms P again; a title read.
/// Expected: P is closed again by that arm (left as it was, no later
/// access of the page would ever be counted); the read is counted; one
/// override noted; P was armed twice and nothing is armed at the end.
#[test]
fn t10_lost_arming_is_renewed() {
    const P: usize = 0x70;
    let mock = Mock::new(&[P]);
    let net = PageNet::new(Mode::Measure, mock.clone());
    net.arm(P, Reason::AfterCopyBack);
    mock.st().prot.insert(P, Prot::ReadWrite);
    mock.engine_copy_back(P);
    assert_eq!(net.state(P), State::ArmedMeasure, "no fault came");
    net.arm(P, Reason::AfterCopyBack);
    assert_eq!(mock.prot(P), Prot::NoAccess, "armed afresh");
    access(&net, &mock, at(P, 4, 0), None, TITLE_TID);
    assert_eq!(net.stats().reads, BTreeMap::from([((P, 4), 1)]));
    assert_eq!(
        net.oddities(),
        Oddities {
            overridden: 1,
            ..Oddities::default()
        }
    );
    assert_eq!(net.armed_counts(), BTreeMap::from([(P, 2)]));
    assert!(!net.any_armed());
}
