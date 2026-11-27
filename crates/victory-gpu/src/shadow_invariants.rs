//! Invariants of the host copies of title memory (`crate::shadow`, with
//! `crate::watch`'s page tracking), tested without game data: what the
//! memory-model rebuild (night 2, stages 3 and 4) must keep.
//!
//! (a) A page the title (the CPU) wrote is in the copy before any GPU read
//!     of it: the refresh at submission takes every page written since the
//!     copy took it (guard fault), also under ranges a list already
//!     declared as written by shaders.
//! (b) A page shaders (the GPU) wrote is in title memory at the sync point
//!     after the submission, on each of the three ways back: declared
//!     ranges (snapshot compare), pages marked through views, and pages
//!     written at computed addresses.
//! (c) Interleaved submissions in a simulated order (two lists in one
//!     submission, then a second submission; title writes between any two
//!     steps, also while the GPU runs): every check point sees exactly what
//!     a byte-wise model says.
//! (d) Demotion (stage 3 safety net): ignored scaffolds with the expected
//!     behaviour, to be enabled with the safety net.
//!
//! The GPU's writes are simulated by writing the host copy (as the tests
//! in `shadow::tests` do); the copies need a host GPU to exist (a buffer),
//! but nothing runs on it except in
//! [`a_title_write_reaches_the_gpu_side_of_a_copy_in_gpu_memory`].

use super::*;
use crate::watch::tests::pool;

/// A title buffer over a fresh write-watched pool of `pages` pages and its
/// copy; unregistered when dropped. The pool is not freed (as in
/// `shadow::tests`): other tests running meanwhile may keep the copy alive
/// (page tables, lists), and memory allocated again at the same address
/// while a tracker holds it is not seen (`crate::watch`'s shortcut).
struct Fixture {
    p: usize,
    len: usize,
    copy: Option<Arc<Shadow>>,
    /// Of a fixture over a section's views ([`Fixture::views`]): the
    /// section, and the title's second view of the same pages.
    section: Option<(usize, usize)>,
}

impl Fixture {
    /// None without a host GPU (no copies).
    fn new(pages: usize, pattern: impl Fn(usize) -> u8) -> Option<Fixture> {
        let p = pool(pages);
        let len = pages * PAGE;
        for at in 0..len {
            // SAFETY: a byte of our pool, not guarded yet.
            unsafe { ((p + at) as *mut u8).write_volatile(pattern(at)) };
        }
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return None;
        };
        Some(Fixture { p, len, copy: Some(copy), section: None })
    }

    /// [`Fixture::new`] over the title's view of a section of physical
    /// pages (we have a view of our own of it, `crate::watch`), which the
    /// title has mapped a second time elsewhere.
    fn views(pages: usize, pattern: impl Fn(usize) -> u8) -> Option<Fixture> {
        let (section, p, second) = crate::watch::tests::reported_section(pages);
        let len = pages * PAGE;
        for at in 0..len {
            // SAFETY: a byte of the view, not guarded yet.
            unsafe { ((p + at) as *mut u8).write_volatile(pattern(at)) };
        }
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            crate::watch::title_memory_section_gone(section);
            return None;
        };
        Some(Fixture { p, len, copy: Some(copy), section: Some((section, second)) })
    }

    fn copy(&self) -> &Arc<Shadow> {
        self.copy.as_ref().expect("live")
    }

    /// The title writes byte `value` at offset `at` (a guarded page faults
    /// into the handler, as a title thread's write).
    fn title_write(&self, at: usize, value: u8) {
        // SAFETY: a byte of our pool.
        unsafe { ((self.p + at) as *mut u8).write_volatile(value) };
    }

    fn title(&self, at: usize) -> u8 {
        // SAFETY: a byte of our pool.
        unsafe { ((self.p + at) as *const u8).read_volatile() }
    }

    /// Title memory `[from, to)` (offsets).
    fn title_bytes(&self, from: usize, to: usize) -> Vec<u8> {
        (from..to).map(|at| self.title(at)).collect()
    }

    /// The copy's bytes `[from, to)` (offsets, dword aligned).
    fn copy_bytes(&self, from: usize, to: usize) -> Vec<u8> {
        assert!(from.is_multiple_of(4) && to.is_multiple_of(4));
        self.copy()
            .read_host((self.p + from) as u64, (to - from) / 4)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    /// The first offset in `[from, to)` where copy and title memory differ.
    fn first_difference(&self, from: usize, to: usize) -> Option<(usize, u8, u8)> {
        let (copy, title) = (self.copy_bytes(from, to), self.title_bytes(from, to));
        (0..to - from).find(|&i| copy[i] != title[i]).map(|i| (from + i, copy[i], title[i]))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        drop(self.copy.take());
        unregister(self.p, self.p as u64);
        if let Some((section, _)) = self.section {
            crate::watch::title_memory_section_gone(section);
        }
    }
}

fn pattern(at: usize) -> u8 {
    (at * 7 + 1) as u8
}

/// A small deterministic generator (xorshift64*), so a failing schedule
/// can be run again by its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---------------------------------------------------------------- (a) ---

/// (a) Every byte the title writes is in the copy after the next refresh
/// (the GPU reads the copy only after it): over rounds of writes to
/// random pages, the whole copy equals title memory after each refresh,
/// and a refresh with no title write since copies nothing.
#[test]
fn a_title_write_is_in_the_copy_before_the_next_gpu_read() {
    let Some(f) = Fixture::new(4, pattern) else {
        return; // no host GPU here
    };
    let copy = f.copy();
    copy.update(f.p, f.len);
    assert_eq!(f.first_difference(0, f.len), None, "the first refresh copies everything");
    assert!(copy.clean(), "taken: guarded and clean");
    let mut rng = Rng(0x0a11_ce5e_ed00_0001);
    for round in 0..32 {
        let writes = 1 + rng.below(6);
        for _ in 0..writes {
            let at = rng.below(f.len);
            f.title_write(at, rng.next() as u8);
        }
        assert!(!copy.clean(), "round {round}: a title write makes the copy dirty");
        assert!(copy.update(f.p, f.len) > 0, "round {round}: the refresh copies the written pages");
        assert_eq!(f.first_difference(0, f.len), None, "round {round}: after the refresh the copy is title memory");
        assert!(copy.clean(), "round {round}: clean again");
        assert_eq!(copy.update(f.p, f.len), 0, "round {round}: nothing written since: nothing copied");
    }
}

/// (a) The same under ranges an earlier list of the submission declared as
/// written by shaders (snapshot and marked alike): a title write there
/// before the submission runs still reaches the copy at the refresh (skipped,
/// a dispatch read stale sizes and looped until the device was removed).
#[test]
fn a_title_write_under_a_declared_range_is_in_the_copy_before_the_gpu_reads() {
    for marked in [false, true] {
        let Some(f) = Fixture::new(2, pattern) else {
            return; // no host GPU here
        };
        let copy = f.copy();
        copy.update(f.p, f.len);
        copy.writing(f.p, f.len, marked);
        f.title_write(PAGE + 20, 0x55);
        f.title_write(7, 0x66);
        assert!(copy.update(f.p + PAGE, PAGE) > 0, "marked {marked}: the written page is copied");
        assert!(copy.update(f.p, PAGE) > 0, "marked {marked}: so is the other");
        assert_eq!(f.first_difference(0, f.len), None, "marked {marked}: the copy is title memory");
        copy.copy_back(f.p, f.len);
        assert_eq!((f.title(PAGE + 20), f.title(7)), (0x55, 0x66), "marked {marked}: no copy-back undoes them");
    }
}

/// (a) A copy in the GPU's memory: a title write reaches the GPU's side
/// with the next submission (the refresh, then the upload), checked on
/// the GPU's side itself.
#[test]
fn a_title_write_reaches_the_gpu_side_of_a_copy_in_gpu_memory() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    let p = pool(2);
    let len = 2 * PAGE;
    register(p, p as u64, len as u64);
    tests::LEAST.with(|l| l.set(Some(0)));
    let copy = find(p as u64, len as u64);
    tests::LEAST.with(|l| l.set(None));
    let copy = copy.expect("a copy");
    if !copy.carded() {
        assert!(!cards(), "the copy is in GPU memory unless that is switched off");
        drop(copy);
        unregister(p, p as u64);
        return;
    }
    let gpu_side = |at: usize| {
        crate::host::card_fetch(&[(&*copy, at & !3, (at & !3) + 4)]);
        copy.read_host((p + (at & !3)) as u64, 1)[0].to_le_bytes()[at % 4]
    };
    // A new list is open; after each submission it is open again.
    let list = host.new_list().unwrap();
    let submit = || {
        copy.refresh(&list, p as u64, len as u64);
        list.close();
        host.execute(&[&list]);
        list.reset();
    };
    submit();
    for (at, value) in [(5, 0x11u8), (PAGE + 9, 0x22), (PAGE - 1, 0x33)] {
        // SAFETY: a byte of our pool (guarded: the handler opens it).
        unsafe { ((p + at) as *mut u8).write_volatile(value) };
        submit();
        assert_eq!(gpu_side(at), value, "the title's byte at {at:#x} is on the GPU's side");
    }
    drop(list);
    drop(copy);
    unregister(p, p as u64);
}

// ---------------------------------------------------------------- (b) ---

/// The ways shaders' writes go back into title memory.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Back {
    /// A range declared without marks: compared with its snapshot.
    Declared,
    /// Pages marked through views ([`Shadow::copy_back_marked`], as
    /// [`copy_back_marks`] calls it for each marked page).
    Marked,
    /// Pages written at addresses shaders computed
    /// ([`Shadow::copy_back_page`], `crate::memory`).
    Computed,
}

/// One submission on `f`: refresh, declare `[from, to)`, the GPU writes
/// `gpu`, then the sync point: the copy-back of `way`.
fn submission(f: &Fixture, way: Back, from: usize, to: usize, gpu: &[(usize, u8)], during: &[(usize, u8)]) {
    let copy = f.copy();
    copy.update(f.p + from, to - from);
    if way != Back::Computed {
        copy.writing(f.p + from, to - from, way == Back::Marked);
    }
    for &(at, value) in gpu {
        copy.write_host((f.p + at) as u64, &[value]);
    }
    for &(at, value) in during {
        f.title_write(at, value);
    }
    let mut pages: Vec<usize> = gpu.iter().map(|&(at, _)| (f.p + at) & !(PAGE - 1)).collect();
    pages.dedup();
    match way {
        Back::Declared => {}
        Back::Marked => pages.iter().for_each(|&page| {
            copy.copy_back_marked(page);
        }),
        Back::Computed => pages.iter().for_each(|&page| {
            copy.copy_back_page(page);
        }),
    }
    if way != Back::Computed {
        copy.copy_back(f.p + from, to - from);
    }
}

/// (b) What shaders wrote is in title memory at the sync point, on each way
/// back, also where the title wrote the same page while the GPU ran (both
/// writers' bytes stay); bytes nobody wrote are left as they were, and the
/// copy then equals title memory.
///
/// Pages written at computed addresses are not armed (no list declares
/// them), so a title write in such a page while the GPU runs keeps the
/// title's bytes and drops the shaders' (`Shadow::copy_back_page`'s
/// documented shortcut, logged once): for that way the title writes another
/// page here.
#[test]
fn a_gpu_write_is_in_title_memory_at_the_sync_point() {
    for way in [Back::Declared, Back::Marked, Back::Computed] {
        let Some(f) = Fixture::new(4, pattern) else {
            return; // no host GPU here
        };
        let gpu = [(10, 0xa1), (PAGE + 100, 0xa2), (PAGE + 101, 0xa3), (2 * PAGE + 4000, 0xa4)];
        let during = match way {
            Back::Computed => [(3 * PAGE + 200, 0xb1), (3 * PAGE + 40, 0xb2)],
            _ => [(PAGE + 200, 0xb1), (40, 0xb2)],
        };
        submission(&f, way, 0, f.len, &gpu, &during);
        for &(at, value) in &gpu {
            assert_eq!(f.title(at), value, "{way:?}: the GPU's byte at {at:#x} is in title memory");
        }
        for &(at, value) in &during {
            assert_eq!(f.title(at), value, "{way:?}: the title's byte at {at:#x} stays");
        }
        for at in [0, 9, 11, PAGE + 99, PAGE + 102, 2 * PAGE + 3999, f.len - 1] {
            assert_eq!(f.title(at), pattern(at), "{way:?}: nothing else changed ({at:#x})");
        }
        f.copy().update(f.p, f.len);
        assert_eq!(f.first_difference(0, f.len), None, "{way:?}: then the copy is title memory");
    }
}

// ---------------------------------------------------------------- (c) ---

/// A step of a simulated schedule.
#[derive(Clone, Debug)]
enum Step {
    /// A title thread writes a byte.
    Title(usize, u8),
    /// A list is prepared: its range refreshed, then declared as written.
    Prepare(usize),
    /// The GPU runs list `k`'s writes.
    Run(usize),
    /// The submission is done: the sync point (marked pages back, then
    /// every declared range), lists `[first, last)`.
    Done(usize, usize),
}

/// A list of a simulated schedule: its declared range and the bytes its
/// shaders write in it.
struct ListPlan {
    from: usize,
    to: usize,
    gpu: Vec<(usize, u8)>,
}

/// One schedule from `seed`: submission 1 with two lists (A, B), then
/// submission 2 with one (C); ranges overlap; every byte is written by at
/// most one writer (the model needs no tie-break: a byte two writers hit
/// within one submission is a race in the title too), but pages are
/// shared by all; title writes go between any two steps.
fn schedule(seed: u64, len: usize) -> (Vec<ListPlan>, Vec<Step>) {
    let mut rng = Rng(seed | 1);
    let mut used = std::collections::HashSet::new();
    let mut fresh = |rng: &mut Rng, from: usize, to: usize| loop {
        let at = from + rng.below(to - from);
        if used.insert(at) {
            return at;
        }
    };
    let value = |rng: &mut Rng, at: usize| {
        let v = rng.next() as u8;
        if v == pattern(at) { v ^ 0x80 } else { v }
    };
    let ranges = [(0, len), (PAGE / 2, len - PAGE / 3), (PAGE + 64, len)];
    let lists: Vec<ListPlan> = ranges
        .iter()
        .map(|&(from, to)| {
            let n = 2 + rng.below(6);
            let gpu = (0..n)
                .map(|_| {
                    let at = fresh(&mut rng, from, to);
                    (at, value(&mut rng, at))
                })
                .collect();
            ListPlan { from, to, gpu }
        })
        .collect();
    let mut steps = vec![
        Step::Prepare(0),
        Step::Prepare(1),
        Step::Run(0),
        Step::Run(1),
        Step::Done(0, 2),
        Step::Prepare(2),
        Step::Run(2),
        Step::Done(2, 3),
    ];
    for _ in 0..4 + rng.below(8) {
        let at = fresh(&mut rng, 0, len);
        let step = Step::Title(at, value(&mut rng, at));
        let slot = rng.below(steps.len() + 1);
        steps.insert(slot, step);
    }
    (lists, steps)
}

/// Runs a schedule on a fresh fixture with every list `marked` or not,
/// checking against a byte-wise model at each step. Err: what went wrong.
fn run_schedule(seed: u64, marked: bool) -> Result<(), String> {
    let Some(f) = Fixture::new(4, pattern) else {
        return Ok(()); // no host GPU here
    };
    let copy = f.copy();
    copy.update(f.p, f.len);
    let (lists, steps) = schedule(seed, f.len);
    // Title memory as it must be: the title's writes at once, the GPU's at
    // their submission's sync point.
    let mut model: Vec<u8> = (0..f.len).map(pattern).collect();
    let fail = |what: String| Err(format!("seed {seed:#x} marked {marked}: {what}; steps {steps:?}"));
    for (i, step) in steps.iter().enumerate() {
        match *step {
            Step::Title(at, value) => {
                f.title_write(at, value);
                model[at] = value;
            }
            Step::Prepare(k) => {
                let l = &lists[k];
                // Marked writes refresh whole pages, as the engine does
                // (`draw`, `bindless`: `Shadow::page_range`), so the pages
                // are taken and guarded for `copy_back_marked`.
                if marked {
                    let (at, n) = copy.page_range((f.p + l.from) as u64, (l.to - l.from) as u64);
                    copy.update(at as usize, n as usize);
                } else {
                    copy.update(f.p + l.from, l.to - l.from);
                }
                // What this list's shaders read: title memory now.
                if let Some((at, c, t)) = f.first_difference(l.from & !3, l.to & !3) {
                    return fail(format!("step {i}: list {k} would read {c:#x} at {at:#x}, title memory has {t:#x}"));
                }
                copy.writing(f.p + l.from, l.to - l.from, marked);
            }
            Step::Run(k) => {
                for &(at, value) in &lists[k].gpu {
                    copy.write_host((f.p + at) as u64, &[value]);
                }
            }
            Step::Done(first, last) => {
                if marked {
                    let mut pages: Vec<usize> = lists[first..last]
                        .iter()
                        .flat_map(|l| l.gpu.iter().map(|&(at, _)| (f.p + at) & !(PAGE - 1)))
                        .collect();
                    pages.sort_unstable();
                    pages.dedup();
                    for page in pages {
                        copy.copy_back_marked(page);
                    }
                }
                for l in &lists[first..last] {
                    copy.copy_back(f.p + l.from, l.to - l.from);
                    for &(at, value) in &l.gpu {
                        model[at] = value;
                    }
                }
                let title = f.title_bytes(0, f.len);
                if let Some(at) = (0..f.len).find(|&at| title[at] != model[at]) {
                    return fail(format!(
                        "step {i}: after the sync point title memory has {:#x} at {at:#x}, the model {:#x}",
                        title[at], model[at]
                    ));
                }
            }
        }
    }
    copy.update(f.p, f.len);
    if let Some((at, c, t)) = f.first_difference(0, f.len) {
        return fail(format!("at the end the copy has {c:#x} at {at:#x}, title memory {t:#x}"));
    }
    let title = f.title_bytes(0, f.len);
    if let Some(at) = (0..f.len).find(|&at| title[at] != model[at]) {
        return fail(format!("at the end title memory has {:#x} at {at:#x}, the model {:#x}", title[at], model[at]));
    }
    Ok(())
}

/// (c) Interleaved submissions in a simulated order, ranges declared
/// without marks (snapshot compare): 64 schedules.
#[test]
fn interleaved_submissions_keep_both_writers_bytes_declared() {
    let failed: Vec<String> = (0..64u64)
        .filter_map(|k| run_schedule(0x5eed_0000 + k * 0x9e37_79b9, false).err())
        .collect();
    assert!(failed.is_empty(), "{} of 64 schedules failed, first: {}", failed.len(), failed[0]);
}

/// (c) The same with every list's pages marked through views (armed pages,
/// saved bytes).
#[test]
fn interleaved_submissions_keep_both_writers_bytes_marked() {
    let failed: Vec<String> = (0..64u64)
        .filter_map(|k| run_schedule(0x3a12_0000 + k * 0x9e37_79b9, true).err())
        .collect();
    assert!(failed.is_empty(), "{} of 64 schedules failed, first: {}", failed.len(), failed[0]);
}

/// (c) The fixed schedule of the incident shape, spelled out: list A
/// declares the whole buffer, the title then writes a constant B reads,
/// B's refresh must bring it, the GPU writes next to a title write in the
/// same page, and the second submission reads both.
#[test]
fn a_constant_written_between_two_lists_reaches_the_second() {
    for marked in [false, true] {
        let Some(f) = Fixture::new(2, pattern) else {
            return; // no host GPU here
        };
        let copy = f.copy();
        copy.update(f.p, f.len);
        // Submission 1, list A.
        copy.update(f.p, f.len);
        copy.writing(f.p, f.len, marked);
        // The title writes a size list B's shaders read.
        f.title_write(PAGE + 16, 0x40);
        // List B.
        copy.update(f.p + PAGE, PAGE);
        copy.writing(f.p + PAGE, PAGE, marked);
        assert_eq!(f.copy_bytes(PAGE + 16, PAGE + 20)[0], 0x40, "marked {marked}: list B reads the title's constant");
        // The GPU runs A and B; the title writes the same page meanwhile.
        copy.write_host((f.p + PAGE + 32) as u64, &[0xc1]);
        copy.write_host((f.p + 8) as u64, &[0xc2]);
        f.title_write(PAGE + 48, 0x77);
        if marked {
            copy.copy_back_marked(f.p);
            copy.copy_back_marked(f.p + PAGE);
        }
        copy.copy_back(f.p, f.len);
        copy.copy_back(f.p + PAGE, PAGE);
        assert_eq!(
            (f.title(PAGE + 16), f.title(PAGE + 32), f.title(8), f.title(PAGE + 48)),
            (0x40, 0xc1, 0xc2, 0x77),
            "marked {marked}: every writer's byte is in title memory"
        );
        // Submission 2 reads all of them.
        copy.update(f.p, f.len);
        assert_eq!(f.first_difference(0, f.len), None, "marked {marked}: submission 2 reads title memory");
    }
}

// ------------------------------------------------- forced copy-back ---

/// The protection of the page at `at`.
fn protection(at: usize) -> u32 {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    info.Protect
}

const CLOSED: u32 = windows_sys::Win32::System::Memory::PAGE_NOACCESS;
const GUARDED: u32 = windows_sys::Win32::System::Memory::PAGE_READONLY;

impl Fixture {
    /// The title's second view of the fixture's pages.
    fn second(&self) -> usize {
        self.section.expect("over views").1
    }

    /// A view of the section nobody knows of: what the physical pages hold,
    /// read without touching the title's views.
    fn peek(&self) -> usize {
        use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile};
        let section = self.section.expect("over views").0;
        // SAFETY: a section the fixture keeps; the view is never unmapped
        // (a test's worth of address space).
        let view = unsafe { MapViewOfFile(section as _, FILE_MAP_READ, 0, 0, self.len) }.Value as usize;
        assert!(view != 0);
        view
    }
}

fn read(at: usize) -> u8 {
    // SAFETY: a byte of a fixture's view; a closed page faults into the
    // handler, which opens it.
    unsafe { (at as *const u8).read_volatile() }
}

/// What a forced copy-back must do, whatever the net does around it: the
/// copy's whole page lands in title memory through our own view while the
/// title's views stay closed, the page's other trackers hear of it, and
/// nothing is ever dropped, also where the ordinary copy-back drops the
/// GPU's bytes (a page its tracker has as changed). Memory with no view of
/// our own, and a page held in part, are refused with nothing written.
#[test]
fn a_forced_copy_back_lands_beside_closed_views_and_drops_nothing() {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let Some(f) = Fixture::views(2, pattern) else {
        return; // no host GPU here
    };
    let copy = f.copy();
    copy.update(f.p, f.len);
    let other = crate::watch::track(f.second(), PAGE, true).expect("views are tracked");
    other.take(f.second(), f.second() + PAGE);
    copy.write_host((f.p + 8) as u64, &[0xaa, 0xbb, 0xcc, 0xdd]);
    copy.write_host((f.p + PAGE - 4) as u64, &[0x11, 0x22, 0x33, 0x44]);
    // The copy's tracker has the page as changed though title memory did
    // not change: the ordinary copy-back keeps title memory's bytes then.
    crate::watch::retake(f.p, PAGE);
    assert_eq!(copy.copy_back_page(f.p), 0, "the ordinary way drops them here");
    let peek = f.peek();
    assert_eq!(read(peek + 8), pattern(8));
    let mut old = 0;
    for view in [f.p, f.second()] {
        // SAFETY: the fixture's views; only their protection changes.
        assert!(unsafe { VirtualProtect(view as *const _, PAGE, CLOSED, &mut old) } != 0);
    }
    assert!(!other.changed(f.second() / PAGE));

    assert!(copy.copy_back_forced(f.p), "landed");
    assert_eq!((protection(f.p), protection(f.second())), (CLOSED, CLOSED), "the title's views stay closed");
    assert_eq!([read(peek + 8), read(peek + 9), read(peek + 10), read(peek + 11)], [0xaa, 0xbb, 0xcc, 0xdd]);
    assert_eq!(read(peek + PAGE - 1), 0x44, "the whole page");
    assert_eq!(read(peek + 100), pattern(100));
    assert_eq!(read(peek + PAGE + 8), pattern(PAGE + 8), "and no other");
    assert!(other.changed(f.second() / PAGE), "the page's other tracker takes it again");

    assert!(!copy.copy_back_forced(f.p + 4096), "not a page");
    assert!(!copy.copy_back_forced(f.p + 2 * PAGE), "not the copy's");
    for view in [f.p, f.second()] {
        // SAFETY: as above.
        unsafe { VirtualProtect(view as *const _, PAGE, GUARDED, &mut old) };
    }
    drop(other);
    let Some(pool) = Fixture::new(1, pattern) else {
        return;
    };
    pool.copy().update(pool.p, pool.len);
    pool.copy().write_host((pool.p + 8) as u64, &[0xee]);
    assert!(!pool.copy().copy_back_forced(pool.p), "no view of our own: refused");
    assert_eq!(pool.title(8), pattern(8), "nothing written");
}

// ---------------------------------------------------------------- (d) ---
//
// Demotion (night 2, stage 3: VICTORY_MEMMODEL=1, the safety net Fable
// implements). Until it exists these are scaffolds: ignored, and they do
// not count as tests. To enable one: replace the `todo!` with the safety
// net's calls, remove `#[ignore]`, show it red with the demotion left out
// and green with it.

/// (d) A page treated as GPU-only (no mirror, no copy-back) that the title
/// writes is detected (guard fault or write watch), synced before the next
/// GPU read, demoted to shared, and the demotion is logged with its reason.
///
/// Expected: after the title write at offset 16 of a GPU-only page, the
/// next refresh brings the title's byte to the copy (as test (a)), the
/// page's class reads `Shared`, and the demotion count is 1 with reason
/// "title write".
#[test]
#[ignore = "stage 3: no GPU-only class or demotion yet (Fable, safety net)"]
fn a_title_write_to_a_gpu_only_page_demotes_it_and_is_synced() {
    let Some(f) = Fixture::new(1, pattern) else {
        return; // no host GPU here
    };
    f.copy().update(f.p, f.len);
    // Stage 3: classify the page GPU-only here (the switch's own call).
    f.title_write(16, 0x42);
    f.copy().update(f.p, f.len);
    assert_eq!(f.copy_bytes(16, 20)[0], 0x42, "synced before the next GPU read");
    todo!("stage 3: assert the page's class is Shared and one demotion (reason: title write) was logged");
}

/// (d) A demoted page is never treated as GPU-only again: not after more
/// frames without title writes, not after the classification runs again.
///
/// Expected: demote a page (as above), run 100 frames (`memory::log_frame`)
/// with only GPU writes to it; it stays `Shared`, its GPU writes go back at
/// each sync point (as test (b)), and the demotion count stays 1.
#[test]
#[ignore = "stage 3: no GPU-only class or demotion yet (Fable, safety net)"]
fn a_demoted_page_is_never_gpu_only_again() {
    todo!("stage 3: demote a page, run frames and a reclassification, assert it stays Shared and copies back");
}

/// (d) A wrong classification costs speed, never correctness: with the
/// whole buffer wrongly classed GPU-only, the interleaved schedules of
/// test (c) still match the model, and every page the title wrote is
/// demoted (count = pages the title wrote).
#[test]
#[ignore = "stage 3: no GPU-only class or demotion yet (Fable, safety net)"]
fn a_wrong_gpu_only_class_costs_speed_not_correctness() {
    todo!("stage 3: class the fixture GPU-only, run run_schedule over 64 seeds, assert Ok and the demotion count");
}

/// Three submissions whose shaders write pages 0 to 2, going back on `way`,
/// with the title reading page 0, writing page 1 and leaving page 2 alone
/// between them (and writing page 3 while the GPU runs); returns title
/// memory and the copy at the end, and what `probe` saw after the first
/// submission.
fn measured_sequence<T>(f: &Fixture, way: Back, probe: impl Fn(&Fixture) -> T) -> (Vec<u8>, Vec<u8>, T) {
    let gpu = |k: u8| [(10, 0xa0 + k), (PAGE + 5000, 0xb0 + k), (2 * PAGE + 9000, 0xc0 + k)];
    submission(f, way, 0, f.len, &gpu(1), &[(3 * PAGE + 7, 0x71)]);
    let seen = probe(f);
    assert_eq!(f.title(10), 0xa1, "the title reads what the GPU wrote");
    f.title_write(PAGE + 5001, 0x51);
    submission(f, way, 0, f.len, &gpu(2), &[]);
    f.title_write(8200, 0x52);
    assert_eq!(f.title(2 * PAGE + 9000), 0xc2);
    submission(f, way, 0, f.len, &gpu(3), &[(3 * PAGE + 8, 0x72)]);
    f.copy().update(f.p, f.len);
    (f.title_bytes(0, f.len), f.copy_bytes(0, f.len), seen)
}

/// Measuring the title's accesses (`VICTORY_PAGENET=measure`) changes no
/// byte of title memory or of the copy over a run of copy-backs on `way`
/// with title reads and writes between them, and it does measure: each page
/// the GPU wrote is closed after its copy-back, with the title's other view
/// of it (`views`: memory of a section, else of a pool), the title's first
/// read and write after it are counted at their 4 KiB blocks, the engine's
/// own accesses (the next copy-back's) are not.
fn measuring_changes_no_byte(way: Back, views: bool) {
    use crate::watch::net::{TEST_ON, tests::seen};
    let fixture = || if views { Fixture::views(4, pattern) } else { Fixture::new(4, pattern) };
    let Some(plain) = fixture() else {
        return; // no host GPU here
    };
    let (title, copy, ()) = measured_sequence(&plain, way, |_| ());
    let f = fixture().expect("as above");
    let net_way = match way {
        Back::Declared => crate::watch::net::Way::Declared,
        Back::Marked => crate::watch::net::Way::Marked,
        Back::Computed => crate::watch::net::Way::Computed,
    };
    let armed_before = crate::watch::net::tests::by_way(net_way).0;
    TEST_ON.set(true);
    let (title_measured, copy_measured, (closed, second)) = measured_sequence(&f, way, |f| {
        let closed = |p: usize| [0, 1, 2, 3].map(|page| seen(p + page * PAGE).1);
        (closed(f.p), f.section.map(|(_, second)| closed(second)))
    });
    TEST_ON.set(false);
    assert_eq!(closed, [true, true, true, false], "closed after their copy-backs; page 3 had none");
    if let Some(second) = second {
        assert_eq!(second, closed, "and the title's other view of the pages with them");
    }
    assert!(
        crate::watch::net::tests::by_way(net_way).0 >= armed_before + 9,
        "three pages armed by each of three copy-backs, counted for their way back"
    );
    assert!(title == title_measured, "no byte of title memory differs");
    assert!(copy == copy_measured, "nor of the copy");
    TEST_ON.set(true);
    assert_eq!(seen(f.p + 10).0, (2, 0), "page 0: the title's read, and this test's of the whole at the end");
    assert_eq!(seen(f.p + PAGE + 5001).0, (0, 1), "its write to page 1");
    assert_eq!(seen(f.p + 8200).0, (0, 1), "and to page 0, after the second copy-back");
    assert_eq!(seen(f.p + 2 * PAGE + 9000).0, (1, 0), "page 2: only the read after the second");
    let summary = crate::watch::net::tests::summary();
    for key in ["armed_computed=", "armed_declared=", "armed_marked=", "not_armed_other_view_open="] {
        assert!(summary.contains(key), "{key} in {summary}");
    }
    if let Some((_, second)) = f.section {
        // Page 2 is closed again (the third copy-back); read through the
        // title's other view, that is where the read is counted.
        // SAFETY: a byte of the second view (closed: the handler opens it).
        let read = unsafe { ((second + 2 * PAGE + 9000) as *const u8).read_volatile() };
        assert_eq!(read, 0xc3, "what the GPU wrote, through the other view");
        assert_eq!(seen(second + 2 * PAGE + 9000).0, (1, 0), "counted where it was read");
    }
    TEST_ON.set(false);
}

/// A copy-back that finds nothing to put back (the shaders wrote the page
/// what it held, or not at all) reads the page all the same, which opens
/// it: a page that was closed is closed again after it, so the title's next
/// read of what the GPU wrote earlier is still counted; a page the GPU
/// never changed is not closed by it.
#[test]
fn a_copy_back_with_nothing_to_put_back_leaves_its_page_closed() {
    use crate::watch::net::{TEST_ON, tests::seen};
    for (way, views) in [(Back::Computed, false), (Back::Marked, false), (Back::Marked, true)] {
        let fixture = if views { Fixture::views(2, pattern) } else { Fixture::new(2, pattern) };
        let Some(f) = fixture else {
            return; // no host GPU here
        };
        TEST_ON.set(true);
        submission(&f, way, 0, f.len, &[(10, 0xa1)], &[]);
        assert_eq!([seen(f.p).1, seen(f.p + PAGE).1], [true, false], "{way:?}: page 0 closed after its copy-back");
        // A submission that changes neither page; both are looked at.
        let copy = f.copy();
        copy.update(f.p, f.len);
        if way == Back::Marked {
            copy.writing(f.p, f.len, true);
        }
        for page in [f.p, f.p + PAGE] {
            let written = match way {
                Back::Marked => copy.copy_back_marked(page),
                _ => copy.copy_back_page(page),
            };
            assert_eq!(written, 0, "{way:?}: nothing to put back");
        }
        if way == Back::Marked {
            copy.copy_back(f.p, f.len);
        }
        assert_eq!([seen(f.p).1, seen(f.p + PAGE).1], [true, false], "{way:?}: still closed; page 1 never was");
        assert_eq!(f.title(10), 0xa1);
        assert_eq!(seen(f.p + 10).0, (1, 0), "{way:?}: the title's read is counted");
        TEST_ON.set(false);
    }
}

/// [`measuring_changes_no_byte`] for pages written at computed addresses.
#[test]
fn measuring_the_titles_accesses_changes_no_byte() {
    measuring_changes_no_byte(Back::Computed, false);
}

/// For declared ranges (compared with their snapshot), in pool memory.
#[test]
fn measuring_changes_no_byte_after_declared_copy_backs() {
    measuring_changes_no_byte(Back::Declared, false);
}

/// For declared ranges in a section's views (written at the title's
/// address, with the title's second view of the pages).
#[test]
fn measuring_changes_no_byte_after_declared_copy_backs_into_views() {
    measuring_changes_no_byte(Back::Declared, true);
}

/// For marked pages of pool memory (written at the title's address).
#[test]
fn measuring_changes_no_byte_after_marked_copy_backs() {
    measuring_changes_no_byte(Back::Marked, false);
}

/// For marked pages of a section's views: written through our own view,
/// the title's guard never opened.
#[test]
fn measuring_changes_no_byte_after_marked_copy_backs_beside_the_guard() {
    measuring_changes_no_byte(Back::Marked, true);
}
