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

    /// [`Fixture::views`] with the copy in the GPU's memory (`cards`; none
    /// where that is switched off).
    fn views_in_gpu_memory(pages: usize, pattern: impl Fn(usize) -> u8) -> Option<Fixture> {
        tests::LEAST.with(|l| l.set(Some(0)));
        let f = Fixture::views(pages, pattern);
        tests::LEAST.with(|l| l.set(None));
        f.filter(|f| f.copy().carded())
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
    run_schedule_on(seed, marked, false)
}

/// [`run_schedule`]; with `views`, over a section's views, where title
/// memory is only looked at when the title looks (its writes, and all of
/// it at the end): with the memory model on, pages are GPU-only in
/// between, and what lists read is checked against the model instead.
fn run_schedule_on(seed: u64, marked: bool, views: bool) -> Result<(), String> {
    let Some(f) = (if views { Fixture::views(4, pattern) } else { Fixture::new(4, pattern) }) else {
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
                if views {
                    let (a, b) = (l.from & !3, l.to & !3);
                    let read = f.copy_bytes(a, b);
                    if let Some(at) = (a..b).find(|&at| read[at - a] != model[at]) {
                        return fail(format!(
                            "step {i}: list {k} would read {:#x} at {at:#x}, the model has {:#x}",
                            read[at - a], model[at]
                        ));
                    }
                } else if let Some((at, c, t)) = f.first_difference(l.from & !3, l.to & !3) {
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
                if views {
                    continue;
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

// ---------------------------------------------------------------- (d) ---
//
// The memory model (stage 3: `VICTORY_MEMMODEL=1`) and its safety net
// (`crate::watch::net`, `crate::pagenet`), on real memory: a section of
// physical pages the title has mapped twice, as its GPU heaps are. A page
// is GPU-only once a copy-back has closed it; these tests then let the
// title (or the engine) touch it in every way it can and look at what it
// sees.

use crate::pagenet::{DemoteReason, State};
use crate::watch::net::tests as net;

/// The memory model on for this thread until dropped.
struct ModelOn;

impl ModelOn {
    fn new() -> ModelOn {
        crate::watch::net::TEST_MODEL.set(true);
        ModelOn
    }
}

impl Drop for ModelOn {
    fn drop(&mut self) {
        crate::watch::net::TEST_MODEL.set(false);
    }
}

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

    /// One submission whose shaders write `gpu` through marked views of the
    /// whole fixture, to its sync point.
    fn marked(&self, gpu: &[(usize, u8)]) {
        submission(self, Back::Marked, 0, self.len, gpu, &[]);
    }

    /// A fixture over views with the model on and every page GPU-only: one
    /// marked submission wrote byte 8 of each page (0xa0 + page), the
    /// copy-back closed them at both addresses.
    fn gpu_only(pages: usize) -> Option<(ModelOn, Fixture)> {
        let on = ModelOn::new();
        let f = Fixture::views(pages, pattern)?;
        let gpu: Vec<(usize, u8)> = (0..pages).map(|k| (k * PAGE + 8, 0xa0 + k as u8)).collect();
        f.marked(&gpu);
        for k in 0..pages {
            assert_eq!(net::state(f.p / PAGE + k), State::GpuOnlyArmed, "page {k} closed by its copy-back");
            assert_eq!(net::state(f.second() / PAGE + k), State::GpuOnlyArmed, "and where it shows too");
            assert_eq!((protection(f.p + k * PAGE), protection(f.second() + k * PAGE)), (CLOSED, CLOSED));
        }
        Some((on, f))
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

/// (d) A page only the GPU uses stays in its copy: after the copy-back that
/// closed it, what shaders write is not copied back (title memory keeps its
/// older bytes, nobody reads them). The first title read, at either
/// address, sees the GPU's newest bytes all the same, and from then on the
/// page is shared for good: later GPU writes go back at each sync point, a
/// title write next to them stays, and it is never closed again.
#[test]
fn a_title_read_of_a_gpu_only_page_sees_the_gpus_bytes_and_demotes_it() {
    let Some((_on, f)) = Fixture::gpu_only(2) else {
        return; // no host GPU here
    };
    let peek = f.peek();
    let skipped = net::skipped();
    f.marked(&[(16, 0xb0), (PAGE + 16, 0xb1)]);
    assert_eq!(net::skipped(), skipped + 2, "both copy-backs skipped");
    assert_eq!((read(peek + 16), read(peek + PAGE + 16)), (pattern(16), pattern(PAGE + 16)), "not in title memory");
    assert_eq!((protection(f.p), protection(f.p + PAGE)), (CLOSED, CLOSED), "still closed");

    // The title reads page 0 where it shows a second time.
    assert_eq!(read(f.second() + 16), 0xb0, "the GPU's newest byte");
    assert_eq!(read(f.second() + 8), 0xa0);
    assert_eq!(read(f.p + 17), pattern(17));
    assert_eq!((protection(f.p), protection(f.second())), (GUARDED, GUARDED), "open together, under the write guard");
    assert_eq!((net::state(f.p / PAGE), net::state(f.second() / PAGE)), (State::Demoted, State::Demoted));
    assert_eq!(net::demotion(f.second() / PAGE), Some((DemoteReason::CpuRead, 2)), "logged with its reason");
    assert_eq!(net::state(f.p / PAGE + 1), State::GpuOnlyArmed, "page 1 is untouched");
    assert_eq!(protection(f.p + PAGE), CLOSED);

    // Shared from now on: both writers' bytes, at every sync point.
    for round in 0..20u8 {
        f.title_write(20, 0x55 ^ round);
        f.marked(&[(24, 0xc0 + round)]);
        assert_eq!((f.title(24), f.title(20)), (0xc0 + round, 0x55 ^ round), "round {round}");
        assert_eq!(net::state(f.p / PAGE), State::Demoted, "never GPU-only again");
        assert!(protection(f.p) != CLOSED && protection(f.second()) != CLOSED);
    }
    f.copy().update(f.p, PAGE);
    assert_eq!(f.first_difference(0, PAGE), None);
    assert_eq!(net::state(f.p / PAGE + 1), State::GpuOnlyArmed);
    assert_eq!(read(f.p + PAGE + 16), 0xb1, "page 1, at its first read");
}

/// (d) A title write to a GPU-only page keeps both writers' bytes: the
/// GPU's go into title memory before the write lands, the write is then the
/// write guard's (the page counts as changed), and the next refresh brings
/// the title's byte to the copy before any GPU read.
#[test]
fn a_title_write_to_a_gpu_only_page_demotes_it_and_is_synced() {
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    f.marked(&[(4096 + 16, 0xb0)]);
    f.title_write(4096 + 20, 0x42);
    assert_eq!(net::state(f.p / PAGE), State::Demoted);
    assert_eq!(net::demotion(f.p / PAGE), Some((DemoteReason::CpuWrite, 2)));
    assert_eq!((f.title(4096 + 16), f.title(4096 + 20), f.title(8)), (0xb0, 0x42, 0xa0), "both writers' bytes");
    f.copy().update(f.p, f.len);
    assert_eq!(f.copy_bytes(4096 + 16, 4096 + 24), [0xb0, pattern(4096 + 17), pattern(4096 + 18), pattern(4096 + 19), 0x42, pattern(4096 + 21), pattern(4096 + 22), pattern(4096 + 23)]);
    assert_eq!(f.first_difference(0, f.len), None, "synced before the next GPU read");
}

/// (d) Two title threads reading a GPU-only page at once: one sync, both
/// see the GPU's bytes, one demotion.
#[test]
fn two_threads_reading_a_gpu_only_page_sync_it_once() {
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    f.marked(&[(16, 0xb0)]);
    let syncs = net::syncs();
    let (a, b) = (f.p + 16, f.second() + 16);
    let seen: Vec<u8> = std::thread::scope(|scope| {
        let threads: Vec<_> = [a, b, a, b]
            .into_iter()
            .map(|at| {
                scope.spawn(move || {
                    crate::watch::net::TEST_MODEL.set(true);
                    read(at)
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    });
    assert_eq!(seen, [0xb0; 4]);
    assert_eq!(net::demotion(f.p / PAGE).or(net::demotion(f.second() / PAGE)).map(|d| d.1), Some(2));
    assert_eq!(net::demotions_of(&[f.p / PAGE, f.second() / PAGE]), 1, "one sync for the page");
    assert!(net::syncs() > syncs);
}

/// (d) The alias rule. A page another host copy holds, here at the page's
/// other address, is not closed: that copy gets the GPU's bytes only by way
/// of title memory, and keeps getting them. A copy that comes to a closed
/// page later reads title memory first, which syncs the page and takes it
/// out of the class.
#[test]
fn a_page_another_copy_holds_is_never_gpu_only() {
    let _on = ModelOn::new();
    let Some(f) = Fixture::views(2, pattern) else {
        return; // no host GPU here
    };
    let second = f.second();
    // Page 1 gets a second copy before anything is closed.
    register(second + PAGE, (second + PAGE) as u64, PAGE as u64);
    let early = find((second + PAGE) as u64, PAGE as u64).expect("a host GPU");
    early.update(second + PAGE, PAGE);
    let refused = net::refused(crate::watch::net::Refusal::Holders);
    f.marked(&[(8, 0xa0), (PAGE + 8, 0xa1)]);
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed, "page 0: one holder");
    assert_eq!(net::state(f.p / PAGE + 1), State::Shared, "page 1: two copies");
    assert!(protection(f.p + PAGE) != CLOSED && protection(second + PAGE) != CLOSED);
    assert!(net::refused(crate::watch::net::Refusal::Holders) > refused);
    for round in 0..3u8 {
        f.marked(&[(PAGE + 16, 0xb0 + round)]);
        early.update(second + PAGE, PAGE);
        assert_eq!(early.read_host((second + PAGE + 16) as u64, 1)[0] as u8, 0xb0 + round, "the other copy follows");
        assert_eq!(net::state(f.p / PAGE + 1), State::Shared);
    }

    // Page 0 gets a second copy while it is closed.
    f.marked(&[(16, 0xc0)]);
    register(second, second as u64, PAGE as u64);
    let late = find(second as u64, PAGE as u64).expect("a host GPU");
    late.update(second, PAGE);
    assert_eq!(late.read_host((second + 16) as u64, 1)[0] as u8, 0xc0, "it reads the GPU's newest bytes");
    assert_eq!(net::state(f.p / PAGE), State::Demoted);
    assert_eq!(net::demotion(second / PAGE).or(net::demotion(f.p / PAGE)).map(|d| d.0), Some(DemoteReason::Host));
    f.marked(&[(24, 0xd0)]);
    late.update(second, PAGE);
    assert_eq!(late.read_host((second + 24) as u64, 1)[0] as u8, 0xd0, "and follows from then on");
    drop((early, late));
    unregister(second + PAGE, (second + PAGE) as u64);
    unregister(second, second as u64);
}

/// (d) A page is only closed while title memory and the copy are equal: one
/// its tracker has as changed (the title wrote it since the copy took it,
/// or the copy is to take it again) is refused, and closed once taken.
#[test]
fn a_page_changed_since_it_was_taken_is_not_closed() {
    use crate::watch::net::{Refusal, Way, close};
    let _on = ModelOn::new();
    let Some(f) = Fixture::views(1, pattern) else {
        return; // no host GPU here
    };
    let copy = f.copy();
    copy.update(f.p, f.len);
    crate::watch::retake(f.p, PAGE);
    let refused = net::refused(Refusal::Changed);
    close(f.p / PAGE, Way::Marked, copy);
    assert_eq!(net::state(f.p / PAGE), State::Shared);
    assert!(protection(f.p) != CLOSED);
    assert!(net::refused(Refusal::Changed) > refused);
    copy.update(f.p, f.len);
    close(f.p / PAGE, Way::Marked, copy);
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed);
    assert_eq!((protection(f.p), protection(f.second())), (CLOSED, CLOSED));
    assert_eq!(read(f.p + 8), pattern(8));
}

/// (d) The engine's own accesses to title memory sync a GPU-only page
/// first: a kernel copy out of it reads the GPU's bytes, and a kernel write
/// the runtime announces (a file read into the buffer, a copy step) lands
/// on top of them.
#[test]
fn the_engines_own_accesses_sync_a_gpu_only_page() {
    let Some((_on, f)) = Fixture::gpu_only(2) else {
        return; // no host GPU here
    };
    f.marked(&[(16, 0xb0), (PAGE + 16, 0xb1)]);
    let mut bytes = [0u8; 4];
    assert!(crate::host::read_into((f.p + 16) as *const u8, &mut bytes), "readable for the engine");
    assert_eq!(bytes[0], 0xb0);
    assert_eq!(net::demotion(f.p / PAGE), Some((DemoteReason::Host, 2)));

    {
        let _writing = crate::watch::writing(f.second() + PAGE + 20, 4);
        assert_eq!(net::state(f.p / PAGE + 1), State::Demoted, "synced before it opens");
        // SAFETY: open for the write under way.
        unsafe { ((f.second() + PAGE + 20) as *mut u8).write_volatile(0x42) };
    }
    assert_eq!((f.title(PAGE + 16), f.title(PAGE + 20), f.title(PAGE + 8)), (0xb1, 0x42, 0xa1));
    assert_eq!(net::demotion(f.second() / PAGE + 1), Some((DemoteReason::Host, 2)));
}

/// (d) The model closes pages of one section only, named by its size
/// (`VICTORY_MEMMODEL_SECTION`): a section of another size, no section
/// named, and memory that is no section's view are left as they are.
#[test]
fn only_pages_of_the_named_section_are_closed() {
    use crate::watch::net::{Refusal, TEST_SECTION, Way, close};
    let _on = ModelOn::new();
    let Some(f) = Fixture::views(1, pattern) else {
        return; // no host GPU here
    };
    f.copy().update(f.p, f.len);
    let refused = net::refused(Refusal::Section);
    for other in [Some(2 * PAGE), Some(0)] {
        TEST_SECTION.set(other);
        close(f.p / PAGE, Way::Marked, f.copy());
        assert_eq!(net::state(f.p / PAGE), State::Shared, "{other:?}");
        assert!(protection(f.p) != CLOSED);
    }
    assert_eq!(net::refused(Refusal::Section), refused + 1, "another size is counted, none named is not");
    TEST_SECTION.set(Some(f.len));
    close(f.p / PAGE, Way::Marked, f.copy());
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed, "the section named");
    TEST_SECTION.set(None);
    let Some(pool) = Fixture::new(1, pattern) else {
        return;
    };
    pool.copy().update(pool.p, pool.len);
    close(pool.p / PAGE, Way::Marked, pool.copy());
    assert_eq!(net::state(pool.p / PAGE), State::Shared, "no section's view");
}

/// (d) A write the runtime makes through a view of its own of the physical
/// pages (DirectStorage's page destinations) neither faults nor is seen by
/// the guards, and is reported only once written. With the memory model it
/// is told before: a GPU-only page gets the GPU's bytes first, so the write
/// lands on top of them and the report after it loses nothing; and while
/// the write is under way no page of it is closed. Without the model the
/// call does nothing.
#[test]
fn a_write_through_the_runtimes_own_view_is_told_before_it_lands() {
    use crate::watch::net::{Way, close};
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
    assert!(crate::watch::title_memory_will_change(PAGE, 16).is_none(), "nothing without the model");
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    let section = f.section.expect("over views").0;
    // SAFETY: the fixture's section; the runtime's own view of its pages.
    let own = unsafe { MapViewOfFile(section as _, FILE_MAP_WRITE, 0, 0, f.len) }.Value as usize;
    assert!(own != 0);
    f.marked(&[(16, 0xb0)]);
    {
        let _told = crate::watch::title_memory_will_change(f.second(), PAGE).expect("told, with the model");
        assert_eq!(net::demotion(f.second() / PAGE), Some((DemoteReason::Host, 2)), "synced before the write");
        assert_eq!(read(own + 16), 0xb0, "the GPU's bytes are in the page");
        // SAFETY: a byte of the runtime's own view.
        unsafe { ((own + 20) as *mut u8).write_volatile(0x42) };
        drop(crate::watch::writing(f.second(), PAGE)); // the report, once written
    }
    assert_eq!((f.title(16), f.title(20), f.title(8)), (0xb0, 0x42, 0xa0), "both writers' bytes");
    f.copy().update(f.p, f.len);
    assert_eq!(f.first_difference(0, f.len), None, "and the copy takes the write");

    // A page that is not closed when the write is told stays open while it
    // is under way.
    let Some(g) = Fixture::views(1, pattern) else {
        return;
    };
    g.copy().update(g.p, g.len);
    let told = crate::watch::title_memory_will_change(g.second(), PAGE).expect("told");
    g.copy().update(g.p, g.len);
    close(g.p / PAGE, Way::Marked, g.copy());
    assert_eq!(net::state(g.p / PAGE), State::Shared, "not closed under a write");
    assert!(protection(g.p) != CLOSED && protection(g.second()) != CLOSED);
    drop(told);
    g.copy().update(g.p, g.len);
    close(g.p / PAGE, Way::Marked, g.copy());
    assert_eq!(net::state(g.p / PAGE), State::GpuOnlyArmed, "closed once it is done");
}

/// (d) A copy in the GPU's memory has two sides, and only the GPU's holds
/// what shaders wrote to a GPU-only page: a sync brings the page down
/// before it goes into title memory, whatever the CPU's side has.
#[test]
fn a_gpu_only_page_of_a_copy_in_gpu_memory_is_synced_from_the_gpus_side() {
    let _on = ModelOn::new();
    let Some(f) = Fixture::views_in_gpu_memory(1, pattern) else {
        return; // no host GPU here, or no copies in its memory
    };
    f.marked(&[(8, 0xa0)]);
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed);
    // The shaders' next write, on the GPU's side (sent up as before a
    // submission); the CPU's side is left with other bytes, as it is when
    // no copy-back fetches the page any more.
    f.copy().write_host((f.p + 16) as u64, &[0xb0]);
    crate::host::card_up();
    f.copy().write_cpu_side((f.p + 16) as u64, &[0xee]);
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed);
    assert_eq!(read(f.second() + 16), 0xb0, "the GPU's side, not the CPU's");
    assert_eq!(read(f.p + 8), 0xa0);
    assert_eq!(net::state(f.p / PAGE), State::Demoted);
}

/// (d) A page closed right after a copy-back that took the title's bytes
/// around the shaders' (the title wrote the page while the GPU ran): those
/// bytes are on the CPU's side of a copy in the GPU's memory and go up only
/// with the next submission. A sync before that must not bring the page
/// down over them: title memory keeps what the title wrote.
#[test]
fn a_sync_before_the_next_submission_keeps_what_the_cpu_side_is_ahead_by() {
    let _on = ModelOn::new();
    let Some(f) = Fixture::views_in_gpu_memory(1, pattern) else {
        return; // no host GPU here, or no copies in its memory
    };
    let copy = f.copy();
    copy.update(f.p, f.len);
    // The title writes; the refresh takes it to the CPU's side (no
    // submission sends it up here), and the page is closed with title
    // memory and the copy equal.
    f.title_write(40, 0x42);
    copy.update(f.p, f.len);
    crate::watch::net::close(f.p / PAGE, crate::watch::net::Way::Marked, copy);
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed);
    assert_eq!(read(f.second() + 40), 0x42, "the title's own byte survives the sync");
    assert_eq!(net::state(f.p / PAGE), State::Demoted);
}

/// (d) A kernel copy that failed because a page was closed between the
/// engine's look at it and the copy is tried once more, with the page
/// synced; one that failed for any other reason is not.
#[test]
fn a_kernel_copy_that_met_a_page_closed_meanwhile_is_tried_again() {
    use crate::watch::net::touched_again;
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    f.marked(&[(16, 0xb0)]);
    let bytes = [0u8; 4];
    let ours = bytes.as_ptr() as usize;
    assert!(!touched_again(ours, ours, 4), "nothing closed there: not tried again");
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed);
    assert!(touched_again(f.second() + 16, ours, 4), "closed: synced, worth another try");
    assert_eq!(net::demotion(f.second() / PAGE), Some((DemoteReason::Host, 2)));
    assert_eq!(read(f.p + 16), 0xb0);
    assert!(!touched_again(f.second() + 16, ours, 4));
}

/// (d) An address of a GPU-only page going away: the GPU's bytes are in the
/// physical pages before it does (they may be mapped again anywhere), and
/// their other addresses open. Every closed page of the title's view goes
/// with it, also where only a part of the view is reported: our own
/// address of the view is forgotten as a whole.
#[test]
fn unmapping_syncs_gpu_only_pages() {
    let Some((_on, f)) = Fixture::gpu_only(2) else {
        return; // no host GPU here
    };
    let peek = f.peek();
    f.marked(&[(16, 0xb0), (PAGE + 16, 0xb1)]);
    assert_eq!((read(peek + 16), read(peek + PAGE + 16)), (pattern(16), pattern(PAGE + 16)));
    // The runtime reports the unmap of the copy's own address of page 0
    // (before it unmaps; the test leaves the view mapped).
    crate::watch::title_memory_remapped(f.p, PAGE);
    assert_eq!((read(peek + 16), read(peek + 8)), (0xb0, 0xa0), "in the physical page");
    assert_eq!(net::demotion(f.p / PAGE), Some((DemoteReason::Remap, 2)));
    assert!(protection(f.second()) != CLOSED, "the other address is open");
    assert_eq!(read(f.second() + 16), 0xb0);
    assert_eq!(read(peek + PAGE + 16), 0xb1, "page 1 of the forgotten view too");
    assert_eq!(net::state(f.p / PAGE + 1), State::Demoted);
}

/// (d) Shaders' writes to a GPU-only page that were to be dropped
/// (withheld): there is nothing older to go back to, so the page keeps the
/// GPU's bytes and leaves the class.
#[test]
fn withheld_writes_take_a_gpu_only_page_out_of_the_class() {
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    let peek = f.peek();
    f.marked(&[(16, 0xb0)]);
    crate::watch::retake(f.second(), PAGE);
    assert_eq!(net::demotion(f.second() / PAGE), Some((DemoteReason::Withheld, 2)));
    assert_eq!(read(peek + 16), 0xb0);
    assert!(protection(f.p) != CLOSED);
}

/// (d) A new mapping of a GPU-only page is closed with it: a read through
/// it syncs like any other and opens all three addresses.
#[test]
fn a_new_mapping_of_a_gpu_only_page_is_closed_with_it() {
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
    let Some((_on, f)) = Fixture::gpu_only(1) else {
        return; // no host GPU here
    };
    f.marked(&[(16, 0xb0)]);
    let section = f.section.expect("over views").0;
    // SAFETY: the fixture's section, mapped a third time as the title would.
    let third = unsafe { MapViewOfFile(section as _, FILE_MAP_WRITE, 0, 0, PAGE) }.Value as usize;
    assert!(third != 0);
    crate::watch::title_memory_remapped(third, PAGE);
    crate::watch::title_memory_view(third, PAGE, section, 0);
    crate::watch::title_memory_aliased(third, &[f.p, f.second()]);
    assert_eq!(protection(third), CLOSED, "closed with the others");
    assert_eq!(net::state(f.p / PAGE), State::GpuOnlyArmed, "which stay GPU-only");
    assert_eq!(read(third + 16), 0xb0);
    assert_eq!(net::demotion(third / PAGE), Some((DemoteReason::CpuRead, 3)));
    assert!(protection(f.p) != CLOSED && protection(f.second()) != CLOSED && protection(third) != CLOSED);
    crate::watch::title_memory_remapped(third, PAGE);
}

/// (d) A wrong classification costs speed, never correctness: the
/// interleaved schedules of (c) over views with the model on. Every page a
/// copy-back leaves equal to title memory is closed and its later
/// copy-backs are skipped; the title writes anywhere between any two steps,
/// and reads everything at the end. What each list's shaders read and what
/// the title reads at the end must be exactly the byte-wise model's, on
/// both ways back; and pages were in fact closed, skipped and demoted.
#[test]
fn a_wrong_gpu_only_class_costs_speed_not_correctness() {
    let _on = ModelOn::new();
    let (closed, skipped, syncs) = (net::closed(), net::skipped(), net::syncs());
    for marked in [true, false] {
        let failed: Vec<String> = (0..64u64)
            .filter_map(|k| run_schedule_on(0x6d0d_0000 + k * 0x9e37_79b9, marked, true).err())
            .collect();
        assert!(failed.is_empty(), "{} of 64 schedules failed (marked {marked}), first: {}", failed.len(), failed[0]);
    }
    if Fixture::views(1, pattern).is_none() {
        return; // no host GPU here
    }
    assert!(net::closed() > closed, "pages were closed");
    assert!(net::skipped() > skipped, "copy-backs were skipped");
    assert!(net::syncs() > syncs, "and pages were synced and demoted");
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
