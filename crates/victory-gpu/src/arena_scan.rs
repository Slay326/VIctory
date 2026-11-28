//! Which of the builder's input lists the arena scans before a builder
//! submission (`VICTORY_ARENA_SCAN`): the interface the dirty-driven scan
//! hooks into, the four conditions it must keep (as tests), and the
//! counter of missed changes for the check mode.
//!
//! - `0` (default): as before, [`crate::arena`]'s `scan_inputs` reads the
//!   input lists it is due to read before every builder submission.
//! - `1`: not while the scan has nothing to gain ([`idle`]): the arena has
//!   stopped growing, so every page the scan finds is refused.
//! - `2`: check mode: scans as `0` and counts, for the scans `1` leaves
//!   out, the pages they prepared and how many of those got a slot (what
//!   `1` would miss). Not a performance mode.
//!
//! What the scan is for decides this: it reads the builder's input lists
//! only to find structure pages to give slots before the builder runs. No
//! byte of a list reaches the GPU through it (the lists are host copies of
//! their own), so a scan left out can cost a slot given early, never stale
//! bytes; and a stopped arena gives no slots. A scan driven by what the
//! title wrote ([`Dirty`], below) would read fewer bytes for the same
//! nothing while the arena is stopped, and while it grows the full scan is
//! what fills it: the dirty-driven scan is not written (Fable, call 17).
//! Its interface, model and tests stay as the frame for one, should the
//! scan ever feed bytes to the GPU.
//!
//! Scanning only what changed would be safe only under four conditions
//! (each a test below, [`conditions`]):
//! 1. the write watch is read and reset before the pages are read, so a
//!    write while they are read is seen in the next frame;
//! 2. writers the write watch does not see (the GPU, copy-back) mark what
//!    they wrote as changed themselves ([`Dirty::written_unwatched`]);
//! 3. pages not read because the frame's budget ran out stay changed;
//! 4. every uncertainty is a full scan: the watch failing, an input list
//!    not seen before (new or moved), the arena starting over
//!    ([`Dirty::forget`]), the first frames.
//!
//! Nothing here decides what changed: that is [`Dirty`]'s implementation
//! (not written). The tests run against [`FullScan`] (keeps all four,
//! reads everything) and a deliberately naive dirty-only scan that breaks
//! each of them.

// Only [`mode`], [`idle`] and [`scans`] are used outside the tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::OnceLock;

/// `VICTORY_ARENA_SCAN`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Full,
    Useful,
    Check,
}

/// Whether a scan before a builder submission has nothing to gain: the
/// arena has stopped growing (it stays stopped; each page found is
/// refused), and what the lists point at is not kept for
/// `VICTORY_BVH_ARENA_REFRESH_READS` (`pointers_kept`).
pub(crate) fn idle(stopped: bool, pointers_kept: bool) -> bool {
    stopped && !pointers_kept
}

/// Whether the scan is made: always, but for [`Mode::Useful`] when it has
/// nothing to gain (`idle`).
pub(crate) fn scans(mode: Mode, idle: bool) -> bool {
    !(mode == Mode::Useful && idle)
}

/// `VICTORY_ARENA_SCAN`: 0 (or unset or anything else) full, 1 only when
/// useful, 2 check.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("VICTORY_ARENA_SCAN").ok().as_deref()))
}

fn parse_mode(value: Option<&str>) -> Mode {
    match value.map(str::trim) {
        Some("1") => Mode::Useful,
        Some("2") => Mode::Check,
        _ => Mode::Full,
    }
}

/// The unit the write watch reports and [`Missed`] compares.
pub(crate) const PIECE: u64 = 4096;

/// Title memory as the scan sees it (the real one is `GetWriteWatch` and
/// `ReadProcessMemory`; tests use a model of it).
pub(crate) trait Title {
    /// The [`PIECE`]s (addresses) written in `[start, start + len)` since
    /// the watch was last reset there; with `reset`, resets it (as
    /// `GetWriteWatch` with `WRITE_WATCH_FLAG_RESET`, in one call). `None`:
    /// the watch could not be read.
    fn written(&mut self, start: u64, len: u64, reset: bool) -> Option<Vec<u64>>;
    /// Reads `[at, at + dst.len())`.
    fn read(&mut self, at: u64, dst: &mut [u8]);
}

/// Which pieces of the input lists a frame's scan reads: the dirty-driven
/// scan implements it. A frame is [`Dirty::plan`], the reads in its order
/// (until the budget runs out, at least one), [`Dirty::scanned`] after each
/// read, then [`Dirty::end_frame`] ([`frame`] drives one).
pub(crate) trait Dirty {
    /// Before the reads: the ranges (start, len) to read this frame, in
    /// order, of the input lists `inputs` (start, len).
    fn plan(&mut self, title: &mut dyn Title, inputs: &[(u64, u64)]) -> Vec<(u64, u64)>;
    /// The range `piece`, one of [`Dirty::plan`]'s, was read whole.
    fn scanned(&mut self, piece: (u64, u64));
    /// After the frame's reads (those not reported by [`Dirty::scanned`]
    /// were not made).
    fn end_frame(&mut self, title: &mut dyn Title);
    /// A writer the watch does not see (GPU, copy-back) wrote
    /// `[at, at + len)` of title memory (any memory, input list or not).
    fn written_unwatched(&mut self, at: u64, len: u64);
    /// Everything known is uncertain (the arena started over).
    fn forget(&mut self);
}

/// The reference: reads every input list, in turn from where the last
/// frame's budget ran out. Keeps the four conditions without the watch.
#[derive(Default)]
pub(crate) struct FullScan {
    /// The input list and offset where the next frame starts.
    next: Option<(u64, u64)>,
}

impl Dirty for FullScan {
    fn plan(&mut self, _: &mut dyn Title, inputs: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut pieces: Vec<(u64, u64)> = inputs
            .iter()
            .flat_map(|&(start, len)| (0..len.div_ceil(PIECE)).map(move |k| (start + k * PIECE, PIECE.min(len - k * PIECE))))
            .collect();
        if let Some((start, offset)) = self.next
            && let Some(k) = pieces.iter().position(|&(at, _)| at == start + offset)
        {
            pieces.rotate_left(k);
        }
        pieces
    }

    fn scanned(&mut self, (at, len): (u64, u64)) {
        // Next frame starts after it (an address, not an index: lists come
        // and go).
        self.next = Some((at + len, 0));
    }

    fn end_frame(&mut self, _: &mut dyn Title) {}

    fn written_unwatched(&mut self, _: u64, _: u64) {}

    fn forget(&mut self) {
        self.next = None;
    }
}

/// What a frame read: the ranges, in order, and their bytes.
pub(crate) struct Read {
    pub(crate) pieces: Vec<(u64, u64)>,
    pub(crate) bytes: u64,
}

/// One frame of the scan: [`Dirty::plan`], the reads in its order while
/// `budget` bytes last (the first always), [`Dirty::scanned`] after each,
/// [`Dirty::end_frame`]. `each` sees each range's bytes (the pointer search).
pub(crate) fn frame(dirty: &mut dyn Dirty, title: &mut dyn Title, inputs: &[(u64, u64)], budget: u64, mut each: impl FnMut(u64, &[u8])) -> Read {
    let plan = dirty.plan(title, inputs);
    let mut read = Read { pieces: Vec::new(), bytes: 0 };
    let mut buffer = Vec::new();
    for piece in plan {
        if !read.pieces.is_empty() && read.bytes + piece.1 > budget {
            break;
        }
        buffer.resize(piece.1 as usize, 0);
        title.read(piece.0, &mut buffer);
        each(piece.0, &buffer);
        dirty.scanned(piece);
        read.pieces.push(piece);
        read.bytes += piece.1;
    }
    dirty.end_frame(title);
    read
}

/// The check mode's counter (`VICTORY_ARENA_SCAN=2`): after the full scan
/// read a range, which of its [`PIECE`]s changed since the full scan last
/// read them, and of those, which the dirty scan did not read this frame
/// (missed changes). Pieces read for the first time are not compared.
#[derive(Default)]
pub(crate) struct Missed {
    /// Each piece's hash when the full scan last read it.
    last: HashMap<u64, u64>,
    /// Since the last [`Missed::take`]: pieces compared, changed, changed
    /// and missed, and bytes the dirty scan read; pieces read the first time
    /// (new or moved lists: the dirty scan reads them whole too) and bytes
    /// whose watch could not be read (a full scan for the dirty scan too).
    pub(crate) first_seen: u64,
    pub(crate) unwatchable_bytes: u64,
    pub(crate) compared: u64,
    pub(crate) changed: u64,
    pub(crate) missed: u64,
    pub(crate) dirty_bytes: u64,
    /// The first missed pieces (addresses) since the last take, at most 16.
    pub(crate) first_missed: Vec<u64>,
}

impl Missed {
    /// The full scan read `bytes` at `start`; `dirty_read(piece)`: whether
    /// the dirty scan read that piece (address) this frame.
    pub(crate) fn compare(&mut self, start: u64, bytes: &[u8], dirty_read: impl Fn(u64) -> bool) {
        for (k, piece) in bytes.chunks(PIECE as usize).enumerate() {
            let at = start + k as u64 * PIECE;
            let h = hash(piece);
            let Some(old) = self.last.insert(at, h) else {
                self.first_seen += 1;
                continue;
            };
            self.compared += 1;
            if old != h {
                self.changed += 1;
                if !dirty_read(at) {
                    self.missed += 1;
                    if self.first_missed.len() < 16 {
                        self.first_missed.push(at);
                    }
                }
            }
        }
    }

    /// The dirty scan read `bytes` this frame (for its share of the full
    /// scan's bytes).
    pub(crate) fn dirty_read(&mut self, bytes: u64) {
        self.dirty_bytes += bytes;
    }

    /// The watch over `bytes` of an input list could not be read.
    pub(crate) fn unwatchable(&mut self, bytes: u64) {
        self.unwatchable_bytes += bytes;
    }

    /// The counts since the last take, zeroed (for the frame line):
    /// compared, changed, missed, dirty bytes, first missed. (Read
    /// `first_seen` and `unwatchable_bytes` before: zeroed too.)
    pub(crate) fn take(&mut self) -> (u64, u64, u64, u64, Vec<u64>) {
        let counts = (self.compared, self.changed, self.missed, self.dirty_bytes, std::mem::take(&mut self.first_missed));
        (self.compared, self.changed, self.missed, self.dirty_bytes, self.first_seen, self.unwatchable_bytes) = (0, 0, 0, 0, 0, 0);
        counts
    }

    /// Forgets the hashes (input lists gone, the arena started over).
    pub(crate) fn forget(&mut self) {
        self.last.clear();
    }
}

/// A 64-bit hash of a piece's bytes (whether they changed).
fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A model of title memory for the tests: a version per [`PIECE`] (each
/// write raises it), the write watch, and the version each piece had when
/// the scan last read it. A write can be set to happen right after a given
/// piece is read (a title thread writing while the scan reads).
#[derive(Default)]
pub(crate) struct Model {
    version: HashMap<u64, u64>,
    watch: std::collections::BTreeSet<u64>,
    read_at: HashMap<u64, u64>,
    /// Ranges whose watch cannot be read.
    pub(crate) failing: Vec<(u64, u64)>,
    /// After reading this piece, the title writes that one (once).
    pub(crate) write_after_read: Option<(u64, u64)>,
    /// Pieces read since [`Model::take_reads`].
    reads: Vec<u64>,
}

impl Model {
    /// The title writes the piece at `at` (the watch sees it).
    pub(crate) fn title_write(&mut self, at: u64) {
        *self.version.entry(at).or_insert(0) += 1;
        self.watch.insert(at);
    }

    /// The title writes `[start, start + len)` (whole pieces).
    pub(crate) fn title_write_range(&mut self, start: u64, len: u64) {
        for k in 0..len.div_ceil(PIECE) {
            self.title_write(start + k * PIECE);
        }
    }

    /// A writer the watch does not see writes the piece at `at`.
    pub(crate) fn unwatched_write(&mut self, at: u64) {
        *self.version.entry(at).or_insert(0) += 1;
    }

    /// Memory that holds bytes the watch does not know about (written
    /// before it was watched, or mapped anew): changed for the scan.
    pub(crate) fn content_unwatched(&mut self, start: u64, len: u64) {
        for k in 0..len.div_ceil(PIECE) {
            self.unwatched_write(start + k * PIECE);
        }
    }

    /// Pieces of `inputs` whose bytes changed since the scan last read them
    /// (or that it never read).
    pub(crate) fn stale(&self, inputs: &[(u64, u64)]) -> Vec<u64> {
        inputs
            .iter()
            .flat_map(|&(start, len)| (0..len.div_ceil(PIECE)).map(move |k| start + k * PIECE))
            .filter(|at| self.read_at.get(at) != Some(&self.version.get(at).copied().unwrap_or(0)))
            .collect()
    }

    /// Pieces read since the last call.
    pub(crate) fn take_reads(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.reads)
    }
}

impl Title for Model {
    fn written(&mut self, start: u64, len: u64, reset: bool) -> Option<Vec<u64>> {
        if self.failing.iter().any(|&(s, l)| s < start + len && start < s + l) {
            return None;
        }
        let found: Vec<u64> = self.watch.range(start..start + len).copied().collect();
        if reset {
            for at in &found {
                self.watch.remove(at);
            }
        }
        Some(found)
    }

    fn read(&mut self, at: u64, dst: &mut [u8]) {
        for k in 0..(dst.len() as u64).div_ceil(PIECE) {
            let p = at + k * PIECE;
            let v = self.version.get(&p).copied().unwrap_or(0);
            self.read_at.insert(p, v);
            self.reads.push(p);
            let from = (k * PIECE) as usize;
            let to = (from + PIECE as usize).min(dst.len());
            for (i, b) in dst[from..to].iter_mut().enumerate() {
                *b = (v >> (8 * (i % 8))) as u8;
            }
            if let Some((after, write)) = self.write_after_read
                && after == p
            {
                self.write_after_read = None;
                self.title_write(write);
            }
        }
    }
}

/// The four conditions (and the plain case), each a scenario on a
/// [`Model`] with a fresh scan from `make`: a name and Ok or what broke.
/// The dirty-driven scan must pass all of them.
pub(crate) fn conditions(make: &dyn Fn() -> Box<dyn Dirty>) -> Vec<(&'static str, Result<(), String>)> {
    const A: u64 = 0x10_0000_0000;
    const B: u64 = 0x20_0000_0000;
    let unlimited = u64::MAX;
    let pieces = |n: u64| n * PIECE;
    let check = |m: &Model, inputs: &[(u64, u64)], when: &str| -> Result<(), String> {
        let stale = m.stale(inputs);
        if stale.is_empty() { Ok(()) } else { Err(format!("{} pieces not read again {when}, first {:#x}", stale.len(), stale[0])) }
    };
    let mut out = Vec::new();

    // Plain: a title write after a frame's scan is read in the next frame.
    out.push(("plain: a write after the scan is read in the next frame", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(8))];
        m.title_write_range(A, pieces(8));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        m.title_write(A + pieces(3));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        check(&m, &inputs, "in the next frame")
    }));

    // 1: the watch read and reset before the pages are read. A title write
    // to a piece right after the scan read it (same frame) is read in the
    // next frame.
    out.push(("1: watch read and reset before the pages are read", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(8))];
        m.title_write_range(A, pieces(8));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        m.title_write(A + pieces(3));
        m.write_after_read = Some((A + pieces(3), A + pieces(3)));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        check(&m, &inputs, "after a write between reading and the end of the frame")
    }));

    // 2: a writer the watch does not see marks what it wrote itself.
    out.push(("2: GPU/copy-back writes are marked changed by their writer", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(8))];
        m.title_write_range(A, pieces(8));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        m.unwatched_write(A + pieces(5));
        d.written_unwatched(A + pieces(5), PIECE);
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        check(&m, &inputs, "after a GPU write its writer reported")
    }));

    // 3: pieces not read because the budget ran out stay changed: 10
    // changed pieces, 4 a frame, read within 4 frames without more writes.
    out.push(("3: pieces left by the budget stay changed", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(10))];
        m.title_write_range(A, pieces(10));
        for _ in 0..4 {
            frame(d.as_mut(), &mut m, &inputs, pieces(4), |_, _| {});
        }
        check(&m, &inputs, "within 4 frames of 4 pieces")
    }));

    // 4a: the watch failing is a full scan of what it covers.
    out.push(("4a: a failing watch is a full scan", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(8))];
        m.title_write_range(A, pieces(8));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        m.title_write(A + pieces(2));
        m.failing.push((A, pieces(8)));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        check(&m, &inputs, "in the frame the watch failed")
    }));

    // 4b: an input list not seen before (new, or moved to where the watch
    // saw nothing) is read whole, also in the very first frame.
    out.push(("4b: a new or moved input list is read whole", {
        let (mut d, mut m) = (make(), Model::default());
        m.content_unwatched(A, pieces(8));
        let first = [(A, pieces(8))];
        frame(d.as_mut(), &mut m, &first, unlimited, |_, _| {});
        let r = check(&m, &first, "in the first frame");
        m.content_unwatched(B, pieces(6));
        let moved = [(A, pieces(8)), (B, pieces(6))];
        frame(d.as_mut(), &mut m, &moved, unlimited, |_, _| {});
        r.and(check(&m, &moved, "in the frame a list appeared"))
    }));

    // 4c: after the arena starts over every input list is read again whole
    // in the next frame.
    out.push(("4c: after the arena starts over everything is read", {
        let (mut d, mut m) = (make(), Model::default());
        let inputs = [(A, pieces(8))];
        m.title_write_range(A, pieces(8));
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        d.forget();
        m.take_reads();
        frame(d.as_mut(), &mut m, &inputs, unlimited, |_, _| {});
        let reads: std::collections::BTreeSet<u64> = m.take_reads().into_iter().collect();
        let not = (0..8).map(|k| A + pieces(k)).filter(|at| !reads.contains(at)).count();
        if not == 0 { Ok(()) } else { Err(format!("{not} of 8 pieces not read in the frame after the arena started over")) }
    }));
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deliberately naive: reads only what the watch shows as written (a
    /// look without reset), resets the watch over every input list after
    /// the frame, ignores unwatched writers and the arena starting over,
    /// and takes a failing watch as nothing written. Breaks each condition.
    #[derive(Default)]
    pub(crate) struct Naive {
        inputs: Vec<(u64, u64)>,
    }

    impl Dirty for Naive {
        fn plan(&mut self, title: &mut dyn Title, inputs: &[(u64, u64)]) -> Vec<(u64, u64)> {
            self.inputs = inputs.to_vec();
            inputs.iter().flat_map(|&(s, l)| title.written(s, l, false).unwrap_or_default()).map(|at| (at, PIECE)).collect()
        }
        fn scanned(&mut self, _: (u64, u64)) {}
        fn end_frame(&mut self, title: &mut dyn Title) {
            for &(s, l) in &self.inputs {
                title.written(s, l, true);
            }
        }
        fn written_unwatched(&mut self, _: u64, _: u64) {}
        fn forget(&mut self) {}
    }

    fn report(results: &[(&str, Result<(), String>)]) -> String {
        results.iter().map(|(n, r)| format!("{n}: {}\n", r.as_ref().map(|_| "ok".to_owned()).unwrap_or_else(|e| e.clone()))).collect()
    }

    #[test]
    fn the_full_scan_keeps_every_condition() {
        let results = conditions(&|| Box::new(FullScan::default()));
        assert!(results.iter().all(|(_, r)| r.is_ok()), "{}", report(&results));
        assert_eq!(results.len(), 7);
    }

    /// Each condition's test fails for a scan that breaks it (red proof):
    /// the naive scan keeps only the plain case.
    #[test]
    fn a_naive_dirty_scan_breaks_each_condition() {
        let results = conditions(&|| Box::new(Naive::default()));
        let broken: Vec<&str> = results.iter().filter(|(_, r)| r.is_err()).map(|(n, _)| *n).collect();
        assert_eq!(results[0].1, Ok(()), "{}", report(&results));
        assert_eq!(broken.len(), 6, "{}", report(&results));
    }

    /// A dirty-driven scan must keep every condition. None is written
    /// (module text): this runs the naive scan and is red, `--ignored`
    /// shows it.
    #[test]
    #[ignore = "red: no dirty-driven scan is written (see the module text)"]
    fn the_dirty_scan_keeps_every_condition() {
        let results = conditions(&|| Box::new(Naive::default()));
        assert!(results.iter().all(|(_, r)| r.is_ok()), "{}", report(&results));
    }

    #[test]
    fn the_mode_comes_from_the_switch() {
        assert_eq!(parse_mode(None), Mode::Full);
        assert_eq!(parse_mode(Some("0")), Mode::Full);
        assert_eq!(parse_mode(Some("1")), Mode::Useful);
        assert_eq!(parse_mode(Some(" 2 ")), Mode::Check);
        assert_eq!(parse_mode(Some("x")), Mode::Full);
    }

    /// The scan is left out only with the switch at 1, only once the arena
    /// has stopped, and never while the lists' pointers are kept for the
    /// refresh of what builders read.
    #[test]
    fn the_scan_is_left_out_only_when_it_has_nothing_to_gain() {
        for mode in [Mode::Full, Mode::Useful, Mode::Check] {
            for stopped in [false, true] {
                for kept in [false, true] {
                    let left_out = !scans(mode, idle(stopped, kept));
                    assert_eq!(left_out, mode == Mode::Useful && stopped && !kept, "{mode:?} stopped {stopped} kept {kept}");
                }
            }
        }
    }

    /// The counter: a piece seen the first time is not compared; a changed
    /// piece the dirty scan read is changed, not missed; one it did not
    /// read is missed; an unchanged one neither.
    #[test]
    fn the_counter_counts_changes_the_dirty_scan_did_not_read() {
        let mut c = Missed::default();
        let mut bytes = vec![0u8; 4 * PIECE as usize];
        c.compare(0x1000_0000, &bytes, |_| false);
        c.unwatchable(PIECE);
        assert_eq!((c.first_seen, c.unwatchable_bytes), (4, PIECE));
        assert_eq!(c.take(), (0, 0, 0, 0, vec![]));
        assert_eq!((c.first_seen, c.unwatchable_bytes), (0, 0));
        bytes[PIECE as usize] = 1; // piece 1 changes, read by the dirty scan
        bytes[3 * PIECE as usize + 7] = 1; // piece 3 changes, not read
        c.dirty_read(PIECE);
        c.compare(0x1000_0000, &bytes, |at| at == 0x1000_0000 + PIECE);
        assert_eq!(c.take(), (4, 2, 1, PIECE, vec![0x1000_0000 + 3 * PIECE]));
        c.compare(0x1000_0000, &bytes, |_| false);
        assert_eq!(c.take(), (4, 0, 0, 0, vec![]));
        c.forget();
        bytes[0] = 9;
        c.compare(0x1000_0000, &bytes, |_| false);
        assert_eq!(c.take(), (0, 0, 0, 0, vec![]));
    }

    /// With the check mode, the naive scan's missed changes show in the
    /// counter (the measuring tool sees what the condition tests see).
    #[test]
    fn the_counter_sees_the_naive_scans_misses() {
        const A: u64 = 0x10_0000_0000;
        let inputs = [(A, 8 * PIECE)];
        let mut m = Model::default();
        m.title_write_range(A, 8 * PIECE);
        let (mut naive, mut full, mut c) = (Naive::default(), FullScan::default(), Missed::default());
        for frame_no in 0..3 {
            if frame_no == 2 {
                m.unwatched_write(A + 5 * PIECE);
            }
            let dirty = frame(&mut naive, &mut m, &inputs, u64::MAX, |_, _| {});
            let read: std::collections::BTreeSet<u64> = dirty.pieces.iter().map(|p| p.0).collect();
            c.dirty_read(dirty.bytes);
            frame(&mut full, &mut m, &inputs, u64::MAX, |at, bytes| c.compare(at, bytes, |p| read.contains(&p)));
        }
        let (compared, changed, missed, _, first) = c.take();
        assert_eq!((compared, changed, missed), (16, 1, 1));
        assert_eq!(first, vec![A + 5 * PIECE]);
    }
}
