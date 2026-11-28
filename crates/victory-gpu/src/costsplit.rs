//! Diagnostics: the frame's submission time by kind of work
//! (`VICTORY_COST_SPLIT=1`, off by default), logged as per-frame sums with
//! the frame-rate line.
//!
//! The BVH builder on costs a 3070 about +344 ms a frame (GPU +205, preparing
//! +216); this tells how much of it is the builder's chain (its unpacking and
//! builds, and the host work preparing their submissions) and how much the
//! programs reading the built structure (`f607780a` by default,
//! `VICTORY_COST_SPLIT_READERS` names others, comma-separated hash prefixes).
//!
//! - Per submission, grouped by what its lists bound (builder pipelines,
//!   reader pipelines, both, neither): preparing, waiting for the GPU, the
//!   GPU's own time between two timestamps around the submission, and the
//!   copy-back after it.
//! - Within lists, per pipeline class: a timestamp where the bound pipeline's
//!   class changes, so a submission holding both kinds of work is split too.
//! - `crate::prepbd` sums its parts per group while this is on.
//!
//! Off, every hook is a load of a set-once flag and a branch: no clock, no
//! lock, no allocation, no timestamp (the host's hooks check [`enabled`]
//! before anything else).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// A pipeline's kind of work.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    /// One of the title's acceleration-structure builder's
    /// (`crate::pipeline::is_builder`).
    Builder = 0,
    /// A program reading the built structure (by hash).
    Reader = 1,
    Rest = 2,
}
pub(crate) const CLASSES: usize = 3;
const CLASS_NAMES: [&str; CLASSES] = ["builder", "reader", "rest"];

/// A submission's group, by the classes its lists bound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Group {
    Builder = 0,
    Reader = 1,
    Both = 2,
    Rest = 3,
}
pub(crate) const GROUPS: usize = 4;
pub(crate) const GROUP_NAMES: [&str; GROUPS] = ["builder", "reader", "both", "rest"];

/// What is summed per group (microseconds, or a count).
#[derive(Clone, Copy)]
pub(crate) enum Field {
    Submissions = 0,
    /// From taking the submission lock to the lists going to the queue (the
    /// frame line's preparing).
    Prep,
    /// Waiting for the GPU (the frame line's "on the GPU").
    Wait,
    /// Between the timestamps before and after the submission's lists.
    Gpu,
    /// Submissions with both timestamps read.
    Stamped,
    /// Copying back after the lists ran (the frame line's copying back).
    Back,
}
const FIELDS: usize = 6;

static SUM: [[AtomicU64; FIELDS]; GROUPS] = [const { [const { AtomicU64::new(0) }; FIELDS] }; GROUPS];
/// GPU microseconds per class from the timestamps within lists.
static CLASS_US: [AtomicU64; CLASSES] = [const { AtomicU64::new(0) }; CLASSES];
/// The group of the submission being prepared (one at a time, under the
/// submission lock), for `crate::prepbd`.
static CURRENT: AtomicUsize = AtomicUsize::new(Group::Rest as usize);

/// Whether the split is on (`VICTORY_COST_SPLIT=1`; a test sets its own on
/// its thread).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_COST_SPLIT").is_ok_and(|v| v.trim() == "1"))
}

/// The reader hash prefixes (`VICTORY_COST_SPLIT_READERS`, default
/// `f607780a`).
fn readers() -> &'static [String] {
    static READERS: OnceLock<Vec<String>> = OnceLock::new();
    READERS.get_or_init(|| {
        let value = std::env::var("VICTORY_COST_SPLIT_READERS").unwrap_or_default();
        parse_readers(&value)
    })
}

fn parse_readers(value: &str) -> Vec<String> {
    let list: Vec<String> = value
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if list.is_empty() { vec!["f607780a".to_owned()] } else { list }
}

/// The class of a pipeline named `name` (its shader hashes in it).
pub(crate) fn class(builder: bool, name: &str) -> Class {
    class_with(builder, name, readers())
}

fn class_with(builder: bool, name: &str, readers: &[String]) -> Class {
    if builder {
        Class::Builder
    } else if readers.iter().any(|r| name.contains(r.as_str())) {
        Class::Reader
    } else {
        Class::Rest
    }
}

/// The group of a submission whose lists bound the classes in `bound` (a
/// bit per [`Class`]).
pub(crate) fn group(bound: u8) -> Group {
    let builder = bound & (1 << Class::Builder as u8) != 0;
    let reader = bound & (1 << Class::Reader as u8) != 0;
    match (builder, reader) {
        (true, true) => Group::Both,
        (true, false) => Group::Builder,
        (false, true) => Group::Reader,
        (false, false) => Group::Rest,
    }
}

/// Sets the group of the submission being prepared, if the split is on.
pub(crate) fn set_current(group: Group) {
    if enabled() {
        CURRENT.store(group as usize, Relaxed);
    }
}

/// The group of the submission being prepared (`Rest` while off).
pub(crate) fn current() -> usize {
    CURRENT.load(Relaxed)
}

/// Adds `n` to `field` of `group`, if the split is on.
pub(crate) fn add(group: Group, field: Field, n: u64) {
    if enabled() {
        SUM[group as usize][field as usize].fetch_add(n, Relaxed);
    }
}

/// Timestamps per list for the class changes; changes past them are not
/// timed (their time goes to the last timed class).
pub(crate) const SLOTS: u32 = 512;

/// A list's class changes, in recording order: segment `i` runs from
/// timestamp `i` to timestamp `i + 1` (the last one written when the list
/// closes) with class `classes[i]`.
#[derive(Default)]
pub(crate) struct Segments {
    classes: Vec<Class>,
    /// The classes bound (a bit per class).
    bound: u8,
    /// The list was closed with its last timestamp at this slot.
    closed: Option<u32>,
}

impl Segments {
    /// A pipeline of `class` is bound: the slot to write a timestamp to,
    /// where the class changes (nothing while the split is off).
    pub(crate) fn mark(&mut self, class: Class) -> Option<u32> {
        if !enabled() || self.closed.is_some() {
            return None;
        }
        self.bound |= 1 << class as u8;
        if self.classes.last() == Some(&class) || self.classes.len() as u32 >= SLOTS - 1 {
            return None;
        }
        self.classes.push(class);
        Some(self.classes.len() as u32 - 1)
    }

    /// The list closes: the slot of its last timestamp, if it has segments
    /// (timestamps `0..=slot` are then resolved).
    pub(crate) fn close(&mut self) -> Option<u32> {
        if self.classes.is_empty() {
            return None;
        }
        let slot = self.classes.len() as u32;
        self.closed = Some(slot);
        Some(slot)
    }

    /// How many timestamps a closed list resolved.
    pub(crate) fn resolved(&self) -> Option<usize> {
        self.closed.map(|slot| slot as usize + 1)
    }

    /// The classes bound (a bit per [`Class`]).
    pub(crate) fn bound(&self) -> u8 {
        self.bound
    }

    /// Adds the list's GPU time per class from its timestamps (ticks at
    /// `frequency` per second) to the sums.
    pub(crate) fn attribute(&self, stamps: &[u64], frequency: u64) {
        for (class, us) in self.per_class(stamps, frequency).into_iter().enumerate() {
            CLASS_US[class].fetch_add(us, Relaxed);
        }
    }

    /// The list's GPU microseconds per class from its timestamps.
    fn per_class(&self, stamps: &[u64], frequency: u64) -> [u64; CLASSES] {
        let mut us = [0u64; CLASSES];
        for (i, class) in self.classes.iter().enumerate() {
            let (Some(&a), Some(&b)) = (stamps.get(i), stamps.get(i + 1)) else {
                break;
            };
            us[*class as usize] += ticks_us(b.saturating_sub(a), frequency);
        }
        us
    }
}

/// Timestamp ticks at `frequency` per second, in microseconds.
pub(crate) fn ticks_us(ticks: u64, frequency: u64) -> u64 {
    (u128::from(ticks) * 1_000_000 / u128::from(frequency.max(1))) as u64
}

fn take() -> ([[u64; FIELDS]; GROUPS], [u64; CLASSES]) {
    (
        std::array::from_fn(|g| std::array::from_fn(|f| SUM[g][f].swap(0, Relaxed))),
        std::array::from_fn(|c| CLASS_US[c].swap(0, Relaxed)),
    )
}

/// Logs the sums per frame over `frames` frames, if the split is on.
pub(crate) fn log(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let (s, c) = take();
    // The wall time per frame over the same frames (from the last line on).
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let now = std::time::Instant::now();
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).replace(now);
    let wall = last.map_or(0.0, |l| (now - l).as_secs_f64() * 1e3 / frames);
    tracing::info!("{} frame {wall:.1} ms", line(&s, &c, frames));
}

fn line(s: &[[u64; FIELDS]; GROUPS], c: &[u64; CLASSES], frames: f64) -> String {
    let ms = |us: u64| us as f64 / 1e3 / frames;
    let mut text = String::from("cost split per frame:");
    for (g, name) in GROUP_NAMES.iter().enumerate() {
        let f = &s[g];
        text.push_str(&format!(
            " {name}: submissions {:.2}, prep {:.1}, wait {:.1}, gpu {:.1} ({:.2} stamped), back {:.1};",
            f[Field::Submissions as usize] as f64 / frames,
            ms(f[Field::Prep as usize]),
            ms(f[Field::Wait as usize]),
            ms(f[Field::Gpu as usize]),
            f[Field::Stamped as usize] as f64 / frames,
            ms(f[Field::Back as usize]),
        ));
    }
    let gpu: u64 = s.iter().map(|f| f[Field::Gpu as usize]).sum();
    let classes: u64 = c.iter().sum();
    text.push_str(" gpu by pipeline class:");
    for (i, name) in CLASS_NAMES.iter().enumerate() {
        text.push_str(&format!(" {name} {:.1},", ms(c[i])));
    }
    text.push_str(&format!(" outside class segments {:.1};", ms(gpu.saturating_sub(classes))));
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// The split on or off for this thread's test, whatever the
        /// environment says.
        pub(crate) static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    /// The sums are shared by every test thread: the tests that change them
    /// run one at a time.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sums() -> (Vec<u64>, Vec<u64>, usize) {
        (
            SUM.iter().flat_map(|g| g.iter().map(|a| a.load(Relaxed))).collect(),
            CLASS_US.iter().map(|a| a.load(Relaxed)).collect(),
            CURRENT.load(Relaxed),
        )
    }

    #[test]
    fn off_it_records_nothing_stamps_nothing_and_allocates_nothing() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|f| f.set(Some(false)));
        let before = sums();
        let allocations = crate::prepbd::tests::allocations();
        let mut segments = Segments::default();
        for _ in 0..1000 {
            assert_eq!(segments.mark(Class::Builder), None, "off: no timestamp");
            assert_eq!(segments.mark(Class::Reader), None, "off: no timestamp");
            set_current(Group::Builder);
            add(Group::Builder, Field::Submissions, 1);
            log(1.0);
        }
        assert_eq!(segments.close(), None, "off: no closing timestamp, nothing resolved");
        assert_eq!(segments.bound(), 0, "off: no class noted");
        assert_eq!(crate::prepbd::tests::allocations(), allocations, "off: no allocation");
        assert_eq!(sums(), before, "off: nothing recorded");
        FORCED.with(|f| f.set(None));
    }

    #[test]
    fn classes_and_groups() {
        let readers = parse_readers("");
        assert_eq!(readers, vec!["f607780a".to_owned()]);
        assert_eq!(parse_readers(" F607780A ,abcd1234"), vec!["f607780a".to_owned(), "abcd1234".to_owned()]);
        assert_eq!(class_with(true, "[compute f607780a00000000]", &readers), Class::Builder);
        assert_eq!(class_with(false, "[compute f607780a00000000]", &readers), Class::Reader);
        assert_eq!(class_with(false, "[compute 5552dac6a2b7a1fd]", &readers), Class::Rest);
        assert_eq!(group(0), Group::Rest);
        assert_eq!(group(1 << Class::Rest as u8), Group::Rest);
        assert_eq!(group(1 << Class::Builder as u8 | 1 << Class::Rest as u8), Group::Builder);
        assert_eq!(group(1 << Class::Reader as u8), Group::Reader);
        assert_eq!(group(1 << Class::Builder as u8 | 1 << Class::Reader as u8), Group::Both);
    }

    #[test]
    fn on_segments_split_a_list_by_class() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|f| f.set(Some(true)));
        let mut s = Segments::default();
        assert_eq!(s.mark(Class::Rest), Some(0));
        assert_eq!(s.mark(Class::Rest), None, "same class: no new timestamp");
        assert_eq!(s.mark(Class::Builder), Some(1));
        assert_eq!(s.mark(Class::Reader), Some(2));
        assert_eq!(s.mark(Class::Rest), Some(3));
        assert_eq!(s.close(), Some(4));
        assert_eq!(s.resolved(), Some(5));
        assert_eq!(s.mark(Class::Builder), None, "closed: nothing more");
        assert_eq!(s.bound(), 0b111);
        // 1 MHz ticks: rest 10 + 5, builder 100, reader 1000.
        assert_eq!(s.per_class(&[0, 10, 110, 1110, 1115], 1_000_000), [100, 1000, 15]);
        // Missing stamps: only whole segments count.
        assert_eq!(s.per_class(&[0, 10, 110], 1_000_000), [100, 0, 10]);
        FORCED.with(|f| f.set(None));
    }

    #[test]
    fn on_it_sums_per_group_and_takes_them_per_line() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        FORCED.with(|f| f.set(Some(true)));
        let _ = take();
        add(Group::Builder, Field::Submissions, 2);
        add(Group::Builder, Field::Prep, 4000);
        add(Group::Reader, Field::Gpu, 3000);
        add(Group::Reader, Field::Stamped, 1);
        let mut seg = Segments::default();
        seg.mark(Class::Reader);
        seg.close();
        seg.attribute(&[0, 2600], 1_000_000);
        set_current(Group::Both);
        assert_eq!(current(), Group::Both as usize);
        set_current(Group::Rest);
        let (s, c) = take();
        assert_eq!(s[Group::Builder as usize][Field::Submissions as usize], 2);
        let text = line(&s, &c, 2.0);
        assert!(text.contains("builder: submissions 1.00, prep 2.0,"), "{text}");
        assert!(text.contains("reader: submissions 0.00, prep 0.0, wait 0.0, gpu 1.5 (0.50 stamped)"), "{text}");
        assert!(text.contains("gpu by pipeline class: builder 0.0, reader 1.3, rest 0.0, outside class segments 0.2;"), "{text}");
        assert_eq!(take().0, [[0; FIELDS]; GROUPS], "taken: the next line starts from zero");
        FORCED.with(|f| f.set(None));
    }

    #[test]
    fn ticks() {
        assert_eq!(ticks_us(1_000_000_000, 1_000_000_000), 1_000_000);
        assert_eq!(ticks_us(5, 0), 5_000_000, "a zero frequency counts as 1");
    }
}
