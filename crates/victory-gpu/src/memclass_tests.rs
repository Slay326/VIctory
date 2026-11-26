//! Tests of the stage 1 classification (`crate::memclass`), without game
//! data: addresses are only numbers to the classifier.

use super::*;

const K64: usize = 64 << 10;
const K4: usize = 4 << 10;

fn config(shift: u32) -> Config {
    Config { shift, stream: None, stream_frames: 0, interval_s: 3600, table_s: 3600 }
}

fn classifier(shift: u32) -> Classifier {
    Classifier::new(config(shift))
}

fn frame(c: &Classifier) {
    c.frame_done(std::time::Instant::now());
}

fn units(c: &Classifier) -> Vec<((u32, u64), Unit)> {
    let st = c.state.lock().unwrap();
    let mut v: Vec<_> = st.units.iter().map(|(k, u)| (*k, *u)).collect();
    v.sort_by_key(|(k, _)| *k);
    v
}

fn stats(c: &Classifier, region: u32) -> Stats {
    c.state.lock().unwrap().stats.get(&region).cloned().unwrap_or_default()
}

fn bit(flag: u16) -> usize {
    flag.trailing_zeros() as usize
}

/// Hooks never make the classifier: before `init` (and off) they do
/// nothing, read no environment and allocate nothing (the guard handler is
/// one of them). No test calls `init`.
#[test]
fn hooks_before_init_make_nothing() {
    tag(f::CPU_W_TITLE, 0x1000, 1);
    tag_units(f::GPU_R_TABLE, 0x1000, 1);
    record(f::GPU_R_TABLE, 0x1000, 1);
    sync(Sync::Up, 0x1000, 1, 1);
    refresh_copied(1);
    point(Point::GpuDone);
    drop(submission(1));
    frame_done();
    allocated(0x1000, 1, true);
    assert!(!on() && !first_in_frame(1));
    assert!(GLOBAL.get().is_none(), "a hook made the classifier");
    assert!(!ON.load(Relaxed));
}

/// Off unless `VICTORY_MEMCLASS` is exactly 1; the options parse.
#[test]
fn it_is_off_unless_asked_for() {
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    };
    assert_eq!(Config::from(env(&[])), None);
    assert_eq!(Config::from(env(&[("VICTORY_MEMCLASS", "0")])), None);
    assert_eq!(Config::from(env(&[("VICTORY_MEMCLASS", "")])), None);
    assert_eq!(Config::from(env(&[("VICTORY_MEMCLASS", "yes")])), None);
    let on = Config::from(env(&[("VICTORY_MEMCLASS", "1")])).expect("on");
    assert_eq!((on.shift, on.stream.clone(), on.stream_frames), (16, None, 0));
    let fine = Config::from(env(&[
        ("VICTORY_MEMCLASS", "1"),
        ("VICTORY_MEMCLASS_GRAN", "4k"),
        ("VICTORY_MEMCLASS_STREAM", "x.vmcs"),
        ("VICTORY_MEMCLASS_STREAM_FRAMES", "7"),
    ]))
    .expect("on");
    assert_eq!((fine.shift, fine.stream_frames), (12, 7));
    assert_eq!(fine.stream.as_deref(), Some(std::path::Path::new("x.vmcs")));
}

/// A unit's flags are the frame's: a frame without accesses adds nothing,
/// and the next access counts in a frame of its own.
#[test]
fn flags_are_taken_at_each_frame() {
    let c = classifier(16);
    let a = 0x1_2340_0000;
    c.allocated(a, 4 * K64, true);
    c.tag(f::GPU_R_REFRESH, a + 10, 1);
    c.tag(f::GPU_R_REFRESH, a + 20, 1);
    frame(&c);
    frame(&c);
    let u = units(&c);
    assert_eq!(u.len(), 1);
    assert_eq!((u[0].1.frames, u[0].1.first, u[0].1.last), (1, 1, 1));
    assert_eq!(stats(&c, 0).frames_with[bit(f::GPU_R_REFRESH)], 1);
    c.tag(f::CPU_W_TITLE, a + 30, 1);
    frame(&c);
    let u = units(&c);
    assert_eq!((u[0].1.ever, u[0].1.frames, u[0].1.last), (f::GPU_R_REFRESH | f::CPU_W_TITLE, 2, 3));
    let s = stats(&c, 0);
    assert_eq!((s.frames_with[bit(f::GPU_R_REFRESH)], s.frames_with[bit(f::CPU_W_TITLE)]), (1, 1));
    assert_eq!(s.hits[bit(f::GPU_R_REFRESH)], 1);
}

/// The title's writes and the runtime's are apart: under the proposed rule
/// memory only the runtime and the GPU touch is GPU-only, memory the title
/// writes is shared.
#[test]
fn title_writes_and_runtime_writes_are_apart() {
    let c = classifier(16);
    let a = 0x2_0000_0000;
    c.allocated(a, 2 * K64, true);
    c.tag(f::CPU_W_RUNTIME, a, 8);
    c.tag(f::GPU_W_MARK, a, K64);
    c.tag(f::CPU_W_TITLE, a + K64, 8);
    c.tag(f::GPU_W_MARK, a + K64, K64);
    frame(&c);
    let u = units(&c);
    assert_eq!(u[0].1.ever, f::CPU_W_RUNTIME | f::GPU_W_MARK);
    assert_eq!(u[1].1.ever, f::CPU_W_TITLE | f::GPU_W_MARK);
    let classes = c.classes(Rule::PROPOSED);
    assert_eq!(classes.get(&Class::GpuOnly), Some(&(K64 as u64)));
    assert_eq!(classes.get(&Class::Shared), Some(&(K64 as u64)));
}

/// A GPU write sets only its own flag: watched memory only shaders wrote
/// is GPU-only, memory only the title wrote CPU-only, untouched memory has
/// no unit.
#[test]
fn a_gpu_mark_is_only_a_gpu_write() {
    let c = classifier(16);
    let a = 0x3_0000_0000;
    c.allocated(a, 8 * K64, true);
    c.tag(f::GPU_W_MARK, a + 2 * K64, K64);
    c.tag(f::CPU_W_TITLE, a + 5 * K64 + 3, 1);
    frame(&c);
    let u = units(&c);
    assert_eq!(u.len(), 2);
    assert_eq!((u[0].0.1, u[0].1.ever), (2, f::GPU_W_MARK));
    assert_eq!((u[1].0.1, u[1].1.ever), (5, f::CPU_W_TITLE));
    let classes = c.classes(Rule::PROPOSED);
    assert_eq!(classes.get(&Class::GpuOnly), Some(&(K64 as u64)));
    assert_eq!(classes.get(&Class::CpuOnly), Some(&(K64 as u64)));
    assert_eq!(classes.get(&Class::Shared), None);
}

/// Views of a section count at the section's offset: the same page seen
/// through two addresses in a frame is one unit, once.
#[test]
fn views_of_a_section_count_once_at_the_section() {
    let c = classifier(16);
    let (section, a, b) = (0x44, 0x5_0000_0000, 0x7_1230_0000);
    c.section(section, 4 * K64);
    c.view(a, 2 * K64, section, K64);
    c.view(b, 4 * K64, section, 0);
    // Page 1 of the section: a + 0 and b + 64 KiB.
    c.tag(f::CPU_W_TITLE, a + 100, 1);
    c.tag(f::GPU_R_HEAP, b + K64 + 200, 1);
    frame(&c);
    let u = units(&c);
    assert_eq!(u.len(), 1, "{u:?}");
    assert_eq!(u[0].0, (0, 1));
    assert_eq!((u[0].1.ever, u[0].1.frames), (f::CPU_W_TITLE | f::GPU_R_HEAP, 1));
    assert_eq!(stats(&c, 0).hits[bit(f::CPU_W_TITLE)], 1);
    assert_eq!(c.classes(Rule::PROPOSED).get(&Class::Shared), Some(&(K64 as u64)));
}

/// Memory whose title writes nothing sees (unwatched allocations, memory
/// no region names) is never GPU-only: unknown.
#[test]
fn memory_the_guards_do_not_see_is_never_gpu_only() {
    let c = classifier(16);
    let (watched, unwatched, nowhere) = (0x8_0000_0000, 0x8_1000_0000, 0x9_0000_0000);
    c.allocated(watched, K64, true);
    c.allocated(unwatched, K64, false);
    for at in [watched, unwatched, nowhere] {
        c.tag(f::GPU_W_MARK, at, K64);
    }
    frame(&c);
    let classes = c.classes(Rule::PROPOSED);
    assert_eq!(classes.get(&Class::GpuOnly), Some(&(K64 as u64)));
    assert_eq!(classes.get(&Class::Unknown), Some(&(2 * K64 as u64)));
}

/// The unit follows `VICTORY_MEMCLASS_GRAN`; a range covers every unit it
/// touches, its first and last too.
#[test]
fn ranges_cover_every_unit_they_touch() {
    for (shift, unit) in [(12, K4), (16, K64)] {
        let c = classifier(shift);
        let a = 0xa_0000_0000;
        c.allocated(a, 64 * K64, true);
        c.tag(f::CPU_W_TITLE, a + 5, 1);
        // From the last byte of unit 2 to the first byte of unit 5.
        c.tag(f::GPU_W_BACK, a + 3 * unit - 1, 2 * unit + 2);
        frame(&c);
        let u: Vec<(u64, u16)> = units(&c).iter().map(|((_, k), u)| (*k, u.ever)).collect();
        assert_eq!(
            u,
            [(0, f::CPU_W_TITLE), (2, f::GPU_W_BACK), (3, f::GPU_W_BACK), (4, f::GPU_W_BACK), (5, f::GPU_W_BACK)],
            "unit {unit}"
        );
        let classes = c.classes(Rule::PROPOSED);
        assert_eq!(classes.get(&Class::CpuOnly), Some(&(unit as u64)));
        assert_eq!(classes.get(&Class::GpuOnly), Some(&(4 * unit as u64)));
    }
}

/// Work done once per frame is asked for again in the next frame.
#[test]
fn first_in_frame_is_per_frame() {
    let c = classifier(16);
    assert!(c.first_in_frame(7) && !c.first_in_frame(7) && c.first_in_frame(8));
    frame(&c);
    assert!(c.first_in_frame(7));
}

/// A range over the end of a 4 GiB window reaches into the next one.
#[test]
fn a_range_over_a_window_end_goes_on_in_the_next() {
    let c = classifier(16);
    let end = 0xc_0000_0000;
    c.allocated(end - 2 * K64, 4 * K64, true);
    c.tag(f::GPU_R_TABLE, end - K64, 2 * K64);
    frame(&c);
    let u: Vec<u64> = units(&c).iter().map(|((_, k), _)| *k).collect();
    assert_eq!(u, [1, 2]);
}

/// Sync traffic goes to the region it was at, and the table names it.
#[test]
fn sync_traffic_goes_to_its_region() {
    let c = classifier(16);
    let (a, b) = (0xd_0000_0000, 0xd_1000_0000);
    c.allocated(a, K64, true);
    c.allocated(b, K64, true);
    c.sync(Sync::Up, b + 4, 1000, 30);
    c.sync(Sync::Back, a, 64, 5);
    c.sync(Sync::Up, b, 24, 2);
    frame(&c);
    assert_eq!(stats(&c, 1).sync, [(1024, 32), (0, 0)]);
    assert_eq!(stats(&c, 0).sync, [(0, 0), (64, 5)]);
    let regions = c.regions.read().unwrap();
    let st = c.state.lock().unwrap();
    let table = c.table(&regions, &st);
    assert!(table.iter().any(|l| l.starts_with("memclass region id=1 kind=virtual")
        && l.contains(" sync_up_bytes=1024 sync_up_us=32")));
}

/// A view unmapped (remapped) no longer names its addresses: what touches
/// them later is nowhere until a new view says otherwise.
#[test]
fn an_unmapped_view_names_nothing() {
    let c = classifier(16);
    let (section, a) = (0x9, 0xe_0000_0000);
    c.section(section, 2 * K64);
    c.view(a, 2 * K64, section, 0);
    c.remapped(a, 2 * K64);
    c.tag(f::GPU_W_MARK, a, 1);
    frame(&c);
    let u = units(&c);
    let regions = c.regions.read().unwrap();
    assert_eq!(regions.list[u[0].0.0 as usize].kind, Kind::Unnamed, "{u:?}");
}

/// The table lists units per region and flag combination with their bytes.
#[test]
fn the_table_lists_flag_combinations() {
    let c = classifier(12);
    let a = 0xf_0000_0000;
    c.allocated(a, K64, true);
    c.tag(f::GPU_W_MARK, a, 3 * K4);
    c.tag(f::CPU_W_TITLE, a + 2 * K4, 1);
    frame(&c);
    let regions = c.regions.read().unwrap();
    let st = c.state.lock().unwrap();
    let table = c.table(&regions, &st);
    assert_eq!(table.first().map(|l| l.starts_with("memclass table begin frame=1 unit=4096")), Some(true));
    assert!(table.contains(&"memclass units region=0 ever=0x0008 units=2 bytes=8192".to_string()), "{table:#?}");
    assert!(table.contains(&"memclass units region=0 ever=0x0009 units=1 bytes=4096".to_string()), "{table:#?}");
    assert_eq!(table.last().map(String::as_str), Some("memclass table end frame=1"));
}

/// A record of the stream, decoded (`docs/memclass-format.md`).
#[derive(Debug, PartialEq, Eq)]
enum Record {
    Access { flag: u16, seq: u64, at: u64, len: u64 },
    Counters([u64; 4]),
    Submit { seq: u64, lists: u64 },
    Point { kind: u8, seq: u64 },
    Sync { kind: u8, at: u64, len: u64, us: u64 },
    Frame(u64),
    Region { kind: u8, v: [u64; 4] },
    End(u64),
}

fn decode(bytes: &[u8]) -> (u8, Vec<Record>) {
    assert_eq!(&bytes[..4], STREAM_MAGIC);
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), STREAM_VERSION);
    let shift = bytes[6];
    let mut i = 8;
    let mut next = || {
        let (mut v, mut s) = (0u64, 0);
        loop {
            let b = bytes[i];
            i += 1;
            v |= u64::from(b & 0x7f) << s;
            s += 7;
            if b < 0x80 {
                return v;
            }
        }
    };
    let mut out = Vec::new();
    let mut last = 0i64;
    loop {
        let tag = match next() {
            t if t <= 0xff => t as u8,
            t => panic!("bad tag {t}"),
        };
        out.push(match tag {
            rec::ACCESS => {
                let flag = next() as u16;
                let _tid = next();
                let seq = next();
                let z = next();
                let at = last + ((z >> 1) as i64 ^ -((z & 1) as i64));
                last = at;
                Record::Access { flag, seq, at: at as u64, len: next() }
            }
            rec::SUBMIT => Record::Submit { seq: next(), lists: next() },
            rec::POINT => Record::Point { kind: next() as u8, seq: next() },
            rec::SYNC => Record::Sync { kind: next() as u8, at: next(), len: next(), us: next() },
            rec::FRAME => Record::Frame(next()),
            rec::COUNTERS => Record::Counters([next(), next(), next(), next()]),
            rec::REGION => Record::Region { kind: next() as u8, v: [next(), next(), next(), next()] },
            rec::END => {
                out.push(Record::End(next()));
                return (shift, out);
            }
            t => panic!("unknown record {t}"),
        });
    }
}

/// The counters a frame record carries in this build: copystat's only in
/// diagnostics builds (where other tests move them too: not compared).
fn counters(copied: u64) -> [u64; 4] {
    [0, 0, copied, 0]
}

fn without_copystat(records: Vec<Record>) -> Vec<Record> {
    records
        .into_iter()
        .map(|r| match r {
            Record::Counters([_, _, copied, _]) => Record::Counters(counters(copied)),
            r => r,
        })
        .collect()
}

/// The stream holds every hook's call in order, with submissions, points,
/// sync and frames, and ends after the frames asked for.
#[test]
fn the_stream_keeps_the_order_of_events() {
    let dir = std::env::temp_dir().join(format!("memclass-{}-{:?}", std::process::id(), std::thread::current().id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.vmcs");
    {
        let c = Classifier::new(Config { stream: Some(path.clone()), stream_frames: 2, ..config(12) });
        let a = 0x7ff6_1234_0000usize;
        c.allocated(a, 2 * K64, true);
        let sub = c.submit(3);
        c.tag(f::GPU_R_REFRESH, a + K64, 100);
        c.point(Point::Executed);
        c.tag(f::CPU_W_TITLE, a + 8, 1);
        c.point(Point::GpuDone);
        c.sync(Sync::Back, a + 16, 32, 7);
        // A refresh copy and the counter the check compares it with.
        c.sync(Sync::Up, a, 4096, 3);
        c.copied.fetch_add(4096, Relaxed);
        drop(sub);
        frame(&c);
        c.tag(f::GPU_W_MARK, a, K64);
        frame(&c);
        // Past the end: not in the stream.
        c.tag(f::GPU_W_MARK, a + K64, K64);
        frame(&c);
    }
    let bytes = std::fs::read(&path).unwrap();
    // For the reader's cross-check (`tools/memclass/memclass_stream.py`).
    if let Ok(keep) = std::env::var("MEMCLASS_KEEP_STREAM") {
        std::fs::write(keep, &bytes).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
    let (shift, records) = decode(&bytes);
    let records = without_copystat(records);
    let a = 0x7ff6_1234_0000u64;
    assert_eq!(shift, 12);
    assert_eq!(
        records,
        [
            Record::Region { kind: 1, v: [a, 2 * K64 as u64, 0, 0] },
            Record::Submit { seq: 1, lists: 3 },
            Record::Access { flag: f::GPU_R_REFRESH, seq: 1, at: a + K64 as u64, len: 100 },
            Record::Point { kind: Point::Executed as u8, seq: 1 },
            Record::Access { flag: f::CPU_W_TITLE, seq: 1, at: a + 8, len: 1 },
            Record::Point { kind: Point::GpuDone as u8, seq: 1 },
            Record::Sync { kind: Sync::Back as u8, at: a + 16, len: 32, us: 7 },
            Record::Sync { kind: Sync::Up as u8, at: a, len: 4096, us: 3 },
            Record::Point { kind: Point::End as u8, seq: 1 },
            Record::Frame(1),
            Record::Counters(counters(4096)),
            Record::Access { flag: f::GPU_W_MARK, seq: 1, at: a, len: K64 as u64 },
            Record::Frame(2),
            Record::Counters(counters(4096)),
            Record::End(2),
        ]
    );
}
