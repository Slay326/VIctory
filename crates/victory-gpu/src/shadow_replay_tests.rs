//! Tests of the replayer (`shadow::replay`), without game data: a
//! synthetic stream written from the format document, and the stream the
//! recorder itself writes.

use super::*;
use crate::memclass::{f, rec};

const P: u64 = KIB64;

fn var(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// A stream writer from `docs/memclass-format.md` (v1).
struct W {
    b: Vec<u8>,
    last: u64,
    seq: u64,
    frame: u64,
    /// `SYNC up` bytes so far: the `copied` counter of the frame records.
    copied: u64,
}

impl W {
    fn new() -> W {
        W::with_shift(16)
    }
    fn with_shift(shift: u8) -> W {
        let mut b = b"VMCS".to_vec();
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&[shift, 0]);
        W {
            b,
            last: 0,
            seq: 0,
            frame: 0,
            copied: 0,
        }
    }
    fn access(&mut self, flag: u16, at: u64, len: u64) {
        let d = at.wrapping_sub(self.last) as i64;
        self.last = at;
        let z = ((d << 1) ^ (d >> 63)) as u64;
        var(&mut self.b, u64::from(rec::ACCESS));
        for v in [u64::from(flag), 7, self.seq, z, len] {
            var(&mut self.b, v);
        }
    }
    fn submit(&mut self, lists: u64) {
        self.seq += 1;
        for v in [u64::from(rec::SUBMIT), self.seq, lists] {
            var(&mut self.b, v);
        }
    }
    fn point(&mut self, kind: u64) {
        for v in [u64::from(rec::POINT), kind, self.seq] {
            var(&mut self.b, v);
        }
    }
    fn sync(&mut self, kind: u64, at: u64, len: u64) {
        if kind == 0 {
            self.copied += len;
        }
        for v in [u64::from(rec::SYNC), kind, at, len, 3] {
            var(&mut self.b, v);
        }
    }
    fn frame(&mut self) {
        self.frame_with([0, 0, self.copied, 0]);
    }
    fn frame_with(&mut self, counters: [u64; 4]) {
        self.frame += 1;
        for v in [u64::from(rec::FRAME), self.frame, u64::from(rec::COUNTERS)] {
            var(&mut self.b, v);
        }
        for v in counters {
            var(&mut self.b, v);
        }
    }
    fn end(mut self) -> Vec<u8> {
        var(&mut self.b, u64::from(rec::END));
        var(&mut self.b, self.frame);
        self.b
    }
}

/// Stream addresses of the synthetic regions: G only the GPU uses, S both,
/// C only the CPU; OUT is in no region.
const G: u64 = 0x2_0000_0000;
const S: u64 = 0x2_0010_0000;
const C: u64 = 0x2_0020_0000;
const OUT: u64 = 0x3_0000_0000;

/// The classification that is right for [`synthetic`], and one with S
/// (shared) marked GPU-only.
fn classes(s: Class) -> Vec<ClassRegion> {
    vec![
        ClassRegion {
            at: G,
            len: 4 * P,
            class: Class::GpuOnly,
        },
        ClassRegion {
            at: S,
            len: 4 * P,
            class: s,
        },
        ClassRegion {
            at: C,
            len: 2 * P,
            class: Class::CpuOnly,
        },
    ]
}

fn class_list(s: Class) -> Vec<Class> {
    classes(s).iter().map(|c| c.class).collect()
}

/// A synthetic stream, `frames` frames of: title writes to S and C, a
/// runtime write to S, a submission (refreshes and heap reads of G and S,
/// declared ranges, a title write while the GPU runs, GPU writes back into
/// G and S inside and outside the declared ranges, a GPU copy into G), then
/// title writes to pages of S the GPU just wrote, and a write outside every
/// region. Two groups of lists go to the GPU (two `POINT executed`), and the
/// title also writes while the GPU runs and between `gpu_done` and `end`
/// (while the copy-backs run). Covers: invariant 1 (title bytes refreshed before the next read,
/// GPU bytes kept in the copy), invariant 2 (at the sync point on the
/// snapshot and the computed path, and at the title's touch of a page the
/// GPU wrote), and, with S marked GPU-only, demotions for invariant 3.
fn synthetic(frames: u64) -> Vec<u8> {
    let mut w = W::new();
    for k in 1..=frames {
        // Title writes before the submission.
        w.access(f::CPU_W_TITLE, C + k * 29, 1);
        w.access(f::CPU_W_TITLE, C + P + 3 * k, 1);
        w.access(f::CPU_W_TITLE, S + k * 13, 1);
        w.access(f::CPU_W_TITLE, S + P + k * 17 + 5, 1);
        w.access(f::CPU_W_RUNTIME, S + 2 * P + 100, 3000);
        // The submission: preparation.
        w.submit(2);
        w.access(f::GPU_R_REFRESH, S, 4 * P);
        w.access(f::GPU_R_REFRESH, G, 4 * P);
        w.access(f::GPU_R_HEAP, G, 3 * P);
        w.sync(0, S, 4096);
        w.access(f::GPU_W_DECL, G, 3 * P);
        w.access(f::GPU_W_DECL, S, 3 * P);
        w.point(0);
        // The title writes a declared page while the GPU runs.
        w.access(f::CPU_W_TITLE, S + P + 1000 + k * 31, 1);
        w.point(0);
        w.point(1);
        // Copy-backs: what the GPU wrote.
        w.access(f::GPU_W_MARK, S, P);
        w.access(f::GPU_W_BACK, G + k * 101, 16);
        w.access(f::GPU_W_BACK, G + P + 7, 200);
        w.access(f::GPU_W_COPY, G + 2 * P + 300 + k, 4);
        w.access(f::GPU_W_BACK, S + 200 + k * 7, 8);
        w.access(f::GPU_W_BACK, S + P + 4000 + k, 4);
        // Not declared: written at computed addresses.
        w.access(f::GPU_W_BACK, S + 3 * P + 500 + k, 2);
        w.sync(1, S + 200, 8);
        // The title writes a declared page while the copy-backs run.
        w.access(f::CPU_W_TITLE, S + P + 2000 + k, 1);
        w.point(2);
        // The title then writes pages the GPU wrote.
        w.access(f::CPU_W_TITLE, S + 3 * P + 600 + k, 1);
        w.access(f::CPU_W_TITLE, S + 200 + k * 7, 1);
        w.access(f::CPU_W_TITLE, OUT, 1);
        w.frame();
    }
    w.end()
}

const BUDGET: u64 = 1 << 30;

fn replay<M: MemModel>(cl: &[ClassRegion], model: M, data: &[u8]) -> Option<Replay<M>> {
    let stream = read_stream(data).expect("a stream");
    let mut r = Replay::new(cl, model, BUDGET)?;
    r.run(&stream);
    Some(r)
}

/// The reader reads what the format document says, refuses another
/// version, and reads a stream cut off mid-record up to the cut.
#[test]
fn the_reader_reads_what_the_format_says() {
    let mut w = W::new();
    w.submit(3);
    w.access(f::GPU_R_REFRESH, S + P, 100);
    w.access(f::CPU_W_TITLE, S + 8, 1); // negative delta
    w.point(2);
    w.frame();
    let data = w.end();
    let s = read_stream(&data).unwrap();
    assert!(s.ended);
    assert_eq!(s.shift, 16);
    assert_eq!(
        s.records,
        [
            Rec::Submit { seq: 1, lists: 3 },
            Rec::Access {
                flag: f::GPU_R_REFRESH,
                tid: 7,
                seq: 1,
                at: S + P,
                len: 100
            },
            Rec::Access {
                flag: f::CPU_W_TITLE,
                tid: 7,
                seq: 1,
                at: S + 8,
                len: 1
            },
            Rec::Point { kind: 2, seq: 1 },
            Rec::Frame(1),
            Rec::Counters([0; 4]),
            Rec::End(1),
        ]
    );
    let cut = read_stream(&data[..data.len() - 3]).unwrap();
    assert!(!cut.ended);
    assert_eq!(cut.records.len(), 5, "whole records up to the cut");
    let mut v2 = data.clone();
    v2[4] = 2;
    assert!(read_stream(&v2).is_err(), "another version is refused");
    assert!(read_stream(b"XXXX\x01\x00\x10\x00").is_err());
}

/// The reader reads what the recorder (`crate::memclass`) writes.
#[test]
fn the_reader_reads_the_recorders_stream() {
    use crate::memclass::{Classifier, Config, Point, Sync};
    let dir = std::env::temp_dir().join(format!(
        "replay-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.vmcs");
    let a = 0x7ff6_1234_0000usize;
    {
        let c = Classifier::new(Config {
            shift: 16,
            stream: Some(path.clone()),
            stream_frames: 1,
            interval_s: 3600,
            table_s: 3600,
        });
        let sub = c.submit(2);
        c.tag(f::GPU_R_REFRESH, a + PAGE, 100);
        c.point(Point::Executed);
        c.tag(f::CPU_W_TITLE, a + 8, 1);
        c.point(Point::GpuDone);
        c.sync(Sync::Back, a + 16, 32, 7);
        drop(sub);
        c.frame_done(std::time::Instant::now());
    }
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let s = read_stream(&bytes).unwrap();
    assert!(s.ended);
    let a = a as u64;
    let picked: Vec<Rec> = s
        .records
        .into_iter()
        .filter(|r| !matches!(r, Rec::Counters(_)))
        .map(|r| match r {
            Rec::Access {
                flag, seq, at, len, ..
            } => Rec::Access {
                flag,
                tid: 0,
                seq,
                at,
                len,
            },
            r => r,
        })
        .collect();
    assert_eq!(
        picked,
        [
            Rec::Submit { seq: 1, lists: 2 },
            Rec::Access {
                flag: f::GPU_R_REFRESH,
                tid: 0,
                seq: 1,
                at: a + P,
                len: 100
            },
            Rec::Point { kind: 0, seq: 1 },
            Rec::Access {
                flag: f::CPU_W_TITLE,
                tid: 0,
                seq: 1,
                at: a + 8,
                len: 1
            },
            Rec::Point { kind: 1, seq: 1 },
            Rec::Sync {
                kind: 1,
                at: a + 16,
                len: 32,
                us: 7
            },
            Rec::Point { kind: 2, seq: 1 },
            Rec::Frame(1),
            Rec::End(1),
        ]
    );
}

/// Classifications read in both forms; overlaps and bad lines refused.
#[test]
fn classifications_read_in_both_forms() {
    let c = parse_classes(
        "# a comment\n0x20000 0x10000 gpu-only\nid=3 kind=virtual at=0x40000 len=65536 class=shared\n",
    )
    .unwrap();
    assert_eq!(
        c,
        [
            ClassRegion {
                at: 0x20000,
                len: 0x10000,
                class: Class::GpuOnly
            },
            ClassRegion {
                at: 0x40000,
                len: 0x10000,
                class: Class::Shared
            },
        ]
    );
    assert!(
        parse_classes("0x0 0x20000 shared\n0x10000 0x10000 gpu-only\n").is_err(),
        "overlap"
    );
    assert!(parse_classes("0x0 0x10000 maybe\n").is_err());
    assert!(parse_classes("0x0 0x10000\n").is_err());
}

/// (c) The engine as it is keeps the three invariants on the synthetic
/// stream with the right classification, and every invariant's check runs
/// (invariant 3 needs demotions: the wrong classification below). A fault
/// in the copies' refresh or copy-back makes this red: the replay runs the
/// real `Shadow`.
#[test]
fn the_engine_keeps_the_invariants_on_the_synthetic_stream() {
    let Some(r) = replay(&classes(Class::Shared), Today, &synthetic(6)) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert_eq!(r.violations(), [0; 3], "{report}");
    assert!(r.green(), "{report}");
    let checks = r.checks();
    assert!(
        checks[0] > 0 && checks[1] > 0,
        "invariants 1 and 2 are checked: {checks:?}\n{report}"
    );
    assert!(
        report.contains("outside every region: [cpu_w_title:6]"),
        "{report}"
    );
}

/// (b) and (c) The right classification with GPU-only memory left on the
/// GPU (reference safety net): green, nothing demoted, G never refreshed.
#[test]
fn the_right_classification_is_green_with_gpu_only_memory() {
    let Some(r) = replay(
        &classes(Class::Shared),
        RefNet::new(class_list(Class::Shared)),
        &synthetic(6),
    ) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert!(r.green(), "{report}");
    assert_eq!(
        r.demotions_of(0) + r.demotions_of(1) + r.demotions_of(2),
        0,
        "{report}"
    );
    assert!(
        report.contains("region 0 at=0x200000000 len=0x40000 class=gpu-only:"),
        "{report}"
    );
    // 6 frames of a 4-page refresh and a 3-page heap read of G: none done.
    assert!(
        report.contains(" refreshed_pages=0 unrefreshed_pages=42 "),
        "{report}"
    );
}

/// (b) A shared region marked GPU-only is red: without a safety net the
/// replay breaks invariants 1 and 2; with the reference net it keeps them,
/// demotes every page of S the title writes, and refutes S's class.
#[test]
fn a_shared_region_marked_gpu_only_is_red() {
    let wrong = classes(Class::GpuOnly);
    let Some(r) = replay(
        &wrong,
        NoNet {
            classes: class_list(Class::GpuOnly),
        },
        &synthetic(6),
    ) else {
        return; // no host GPU here
    };
    let report = r.report();
    let v = r.violations();
    assert!(!r.green() && report.contains("verdict: RED"), "{report}");
    assert!(
        v[0] > 0,
        "invariant 1: the title's bytes never reach the copy\n{report}"
    );
    assert!(
        v[1] > 0,
        "invariant 2: the GPU's bytes never reach title memory\n{report}"
    );
    assert_eq!(r.refuted(), [1], "{report}");

    let Some(r) = replay(
        &wrong,
        RefNet::new(class_list(Class::GpuOnly)),
        &synthetic(6),
    ) else {
        return;
    };
    let report = r.report();
    assert_eq!(
        r.violations(),
        [0; 3],
        "the net keeps the invariants\n{report}"
    );
    assert_eq!(
        r.demotions_of(1),
        4,
        "every page of S demoted once\n{report}"
    );
    assert!(
        report.contains("demotions=4 [runtime write:1 title write:3]"),
        "{report}"
    );
    assert!(r.checks()[2] > 0, "invariant 3 is checked\n{report}");
    assert_eq!(r.refuted(), [1], "{report}");
    assert!(
        report.contains("classification: REFUTED in regions [1]")
            && report.contains("verdict: RED"),
        "{report}"
    );
}

/// (c) Invariant 3: a model that forgets its demotions at a frame end (a
/// reclassification undoing them) is caught.
#[test]
fn a_demoted_page_treated_as_gpu_only_again_is_caught() {
    let mut net = RefNet::new(class_list(Class::GpuOnly));
    net.forget_on_frame = true;
    let Some(r) = replay(&classes(Class::GpuOnly), net, &synthetic(3)) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert!(r.violations()[2] > 0, "{report}");
    assert!(
        report.contains("demoted page classed GPU-only again after a frame"),
        "{report}"
    );
}

/// (a) A replay is deterministic: the same stream, classification and model
/// give the same report, also where it is red.
#[test]
fn a_replay_is_deterministic() {
    let data = synthetic(5);
    let run = |s: Class| {
        let a = replay(&classes(s), RefNet::new(class_list(s)), &data)?.report();
        let b = replay(
            &classes(s),
            NoNet {
                classes: class_list(s),
            },
            &data,
        )?
        .report();
        Some(a + &b)
    };
    let (Some(first), Some(second)) = (run(Class::GpuOnly), run(Class::GpuOnly)) else {
        return; // no host GPU here
    };
    assert_eq!(first, second);
    assert_eq!(run(Class::Shared), run(Class::Shared));
}

/// A capture from a run, replayed (not run by default; gate item 3):
/// `VICTORY_REPLAY_STREAM` (the .vmcs), `VICTORY_REPLAY_CLASSES` (regions,
/// [`parse_classes`]), `VICTORY_REPLAY_MODEL` (`today`, `refnet`, `nonet`;
/// default `refnet`), `VICTORY_REPLAY_MAX_MB` (pools, default 4096),
/// `VICTORY_REPLAY_OUT` (the report also to this file). Fails if red.
///
/// `cargo test -p victory-gpu replay_a_capture -- --ignored --nocapture`
#[test]
#[ignore = "needs a capture: VICTORY_REPLAY_STREAM and VICTORY_REPLAY_CLASSES"]
fn replay_a_capture() {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let stream = get("VICTORY_REPLAY_STREAM").expect("VICTORY_REPLAY_STREAM");
    let classes = get("VICTORY_REPLAY_CLASSES").expect("VICTORY_REPLAY_CLASSES");
    let data = std::fs::read(&stream).expect("the stream");
    let cl =
        parse_classes(&std::fs::read_to_string(&classes).expect("the classes")).expect("classes");
    let max = get("VICTORY_REPLAY_MAX_MB")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(4096)
        << 20;
    let s = read_stream(&data).expect("a stream");
    let list: Vec<Class> = cl.iter().map(|c| c.class).collect();
    let (report, green) = match get("VICTORY_REPLAY_MODEL").as_deref().unwrap_or("refnet") {
        "today" => {
            let mut r = Replay::new(&cl, Today, max).expect("a host GPU");
            r.run(&s);
            (r.report(), r.green())
        }
        "nonet" => {
            let mut r = Replay::new(&cl, NoNet { classes: list }, max).expect("a host GPU");
            r.run(&s);
            (r.report(), r.green())
        }
        "refnet" => {
            let mut r = Replay::new(&cl, RefNet::new(list), max).expect("a host GPU");
            r.run(&s);
            (r.report(), r.green())
        }
        other => panic!("VICTORY_REPLAY_MODEL: today, refnet or nonet, not {other}"),
    };
    let report = format!(
        "stream ended: {}\n{report}",
        if s.ended { "yes" } else { "no (cut off)" }
    );
    println!("{report}");
    if let Some(out) = get("VICTORY_REPLAY_OUT") {
        std::fs::write(out, &report).expect("the report file");
    }
    assert!(green, "replay red");
}

/// Records are sampled at both ends, at every 4 KiB boundary (the smallest
/// decision unit), and watched one byte before and after (guards, cut to
/// the region).
#[test]
fn every_4k_unit_and_both_ends_are_sampled_with_guards() {
    let Some(r) = Replay::new(&classes(Class::Shared), Today, BUDGET) else {
        return; // no host GPU here
    };
    assert_eq!(r.samples(100, 9000), [100, 4096, 8192, 8999]);
    assert_eq!(r.samples(4096, 4097), [4096]);
    assert_eq!(r.samples(10, 12), [10, 11]);
    assert_eq!(r.guards(1, 100, 9000), [99, 9000]);
    assert_eq!(
        r.guards(1, 0, 4 * P),
        Vec::<u64>::new(),
        "cut to the region"
    );
}

/// Two title writes to two 4 KiB units of one 64 KiB page of S, marked
/// GPU-only, in a stream of 4 KiB units.
fn two_units_of_one_page() -> Vec<u8> {
    let mut w = W::with_shift(12);
    w.access(f::CPU_W_TITLE, S + 0x10, 1);
    w.access(f::CPU_W_TITLE, S + 0x1010, 1);
    w.frame();
    w.end()
}

/// A net deciding per 4 KiB unit that syncs a demoted unit by taking the
/// copy's whole 64 KiB page loses the title's byte of the unit demoted
/// before: red. The reference net (the copies' page sync) keeps it; at
/// 64 KiB decisions the second write demotes nothing and the naive net
/// does no harm (the decision unit matters, and the report names it).
#[test]
fn a_net_syncing_whole_pages_over_the_titles_bytes_is_red() {
    let wrong = classes(Class::GpuOnly);
    let mut net = RefNet::new(class_list(Class::GpuOnly));
    net.whole_page_sync = true;
    let Some(r) = replay(&wrong, net, &two_units_of_one_page()) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert!(
        report.contains("decision unit=4096 sample=4096"),
        "{report}"
    );
    assert!(
        r.violations()[1] > 0,
        "the title's byte at +0x10 is lost\n{report}"
    );
    assert!(
        report.contains("violation inv2 region=1 at=0x200100010"),
        "{report}"
    );

    let r = replay(
        &wrong,
        RefNet::new(class_list(Class::GpuOnly)),
        &two_units_of_one_page(),
    )
    .unwrap();
    let report = r.report();
    assert_eq!(r.violations(), [0; 3], "{report}");
    assert_eq!(r.demotions_of(1), 2, "two units demoted\n{report}");

    let mut net = RefNet::new(class_list(Class::GpuOnly));
    net.whole_page_sync = true;
    let stream = read_stream(&two_units_of_one_page()).unwrap();
    let mut r = Replay::new(&wrong, net, BUDGET).unwrap();
    r.unit = P;
    r.run(&stream);
    let report = r.report();
    assert_eq!(r.violations(), [0; 3], "{report}");
    assert_eq!(r.demotions_of(1), 1, "{report}");
}

/// The GPU writes G; the title only reads it afterwards (`cpu_r_after_gpu`
/// records when `read`, else nothing shows the reads).
fn gpu_writes_cpu_reads(read: bool) -> Vec<u8> {
    let mut w = W::new();
    for k in 1..=2 {
        w.submit(1);
        w.access(f::GPU_R_REFRESH, G, P);
        w.access(f::GPU_W_DECL, G, P);
        w.point(0);
        w.point(1);
        w.access(f::GPU_W_BACK, G + 0x100 + k, 4);
        w.point(2);
        if read {
            w.access(f::CPU_R_AFTER_GPU, G + 0x100 + k, 4);
        }
        w.frame();
    }
    w.end()
}

/// A page the title only reads, wrongly GPU-only: with CPU read records a
/// model without a net is red (the title would read stale bytes) and the
/// reference net demotes it (reason "title read"); without them the replay
/// is green but says it proves no GPU-only class.
#[test]
fn a_page_the_title_only_reads_is_not_gpu_only() {
    let wrong = classes(Class::Shared); // G GPU-only, but the title reads it
    let Some(r) = replay(
        &wrong,
        NoNet {
            classes: class_list(Class::Shared),
        },
        &gpu_writes_cpu_reads(true),
    ) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert!(r.violations()[1] > 0, "{report}");
    assert!(
        report.contains("verdict: RED") && report.contains("cpu reads: 2 records"),
        "{report}"
    );

    let r = replay(
        &wrong,
        RefNet::new(class_list(Class::Shared)),
        &gpu_writes_cpu_reads(true),
    )
    .unwrap();
    let report = r.report();
    assert_eq!(r.violations(), [0; 3], "{report}");
    assert!(
        report.contains("[title read:1]") && r.refuted() == [0],
        "{report}"
    );

    // Without the read records the same wrong class passes the replay ...
    let r = replay(
        &wrong,
        NoNet {
            classes: class_list(Class::Shared),
        },
        &gpu_writes_cpu_reads(false),
    )
    .unwrap();
    let report = r.report();
    assert!(r.green() && !r.gate_proof(), "{report}");
    assert!(
        report.contains("cpu reads: CPU reads not in the stream, GPU-only not provable"),
        "{report}"
    );
    assert!(report.contains("NOT a gate-3 proof"), "{report}");

    // ... and with them and the right class it is a proof.
    let mut right = classes(Class::Shared);
    right[0].class = Class::Shared;
    let r = replay(&right, Today, &gpu_writes_cpu_reads(true)).unwrap();
    let report = r.report();
    assert!(
        r.gate_proof() && report.contains("verdict: GREEN\n"),
        "{report}"
    );
}

/// The self-check: the synthetic stream (two `executed` per submission)
/// passes; counters off, a cut stream, points out of order and an access
/// of a submission not begun make the replay INVALID, not green.
#[test]
fn streams_failing_the_self_check_are_invalid() {
    let ok = read_stream(&synthetic(3)).unwrap();
    assert_eq!(self_check(&ok), Vec::<String>::new());

    let mut w = W::new();
    w.submit(1);
    w.point(0);
    w.sync(0, S, 4096);
    w.point(1);
    w.point(2);
    w.frame_with([0, 0, 4095, 0]);
    let off = w.end();
    let bad = self_check(&read_stream(&off).unwrap());
    assert!(
        bad.iter()
            .any(|b| b.contains("SYNC up 4096 bytes, counter copied 4095")),
        "{bad:?}"
    );
    let Some(r) = replay(&classes(Class::Shared), Today, &off) else {
        return; // no host GPU here
    };
    let report = r.report();
    assert!(!r.green() && !r.invalid().is_empty(), "{report}");
    assert!(report.contains("verdict: INVALID"), "{report}");

    let data = synthetic(2);
    let cut = self_check(&read_stream(&data[..data.len() - 2]).unwrap());
    assert!(cut.iter().any(|b| b.contains("no END")), "{cut:?}");

    let mut w = W::new();
    w.submit(1);
    w.point(0);
    w.point(2); // no gpu_done
    w.frame();
    let order = self_check(&read_stream(&w.end()).unwrap());
    assert!(
        order.iter().any(|b| b.contains("out of order")),
        "{order:?}"
    );

    let mut w = W::new();
    w.seq = 3; // an access naming submission 3 before any began
    w.access(f::CPU_W_TITLE, S, 1);
    w.seq = 0;
    w.frame();
    let early = self_check(&read_stream(&w.end()).unwrap());
    assert!(early.iter().any(|b| b.contains("not begun")), "{early:?}");
}
