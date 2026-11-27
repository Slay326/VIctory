//! The ray tracing reader's output, captured in the game (local test
//! branch, diagnostics builds): `VICTORY_READER_CAPTURE=<folder>` saves,
//! for dispatches of the reader program (f607780a,
//! `IrradianceCache_ShadowTrace_WithFar`), what its views hold once the
//! submission ran, before anything goes back to title memory: the output
//! UAV u52 (a typed buffer, one element per probe), the probe index list
//! t373 and the count t379, the indirect arguments and the root constants.
//! Offline, runs with the BVH builder on and off are compared per probe
//! (`scripts\s102\reader_cmp.py` in the workspace).
//!
//! * `VICTORY_READER_CAPTURE_AT` (0): the first presented frame at which
//!   a dispatch is captured (an event such as the scene trigger, not a
//!   clock).
//! * `VICTORY_READER_CAPTURE_N` (2): how many submissions are captured
//!   from there (two in a row: a temporal pair for an in-run A-A).
//! * `VICTORY_READER_CAPTURE_HASH` (`f607780a`): the reader program's
//!   shader hash prefix.
//! * `VICTORY_READER_CAPTURE_VIEWS` (`u52,t373,t379`): the table views
//!   taken, by kind and register.
//!
//! Without `VICTORY_READER_CAPTURE` nothing is recorded, allocated or
//! written. One file per dispatch, `reader-f<frame>-s<submission>-d<k>.bin`
//! ([`encode`]); a file counts as input only if [`self_check`] passes.
//! Captures hold game data: never commit them.

use std::path::{Path, PathBuf};

/// The file format's version.
pub const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"VRDRCAP1";
const END: &[u8; 8] = b"VRDREND1";

/// What `VICTORY_READER_CAPTURE*` asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub folder: PathBuf,
    /// First presented frame captured.
    pub at: u64,
    /// Submissions captured.
    pub most: u64,
    /// Shader hash prefix, lower case.
    pub hash: String,
    /// Views taken: (UAV, register).
    pub views: Vec<(bool, u32)>,
}

impl Plan {
    /// The plan the variables give (`var` looks one up); None without a
    /// folder.
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Plan> {
        let folder = var("VICTORY_READER_CAPTURE").filter(|v| !v.trim().is_empty())?;
        let number = |name: &str, default: u64| var(name).and_then(|v| v.trim().parse().ok()).unwrap_or(default);
        let views = var("VICTORY_READER_CAPTURE_VIEWS").unwrap_or_else(|| "u52,t373,t379".into());
        Some(Plan {
            folder: PathBuf::from(folder.trim()),
            at: number("VICTORY_READER_CAPTURE_AT", 0),
            most: number("VICTORY_READER_CAPTURE_N", 2),
            hash: var("VICTORY_READER_CAPTURE_HASH")
                .map(|v| v.trim().to_ascii_lowercase())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "f607780a".into()),
            views: views
                .split(',')
                .filter_map(|v| {
                    let v = v.trim();
                    let uav = v.starts_with('u');
                    (uav || v.starts_with('t')).then(|| v[1..].parse().ok().map(|r| (uav, r))).flatten()
                })
                .collect(),
        })
    }
}

/// Whether a dispatch of the program with shader hash `hash` at presented
/// frame `frame` is to be recorded, `taken` submissions captured so far.
/// Never without a plan.
pub fn wanted(plan: Option<&Plan>, hash: &str, frame: u64, taken: u64) -> bool {
    let Some(plan) = plan else { return false };
    frame >= plan.at && taken < plan.most && hash.to_ascii_lowercase().starts_with(&plan.hash)
}

/// One range of a capture: a view, the indirect arguments or the root
/// constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// `u52`, `t373`, `t379`, `args`, `consts`.
    pub name: String,
    /// Title address (0 for the constants).
    pub address: u64,
    /// Bytes the range has.
    pub expected: u64,
    /// The view's format (host DXGI value, 0 if none) and element stride.
    pub format: u32,
    pub stride: u32,
    /// The bytes read; fewer than `expected` if a page could not be read.
    pub bytes: Vec<u8>,
}

/// A dispatch's capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture {
    /// `key=value` lines: source hash, frame, submission, switches, ...
    pub meta: Vec<(String, String)>,
    pub sections: Vec<Section>,
}

/// FNV-1a, 64 bits (the comparison script checks it the same way).
pub fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The file's bytes: magic, version, meta text, sections (name in 8 bytes,
/// address, expected, read length, format, stride, bytes), the end mark,
/// the total length and FNV-1a of everything before it.
pub fn encode(c: &Capture) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let text: String = c.meta.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(&(c.sections.len() as u32).to_le_bytes());
    for s in &c.sections {
        let mut name = [0u8; 8];
        for (d, b) in name.iter_mut().zip(s.name.bytes()) {
            *d = b;
        }
        out.extend_from_slice(&name);
        out.extend_from_slice(&s.address.to_le_bytes());
        out.extend_from_slice(&s.expected.to_le_bytes());
        out.extend_from_slice(&(s.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&s.format.to_le_bytes());
        out.extend_from_slice(&s.stride.to_le_bytes());
        out.extend_from_slice(&s.bytes);
    }
    out.extend_from_slice(END);
    let total = out.len() as u64 + 16;
    out.extend_from_slice(&total.to_le_bytes());
    let h = fnv(&out);
    out.extend_from_slice(&h.to_le_bytes());
    out
}

struct Cursor<'a>(&'a [u8], usize);

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.1.checked_add(n).filter(|&e| e <= self.0.len()).ok_or("file ends early")?;
        let s = &self.0[self.1..end];
        self.1 = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| "u32")?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "u64")?))
    }
}

/// The capture in `bytes`, if they are one of this version (no check of
/// completeness: [`self_check`]).
pub fn decode(bytes: &[u8]) -> Result<Capture, String> {
    let mut c = Cursor(bytes, 0);
    if c.take(8)? != MAGIC {
        return Err("not a reader capture".into());
    }
    let version = c.u32()?;
    if version != VERSION {
        return Err(format!("format version {version}, want {VERSION}"));
    }
    let n = c.u32()? as usize;
    let text = std::str::from_utf8(c.take(n)?).map_err(|_| "meta not UTF-8")?;
    let meta = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let count = c.u32()?;
    let mut sections = Vec::new();
    for _ in 0..count {
        let name = String::from_utf8_lossy(c.take(8)?).trim_end_matches('\0').to_string();
        let address = c.u64()?;
        let expected = c.u64()?;
        let got = c.u64()?;
        let format = c.u32()?;
        let stride = c.u32()?;
        let bytes = c.take(usize::try_from(got).map_err(|_| "length")?)?.to_vec();
        sections.push(Section { name, address, expected, format, stride, bytes });
    }
    if c.take(8)? != END {
        return Err("no end mark".into());
    }
    let total = c.u64()?;
    let h = c.u64()?;
    if total != bytes.len() as u64 || c.1 != bytes.len() {
        return Err(format!("length {} but the file says {total}", bytes.len()));
    }
    if fnv(&bytes[..bytes.len() - 8]) != h {
        return Err("checksum differs".into());
    }
    Ok(Capture { meta, sections })
}

/// Whether `bytes` count as input: they decode, name `source_hash` as the
/// translator, every section was read whole (no page missing), the views
/// in `need` are there with a known element stride, and the output has an
/// element for each probe the count names. The capture, or why not.
pub fn self_check(bytes: &[u8], source_hash: &str, need: &[&str]) -> Result<Capture, String> {
    let c = decode(bytes)?;
    let meta = |k: &str| c.meta.iter().find(|(m, _)| m == k).map(|(_, v)| v.as_str());
    match meta("source_hash") {
        Some(h) if h == source_hash => {}
        h => return Err(format!("source hash {h:?}, want {source_hash}")),
    }
    for s in &c.sections {
        if (s.bytes.len() as u64) < s.expected {
            return Err(format!("{}: {} of {} bytes read", s.name, s.bytes.len(), s.expected));
        }
    }
    for name in need {
        let Some(s) = c.sections.iter().find(|s| s.name == *name) else {
            return Err(format!("{name} missing"));
        };
        if s.expected == 0 || s.stride == 0 {
            return Err(format!("{name}: empty or no element stride"));
        }
    }
    let section = |n: &str| c.sections.iter().find(|s| s.name == n);
    if let (Some(out), Some(count)) = (section("u52"), section("t379")) {
        let n = count.bytes.get(..4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let elements = out.expected / u64::from(out.stride.max(1));
        if u64::from(n) > elements {
            return Err(format!("count {n} above the output's {elements} elements"));
        }
    }
    Ok(c)
}

/// The file name of a dispatch's capture.
pub fn file_name(frame: u64, submission: u64, dispatch: usize) -> String {
    format!("reader-f{frame}-s{submission}-d{dispatch}.bin")
}

/// Writes the captures of one submission into `plan`'s folder; nothing
/// without a plan (and then `read` is never asked). `read(k)` gives the
/// k-th recorded dispatch's capture. The files written.
pub fn write_all(
    plan: Option<&Plan>,
    frame: u64,
    submission: u64,
    dispatches: usize,
    mut read: impl FnMut(usize) -> Capture,
) -> Vec<PathBuf> {
    let Some(plan) = plan else { return Vec::new() };
    if dispatches == 0 {
        return Vec::new();
    }
    let _ = std::fs::create_dir_all(&plan.folder);
    let mut written = Vec::new();
    for k in 0..dispatches {
        let c = read(k);
        let path = plan.folder.join(file_name(frame, submission, k));
        if write_file(&path, &encode(&c)).is_ok() {
            written.push(path);
        }
    }
    written
}

/// Writes through a temporary name, so that a file of the final name is
/// whole.
fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) use game::*;

/// Game side: recording at the dispatch, taking after the fence.
#[cfg(all(windows, feature = "diagnostics"))]
mod game {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock};

    use crate::shadow::Shadow;

    /// The plan from the environment, read once.
    pub(crate) fn plan() -> Option<&'static Plan> {
        static PLAN: OnceLock<Option<Plan>> = OnceLock::new();
        PLAN.get_or_init(|| Plan::from_vars(|k| std::env::var(k).ok())).as_ref()
    }

    /// Submissions captured so far.
    static TAKEN: AtomicU64 = AtomicU64::new(0);

    /// Whether a dispatch of the program `hash` is recorded now.
    pub(crate) fn wanted_now(hash: &str) -> bool {
        wanted(plan(), hash, crate::queue::presented(), TAKEN.load(Ordering::Relaxed))
    }

    /// A view or range as recorded at the dispatch: where it is and the
    /// copy whose GPU bytes hold it.
    pub(crate) struct Range {
        pub(crate) name: String,
        pub(crate) address: u64,
        pub(crate) len: u64,
        pub(crate) format: u32,
        pub(crate) stride: u32,
        pub(crate) copy: Option<Arc<Shadow>>,
    }

    /// A reader dispatch as recorded.
    pub(crate) struct Recorded {
        pub(crate) shader: String,
        pub(crate) pipeline: String,
        pub(crate) indirect: bool,
        pub(crate) ranges: Vec<Range>,
        /// Root constants by root parameter.
        pub(crate) constants: Vec<(u32, Vec<u32>)>,
        /// The program's table views: kind, register, root slot, dword.
        pub(crate) used: Vec<String>,
    }

    /// The bytes of `[address, address + len)` as the GPU left them in
    /// `copy` (one fetch for all pages, the CPU side left as it was);
    /// shorter if pages are missing.
    fn gpu_range(copy: &Shadow, address: u64, len: u64) -> Vec<u8> {
        const PAGE: u64 = 1 << 16;
        let first = address & !(PAGE - 1);
        let pages: Vec<usize> = (first..address + len).step_by(PAGE as usize).map(|p| p as usize).collect();
        let mut got = copy.capture_pages(&pages);
        got.sort_by_key(|(at, _, _)| *at);
        let mut out = Vec::with_capacity(len as usize);
        let mut next = address;
        for (at, bytes, _) in got {
            let (at, end) = (at as u64, at as u64 + bytes.len() as u64);
            if end <= next {
                continue;
            }
            if at > next {
                // A page missing: the rest stays out, the self-check fails.
                break;
            }
            let to = end.min(address + len);
            out.extend_from_slice(&bytes[(next - at) as usize..(to - at) as usize]);
            next = to;
            if next >= address + len {
                break;
            }
        }
        out
    }

    /// After the fence of submission `submission` (before anything goes
    /// back): the recorded reader dispatches of `runs` written out.
    pub(crate) fn take(records: Vec<Recorded>, submission: u64) {
        let Some(plan) = plan() else { return };
        if records.is_empty() || TAKEN.fetch_add(1, Ordering::Relaxed) >= plan.most {
            return;
        }
        let frame = crate::queue::presented();
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
        let switches: Vec<String> = std::env::vars()
            .filter(|(k, _)| k.starts_with("VICTORY_BVH") || k == "VICTORY_SKIP_PIPELINES" || k == "VICTORY_DISPATCHX_THREADS")
            .map(|(k, v)| if k == "VICTORY_SKIP_PIPELINES" { format!("{k}=fnv:{:016x}", fnv(v.as_bytes())) } else { format!("{k}={v}") })
            .collect();
        let written = write_all(Some(plan), frame, submission, records.len(), |k| {
            let r = &records[k];
            let mut meta = vec![
                ("source_hash".to_string(), victory_shader::SOURCE_HASH.to_string()),
                ("exe".into(), exe.clone()),
                ("frame".into(), frame.to_string()),
                ("submission".into(), submission.to_string()),
                ("dispatch".into(), k.to_string()),
                ("dispatches".into(), records.len().to_string()),
                ("shader".into(), r.shader.clone()),
                ("pipeline".into(), r.pipeline.clone()),
                ("indirect".into(), r.indirect.to_string()),
                ("taken".into(), "after the fence, before copy-back".into()),
                ("switches".into(), switches.join(" ")),
            ];
            meta.extend(r.used.iter().map(|u| ("used".to_string(), u.clone())));
            let mut sections: Vec<Section> = r
                .ranges
                .iter()
                .map(|g| Section {
                    name: g.name.clone(),
                    address: g.address,
                    expected: g.len,
                    format: g.format,
                    stride: g.stride,
                    bytes: g.copy.as_ref().map(|c| gpu_range(c, g.address, g.len)).unwrap_or_default(),
                })
                .collect();
            let consts: Vec<u8> = r.constants.iter().flat_map(|(_, v)| v.iter().flat_map(|w| w.to_le_bytes())).collect();
            meta.push((
                "consts".into(),
                r.constants.iter().map(|(p, v)| format!("p{p}:{}", v.len())).collect::<Vec<_>>().join(","),
            ));
            sections.push(Section {
                name: "consts".into(),
                address: 0,
                expected: consts.len() as u64,
                format: 0,
                stride: 4,
                bytes: consts,
            });
            Capture { meta, sections }
        });
        for path in &written {
            let verdict = std::fs::read(path)
                .map_err(|e| e.to_string())
                .and_then(|b| self_check(&b, victory_shader::SOURCE_HASH, &["u52", "t373", "t379"]).map(|_| ()));
            tracing::info!(
                target: "victory_gpu::reader_capture",
                "reader capture s{submission} frame {frame}: {} self-check {}",
                path.display(),
                verdict.map_or_else(|e| format!("FAILED: {e}"), |()| "passed".into()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(dir: &Path) -> Plan {
        Plan::from_vars(|k| (k == "VICTORY_READER_CAPTURE").then(|| dir.display().to_string())).unwrap()
    }

    fn sample() -> Capture {
        let out: Vec<u8> = (0..8u32).flat_map(|i| [i as f32, 0.5, 0.25, 1.0]).flat_map(f32::to_le_bytes).collect();
        let ids: Vec<u8> = [3u32, 1, 7, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        Capture {
            meta: vec![("source_hash".into(), "f1cf82997c12266d".into()), ("frame".into(), "9".into())],
            sections: vec![
                Section { name: "u52".into(), address: 0x1000, expected: out.len() as u64, format: 2, stride: 16, bytes: out },
                Section { name: "t373".into(), address: 0x2000, expected: 16, format: 42, stride: 4, bytes: ids },
                Section { name: "t379".into(), address: 0x3000, expected: 4, format: 42, stride: 4, bytes: 4u32.to_le_bytes().to_vec() },
            ],
        }
    }

    const NEED: [&str; 3] = ["u52", "t373", "t379"];

    #[test]
    fn a_whole_capture_passes_its_self_check() {
        let c = sample();
        assert_eq!(self_check(&encode(&c), "f1cf82997c12266d", &NEED), Ok(c));
    }

    #[test]
    fn the_self_check_rejects_an_incomplete_capture() {
        // A page of the output could not be read: fewer bytes than the view has.
        let mut c = sample();
        c.sections[0].bytes.truncate(64);
        let e = self_check(&encode(&c), "f1cf82997c12266d", &NEED).unwrap_err();
        assert!(e.contains("u52: 64 of 128"), "{e}");
        // A file cut short (the game stopped while writing).
        let whole = encode(&sample());
        assert!(self_check(&whole[..whole.len() - 20], "f1cf82997c12266d", &NEED).is_err());
        // A view not found at the dispatch.
        let mut c = sample();
        c.sections.remove(1);
        assert!(self_check(&encode(&c), "f1cf82997c12266d", &NEED).unwrap_err().contains("t373 missing"));
        // Another translator, another version.
        assert!(self_check(&whole, "0000000000000000", &NEED).unwrap_err().contains("source hash"));
        let mut v2 = whole.clone();
        v2[8] = 2;
        assert!(self_check(&v2, "f1cf82997c12266d", &NEED).is_err());
        // More probes counted than the output has elements.
        let mut c = sample();
        c.sections[2].bytes = 9u32.to_le_bytes().to_vec();
        assert!(self_check(&encode(&c), "f1cf82997c12266d", &NEED).unwrap_err().contains("count 9"));
    }

    #[test]
    fn without_the_switch_nothing_is_recorded_read_or_written() {
        let dir = tempfile::tempdir().unwrap();
        let none = Plan::from_vars(|_| None);
        assert_eq!(none, None);
        assert!(Plan::from_vars(|k| (k == "VICTORY_READER_CAPTURE").then(String::new)).is_none());
        assert!(!wanted(none.as_ref(), "f607780a12345678", 5000, 0));
        let mut asked = 0;
        let written = write_all(none.as_ref(), 5000, 7, 3, |_| {
            asked += 1;
            sample()
        });
        assert!(written.is_empty());
        assert_eq!(asked, 0);
        assert_eq!(written.capacity(), 0, "nothing allocated");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn with_the_switch_the_reader_is_recorded_from_its_frame_on() {
        let dir = tempfile::tempdir().unwrap();
        let p = plan(&dir.path().join("cap"));
        assert_eq!((p.at, p.most, p.hash.as_str()), (0, 2, "f607780a"));
        assert_eq!(p.views, vec![(true, 52), (false, 373), (false, 379)]);
        assert!(wanted(Some(&p), "F607780A12345678", 0, 1));
        assert!(!wanted(Some(&p), "5552dac612345678", 0, 0));
        assert!(!wanted(Some(&p), "f607780a12345678", 0, 2));
        let late = Plan { at: 100, ..p.clone() };
        assert!(!wanted(Some(&late), "f607780a12345678", 99, 0));
        let written = write_all(Some(&p), 100, 7, 2, |_| sample());
        assert_eq!(written.len(), 2);
        assert!(written[1].ends_with("reader-f100-s7-d1.bin"));
        let bytes = std::fs::read(&written[0]).unwrap();
        assert!(self_check(&bytes, "f1cf82997c12266d", &NEED).is_ok());
    }
}
