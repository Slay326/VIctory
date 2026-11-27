//! The BVH builder harness (local test branch): runs a captured submission
//! (`super::capture`, format `super::format`) alone on a D3D12 device of its
//! own and checks the structures it builds.
//!
//! * Segments with captured pages get buffers laid out as in the game: a
//!   copy from its title start (in the GPU's memory or in system memory the
//!   GPU reads, as in the game), an arena chunk as the chunk. Every byte is
//!   first poisoned ([`POISON`], a NaN pattern), then the captured pages go
//!   in.
//! * The page table is the game's, with view pairs of these buffers; an
//!   entry whose page was not captured is the null pair, so the page bitmap
//!   catches every access outside the capture (a miss).
//! * Root constants are the game's, but for the heap indices of root views
//!   (the harness's views); root views are the harness's buffers at the
//!   same title address.
//! * After the run: the page bitmap against the game's (self-check), the
//!   validator (`crate::bvh_gt`) on the game's G and on the harness's
//!   result (parity first, then the invalid structures by address and class
//!   of reason), bytes against G, and across `runs` runs (determinism).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use super::format::*;

/// What bytes no capture filled hold: a quiet NaN with a payload.
pub const POISON: u32 = 0x7fc0_dead;
/// Entries of a page table buffer (level 1 and `CAPACITY + 1` blocks).
const TABLE_DWORDS: usize = LEVEL1 + (CAPACITY + 1) * BLOCK * 2;
/// The most elements of a raw buffer view, as the game makes them.
const MOST_ELEMENTS: u64 = 1 << 27;

/// Options of a harness run.
#[derive(Debug, Clone)]
pub struct Options {
    pub capture: PathBuf,
    /// Runs for the determinism check.
    pub runs: u32,
    /// Also run the dispatches that are not the builder's.
    pub all: bool,
    /// A global UAV barrier before every dispatch (T0 variant).
    pub barriers: bool,
    /// Every buffer in the GPU's own memory (else as in the game).
    pub card_only: bool,
    /// The software adapter.
    pub warp: bool,
    /// Where the harness's result pages go (`<dir>/run<k>.pages` + index).
    pub dump: Option<PathBuf>,
    /// Run an incomplete capture anyway, for its output; the verdict stays
    /// CAPTURE INVALID.
    pub diagnose: bool,
    /// Stage dumps (T4): replay only the first k builder dispatches for
    /// each cut k and check the structures after it.
    pub cuts: Option<Cuts>,
    /// T4 coherence test: a file with the title's shader containers
    /// (`XboxBvhBuilder.dll`); the pipelines of [`Options::only`] get their
    /// DXIL translated afresh from it (`VICTORY_SHADER_COHERENT=1`: with
    /// `victory_shader::emit::set_coherent`).
    pub retranslate: Option<PathBuf>,
    /// Shader hash prefixes to translate afresh (default 5552dac6).
    pub only: Vec<String>,
    /// senior 115: a reader capture (`VICTORY_BVH_CAPTURE_READER`): run the
    /// captured program and compare its unordered table views with the
    /// game's bytes after the fence ([`reader_verdict`]).
    pub reader: bool,
    /// senior 122, `--sentinel` (with `--reader`): the unordered table views
    /// start as [`SENTINEL`] bytes instead of the game's bytes before the
    /// program, so a byte the harness writes shows even when it writes the
    /// value that was there ([`compare_view_sentinel`]).
    pub sentinel: bool,
}

/// Where the stage dumps cut the builder's dispatches.
#[derive(Debug, Clone, PartialEq)]
pub enum Cuts {
    /// After the last dispatch of every run of one pipeline.
    Stages,
    /// After every dispatch.
    Each,
    /// After these many dispatches.
    List(Vec<usize>),
}

impl Cuts {
    fn parse(s: &str) -> anyhow::Result<Cuts> {
        Ok(match s {
            "stages" => Cuts::Stages,
            "each" => Cuts::Each,
            l => Cuts::List(
                l.split(',')
                    .map(|n| {
                        n.trim()
                            .parse::<usize>()
                            .context("--cuts: stages, each or k,k,...")
                    })
                    .collect::<anyhow::Result<_>>()?,
            ),
        })
    }

    /// The cuts for these pipelines (in dispatch order): numbers of
    /// dispatches run, ascending, the last one all of them.
    pub fn points(&self, pipelines: &[u32]) -> Vec<usize> {
        let n = pipelines.len();
        let mut v: Vec<usize> = match self {
            Cuts::Stages => (1..=n)
                .filter(|&k| k == n || pipelines[k] != pipelines[k - 1])
                .collect(),
            Cuts::Each => (1..=n).collect(),
            Cuts::List(l) => l.iter().map(|&k| k.clamp(1, n.max(1))).collect(),
        };
        if n > 0 && v.last() != Some(&n) {
            v.push(n);
        }
        v.sort_unstable();
        v.dedup();
        v
    }
}

impl Options {
    pub fn parse(args: &[String]) -> anyhow::Result<Options> {
        let mut o = Options {
            capture: PathBuf::new(),
            runs: 5,
            all: false,
            barriers: false,
            card_only: false,
            warp: false,
            dump: None,
            diagnose: false,
            cuts: None,
            retranslate: None,
            only: vec!["5552dac6".to_string()],
            reader: false,
            sentinel: false,
        };
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--runs" => o.runs = it.next().context("--runs needs a number")?.parse()?,
                "--all" => o.all = true,
                "--barriers" => o.barriers = true,
                "--card" => o.card_only = true,
                "--warp" => o.warp = true,
                "--diagnose" => o.diagnose = true,
                "--reader" => o.reader = true,
                "--sentinel" => o.sentinel = true,
                "--retranslate" => {
                    o.retranslate = Some(PathBuf::from(
                        it.next().context("--retranslate needs a file")?,
                    ))
                }
                "--only" => {
                    o.only = it
                        .next()
                        .context("--only needs hash prefixes")?
                        .split(',')
                        .map(|h| h.trim().to_ascii_lowercase())
                        .filter(|h| !h.is_empty())
                        .collect()
                }
                "--cuts" => {
                    o.cuts = Some(Cuts::parse(
                        it.next().context("--cuts needs stages, each or a list")?,
                    )?)
                }
                "--dump" => {
                    o.dump = Some(PathBuf::from(it.next().context("--dump needs a folder")?))
                }
                s if s.starts_with("--") => bail!("unknown option {s}"),
                s => o.capture = PathBuf::from(s),
            }
        }
        if o.capture.as_os_str().is_empty() {
            bail!(
                "usage: bvh-harness <capture folder> [--runs N] [--all] [--barriers] [--card] [--warp] [--dump <folder>] [--diagnose] [--cuts stages|each|k,k,..] [--retranslate <dll> [--only h,h]] [--reader [--sentinel]]"
            );
        }
        Ok(o)
    }
}

/// A capture with its page bytes at hand.
pub struct Loaded {
    pub capture: Capture,
    pre: std::fs::File,
    post: std::fs::File,
    /// Pipelines retranslated with the production thread limit
    /// (`VICTORY_DISPATCHX_THREADS`, Fix C): pipeline index -> (the limit
    /// entry's user-data dword, the program's group size in x).
    pub thread_limits: HashMap<u32, (u32, u32)>,
}

impl Loaded {
    pub fn open(dir: &Path) -> anyhow::Result<Loaded> {
        let mut f = std::fs::File::open(dir.join("capture.bin"))
            .with_context(|| format!("{}", dir.display()))?;
        let capture = Capture::read(&mut std::io::BufReader::new(&mut f))?;
        Ok(Loaded {
            capture,
            pre: std::fs::File::open(dir.join("pre.pages"))?,
            post: std::fs::File::open(dir.join("post.pages"))?,
            thread_limits: HashMap::new(),
        })
    }

    /// The bytes of a page record (cut to its segment).
    pub fn bytes(&mut self, p: &PageRec) -> anyhow::Result<Vec<u8>> {
        let seg_len = self
            .capture
            .segments
            .get(p.seg as usize)
            .map_or(PAGE, Segment::len);
        let len = seg_len.saturating_sub(p.at_seg).min(PAGE) as usize;
        let mut v = vec![0u8; len];
        if !p.is_zero() {
            let f = if p.stage == Stage::Pre {
                &mut self.pre
            } else {
                &mut self.post
            };
            f.seek(SeekFrom::Start(p.file_offset))?;
            f.read_exact(&mut v)?;
        }
        Ok(v)
    }
}

/// Why a capture cannot be replayed (yet), if it cannot.
pub fn unsupported(c: &Capture, all: bool) -> Vec<String> {
    let mut why = BTreeSet::new();
    for d in c.dispatches.iter().filter(|d| all || d.builder) {
        // senior 115: tables recorded with their views are replayable, and
        // so are the view tables built from them.
        let tables: Vec<(u32, u32)> = d
            .args
            .iter()
            .filter_map(|a| match a {
                RootArg::Table { param, index, .. } => Some((*param, *index)),
                _ => None,
            })
            .collect();
        for a in &d.args {
            match a {
                RootArg::Constants { dwords, param } => {
                    if dwords.iter().enumerate().any(|(i, d)| {
                        d.kind == DwordKind::TableOffset && !tables.contains(&(*param, i as u32))
                    }) {
                        why.insert(
                            "descriptor tables (table offsets in root constants)".to_string(),
                        );
                    }
                }
                RootArg::Special { what, .. } => match what {
                    Special::MemoryTable | Special::MemoryDirty => {}
                    Special::ViewStrides
                    | Special::ViewAddresses
                    | Special::MarksTable
                    | Special::MarksBitmap
                        if !tables.is_empty() => {}
                    w => {
                        why.insert(format!("root parameter {w:?}"));
                    }
                },
                RootArg::View { .. } => {}
                RootArg::Table { views, slot, .. } => {
                    for v in views {
                        if let Some(w) = view_unusable(v) {
                            why.insert(format!(
                                "table slot {slot} view {}{}: {w}",
                                ["b", "t", "u"].get(usize::from(v.kind)).unwrap_or(&"?"),
                                v.register
                            ));
                        }
                    }
                }
            }
        }
        if let Groups::Indirect { arguments, .. } = &d.groups
            && arguments.first().is_some_and(|a| a[1] != 0)
        {
            why.insert("split indirect dispatch (indirect.rs not shared yet)".to_string());
        }
    }
    why.into_iter().collect()
}

/// Why a table view cannot be replayed: no host view (the capture could
/// not describe it), or fewer bytes than the host view spans.
pub fn view_unusable(v: &TableView) -> Option<String> {
    // No host view (an acceleration structure view such as the reader's
    // t355, read only for its address, or a slot the title left null, such
    // as t373 in the reader's s6 != 16 form): replayed as a null view with
    // its title address in the view addresses. A program that reads through
    // it reads zeros, as through a null view in the game.
    if v.count == 0 {
        return None;
    }
    if v.element() == 0 {
        return Some(format!("format {} of unknown size", v.format));
    }
    if (v.pre.len() as u64) < v.len() {
        return Some(format!("bytes {} of {}", v.pre.len(), v.len()));
    }
    None
}

/// senior 115: an unordered table view after the replay against the game's
/// bytes after the fence. `pre`: before the program ran (both sides
/// started there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewVerdict {
    /// Bytes compared, bytes that differ, largest byte difference.
    pub bytes: usize,
    pub differ: usize,
    pub most: u8,
    /// Bytes the game changed and the harness changed (pre -> after).
    pub game_changed: usize,
    pub harness_changed: usize,
    /// The first differing byte offsets.
    pub first: Vec<usize>,
}

/// [`ViewVerdict`] of `harness` against `game`, both from `pre`.
pub fn compare_view(pre: &[u8], game: &[u8], harness: &[u8]) -> ViewVerdict {
    let n = game.len().min(harness.len());
    let mut v = ViewVerdict {
        bytes: n,
        differ: game.len().max(harness.len()) - n,
        most: 0,
        game_changed: 0,
        harness_changed: 0,
        first: Vec::new(),
    };
    for i in 0..n {
        let (g, h) = (game[i], harness[i]);
        if g != h {
            v.differ += 1;
            v.most = v.most.max(g.abs_diff(h));
            if v.first.len() < 16 {
                v.first.push(i);
            }
        }
        if pre.get(i).is_some_and(|&p| p != g) {
            v.game_changed += 1;
        }
        if pre.get(i).is_some_and(|&p| p != h) {
            v.harness_changed += 1;
        }
    }
    v
}

/// senior 122: the byte a `--sentinel` run puts in every unordered table
/// view before the program runs.
pub const SENTINEL: u8 = 0xA5;

/// senior 122: [`compare_view`] for a `--sentinel` run. `harness` started as
/// [`SENTINEL`] bytes: a byte still [`SENTINEL`] was not written and stands
/// for `pre` (what the game's memory held); `harness_changed` counts the
/// bytes the harness wrote, even with the value `pre` had. (A byte the
/// harness writes as 0xA5 counts as not written.)
pub fn compare_view_sentinel(pre: &[u8], game: &[u8], harness: &[u8]) -> ViewVerdict {
    let written = harness.iter().filter(|&&h| h != SENTINEL).count();
    let seen: Vec<u8> = harness
        .iter()
        .enumerate()
        .map(|(i, &h)| if h == SENTINEL { pre.get(i).copied().unwrap_or(SENTINEL) } else { h })
        .collect();
    ViewVerdict { harness_changed: written, ..compare_view(pre, game, &seen) }
}

/// The verdict over all views: "BYTE-EQUAL", "WITHIN 1" (every byte at most
/// 1 apart: unorm rounding of another GPU), else "DIFFERENT"; "NO VIEW"
/// without any unordered view with game bytes; senior 122: "NOT TESTED"
/// when neither the game nor the harness changed a byte (equal bytes then
/// prove nothing: a program that does nothing would pass too).
pub fn reader_verdict(views: &[ViewVerdict]) -> &'static str {
    if views.is_empty() {
        "NO VIEW"
    } else if views.iter().all(|v| v.game_changed == 0 && v.harness_changed == 0) {
        "NOT TESTED"
    } else if views.iter().all(|v| v.differ == 0) {
        "BYTE-EQUAL"
    } else if views.iter().all(|v| v.most <= 1 && v.bytes > 0) {
        "WITHIN 1"
    } else {
        "DIFFERENT"
    }
}

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    sent: u64,
    event: HANDLE,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

fn buffer_desc(size: u64, uav: bool) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    }
}

impl Gpu {
    fn new(warp: bool) -> anyhow::Result<Gpu> {
        // SAFETY: plain object creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            // senior 115: `VICTORY_HARNESS_DEBUG=1`: the D3D12 debug layer,
            // its messages printed when a run fails ([`Gpu::messages`]).
            if std::env::var_os("VICTORY_HARNESS_DEBUG").is_some() {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok()
                    && let Some(debug) = debug
                {
                    debug.EnableDebugLayer();
                    println!("harness: D3D12 debug layer on");
                }
            }
            if warp {
                let factory: IDXGIFactory4 = CreateDXGIFactory1()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            } else {
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            }
            let device = device.context("no device")?;
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Gpu {
                device,
                queue,
                allocator,
                list,
                fence,
                sent: 0,
                event,
            })
        }
    }

    fn buffer(
        &self,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        uav: bool,
    ) -> anyhow::Result<ID3D12Resource> {
        let heap = if kind == D3D12_HEAP_TYPE_CUSTOM {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            }
        } else {
            D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            }
        };
        let state = match kind {
            D3D12_HEAP_TYPE_UPLOAD => D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_HEAP_TYPE_READBACK => D3D12_RESOURCE_STATE_COPY_DEST,
            _ => D3D12_RESOURCE_STATE_COMMON,
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: a plain buffer description.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, uav),
                state,
                None,
                &mut out,
            )?
        };
        out.context("no buffer")
    }

    fn map(b: &ID3D12Resource) -> anyhow::Result<*mut u8> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a mappable buffer.
        unsafe { b.Map(0, None, Some(&mut p))? };
        Ok(p.cast())
    }

    /// The debug layer's stored messages (`VICTORY_HARNESS_DEBUG`), printed.
    fn messages(&self) {
        let Ok(q) = self.device.cast::<ID3D12InfoQueue>() else { return };
        // SAFETY: plain queries into buffers of the size asked for.
        unsafe {
            for i in 0..q.GetNumStoredMessages().min(64) {
                let mut len = 0usize;
                if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
                    continue;
                }
                let mut buf = vec![0u64; len.div_ceil(8)];
                let m = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if q.GetMessage(i, Some(m), &mut len).is_ok() {
                    let d = &*m;
                    let text = std::slice::from_raw_parts(d.pDescription, d.DescriptionByteLength.saturating_sub(1));
                    println!("d3d12 debug: {}", String::from_utf8_lossy(text));
                }
            }
        }
    }

    /// Records with `f` and runs it, waiting.
    fn run(
        &mut self,
        f: impl FnOnce(&ID3D12GraphicsCommandList) -> anyhow::Result<()>,
    ) -> anyhow::Result<std::time::Duration> {
        // SAFETY: the last run finished (each is waited for).
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            f(&self.list)?;
            if let Err(e) = self.list.Close() {
                self.messages();
                return Err(e.into());
            }
            let started = std::time::Instant::now();
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.sent += 1;
            self.queue.Signal(&self.fence, self.sent)?;
            self.fence.SetEventOnCompletion(self.sent, self.event)?;
            WaitForSingleObject(self.event, INFINITE);
            let took = started.elapsed();
            if let Err(e) = self.device.GetDeviceRemovedReason() {
                bail!("device removed during the harness run: {e}");
            }
            Ok(took)
        }
    }
}

/// A segment's buffer in the harness.
struct Seg {
    buffer: ID3D12Resource,
    gpu: u64,
    /// Its view pair's first slot.
    pair: u32,
}

/// One run's result: the page bitmap's set bits, and the bytes of every
/// page the bitmap marked written (title address of the page, bytes).
pub struct RunResult {
    pub bitmap: BTreeSet<u32>,
    pub pages: BTreeMap<u64, Vec<u8>>,
    /// senior 115: the unordered table views' bytes after the run, by
    /// (host view address, bytes).
    pub views: BTreeMap<(u64, u64), Vec<u8>>,
    pub gpu_ms: f64,
}

/// Runs the capture once on `gpu`.
fn run_once(gpu: &mut Gpu, l: &mut Loaded, o: &Options) -> anyhow::Result<RunResult> {
    run_prefix(gpu, l, o, usize::MAX, &BTreeSet::new())
}

/// Runs the first `upto` dispatches of the capture once on `gpu` and reads
/// back the marked pages and the captured ones of `also`.
fn run_prefix(
    gpu: &mut Gpu,
    l: &mut Loaded,
    o: &Options,
    upto: usize,
    also: &BTreeSet<u64>,
) -> anyhow::Result<RunResult> {
    let c = l.capture.clone();
    let limits = l.thread_limits.clone();
    let pre: Vec<PageRec> = c.pages_of(Stage::Pre).cloned().collect();
    let captured: BTreeSet<u64> = pre.iter().map(|p| p.address & !(PAGE - 1)).collect();
    // Segments to make: those with captured pages, root views, indirect arguments.
    let mut wanted: BTreeSet<u32> = pre.iter().map(|p| p.seg).collect();
    let seg_of = |at: u64| -> Option<u32> {
        match c.table.entry_at(at)?.1 {
            Entry::Copy { seg } | Entry::Arena { seg, .. } => Some(seg),
            Entry::Null => None,
        }
    };
    for d in &c.dispatches {
        for a in &d.args {
            match a {
                RootArg::View { address, .. } => wanted.extend(seg_of(*address)),
                RootArg::Constants { dwords, .. } => {
                    for x in dwords.iter().filter(|x| x.address != 0) {
                        wanted.extend(seg_of(x.address));
                    }
                }
                RootArg::Special { .. } | RootArg::Table { .. } => {}
            }
        }
        if let Groups::Indirect { args, .. } = &d.groups {
            wanted.extend(seg_of(*args));
        }
    }
    // senior 115: tables of the replayed dispatches (reader captures): a
    // heap range each after the segments' pairs, entry k of a table at its
    // base + k.
    let replayed: Vec<(usize, Dispatch)> = c
        .dispatches
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, d)| o.all || d.builder)
        .take(upto)
        .collect();
    let table_slots: u32 = replayed
        .iter()
        .flat_map(|(_, d)| d.args.iter())
        .map(|a| match a {
            RootArg::Table { views, .. } => views.iter().map(|v| v.dw / 8 + 1).max().unwrap_or(1),
            _ => 0,
        })
        .sum();
    // Heap: null pair, a pair per segment, the tables.
    let slots = 2 + 2 * wanted.len() as u32 + 16 + table_slots;
    // SAFETY: plain descriptor heap creation.
    let heap: ID3D12DescriptorHeap = unsafe {
        gpu.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: slots,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).context("descriptor heap")?
    };
    // SAFETY: as above.
    let samplers: ID3D12DescriptorHeap = unsafe {
        gpu.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 16,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).context("descriptor heap")?
    };
    // SAFETY: plain queries.
    let (step, cpu0) = unsafe {
        (
            gpu.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            heap.GetCPUDescriptorHandleForHeapStart(),
        )
    };
    let cpu = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: cpu0.ptr + (i * step) as usize,
    };
    let raw_view = |i: u32, b: Option<&ID3D12Resource>, bytes: u64, uav: bool| {
        let elements = if b.is_some() {
            (bytes / 4).clamp(1, MOST_ELEMENTS) as u32
        } else {
            1
        };
        // SAFETY: a slot of our heap; a live buffer or none.
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                gpu.device
                    .CreateUnorderedAccessView(b, None, Some(&desc), cpu(i));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                gpu.device.CreateShaderResourceView(b, Some(&desc), cpu(i));
            }
        }
    };
    raw_view(0, None, 0, false);
    raw_view(1, None, 0, true);
    let mut segs: HashMap<u32, Seg> = HashMap::new();
    let mut next = 2;
    for &s in &wanted {
        let Some(seg) = c.segments.get(s as usize) else {
            continue;
        };
        let len = seg.len().next_multiple_of(4);
        let kind = match seg {
            Segment::Copy { card: false, .. } if !o.card_only => D3D12_HEAP_TYPE_CUSTOM,
            _ => D3D12_HEAP_TYPE_DEFAULT,
        };
        let buffer = gpu
            .buffer(len, kind, true)
            .with_context(|| format!("segment {s} ({len} bytes)"))?;
        raw_view(next, Some(&buffer), len, false);
        raw_view(next + 1, Some(&buffer), len, true);
        // SAFETY: a live buffer.
        let address = unsafe { buffer.GetGPUVirtualAddress() };
        segs.insert(
            s,
            Seg {
                buffer,
                gpu: address,
                pair: next,
            },
        );
        next += 2;
    }
    // Poison, then the captured pages, through an upload buffer.
    const STAGE: u64 = 64 << 20;
    let upload = gpu.buffer(STAGE, D3D12_HEAP_TYPE_UPLOAD, false)?;
    let up = Gpu::map(&upload)?;
    // Table views (senior 115): a buffer per host range (shared by the
    // dispatches naming it, so that one dispatch's writes reach the next),
    // filled with the bytes before the program's first list; the views as
    // the game's host made them, over the buffer from its byte 0.
    let mut tbufs: BTreeMap<(u64, u64), (ID3D12Resource, bool)> = BTreeMap::new();
    let mut sentinel_views = 0usize;
    let mut table_base: HashMap<(usize, usize), u32> = HashMap::new();
    let mut heap_views: Vec<(u32, TableView)> = Vec::new();
    for (dk, d) in &replayed {
        for (ak, a) in d.args.iter().enumerate() {
            let RootArg::Table { views, .. } = a else { continue };
            let base = next;
            let n = views.iter().map(|v| v.dw / 8 + 1).max().unwrap_or(1);
            for k in 0..n {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            NumElements: 1,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                // SAFETY: a slot of our heap; a null view.
                unsafe { gpu.device.CreateShaderResourceView(None, Some(&desc), cpu(base + k)) };
            }
            next += n;
            table_base.insert((*dk, ak), base);
            for v in views {
                if v.count == 0 {
                    // Address only: the slot stays null.
                    heap_views.push((base + v.dw / 8, v.clone()));
                    continue;
                }
                let key = (v.address, v.len());
                if !tbufs.contains_key(&key) {
                    let len = v.len().next_multiple_of(4).max(4);
                    let b = gpu
                        .buffer(len, D3D12_HEAP_TYPE_DEFAULT, true)
                        .with_context(|| format!("table view {:#x} ({len} bytes)", v.address))?;
                    let n = (v.pre.len() as u64).min(v.len());
                    if n > STAGE {
                        bail!("table view {:#x}: {n} bytes past the upload stage", v.address);
                    }
                    // senior 122, `--sentinel`: unordered views start as
                    // SENTINEL bytes (a range a read view named first keeps
                    // the game's bytes; counted in `sentinel_views`).
                    let sentinel = vec![SENTINEL; if o.sentinel && v.kind == 2 { n as usize } else { 0 }];
                    let fill = if sentinel.is_empty() { &v.pre[..n as usize] } else { &sentinel[..] };
                    sentinel_views += usize::from(!sentinel.is_empty());
                    // SAFETY: inside the mapped upload buffer; the GPU is done with it.
                    unsafe { std::ptr::copy_nonoverlapping(fill.as_ptr(), up, n as usize) };
                    gpu.run(|list| {
                        // SAFETY: live buffers, ranges inside them.
                        unsafe { list.CopyBufferRegion(&b, 0, &upload, 0, n) };
                        Ok(())
                    }).context("harness run: table view fill")?;
                    tbufs.insert(key, (b, false));
                }
                if v.kind == 2
                    && let Some(t) = tbufs.get_mut(&key)
                {
                    t.1 = true;
                }
                if v.kind == INDIRECT_ARGUMENTS {
                    // Read by ExecuteIndirect, bound nowhere.
                    continue;
                }
                let at = cpu(base + v.dw / 8);
                let b = &tbufs[&key].0;
                let format = if v.raw {
                    DXGI_FORMAT_R32_TYPELESS
                } else {
                    DXGI_FORMAT(v.format as i32)
                };
                // SAFETY: a slot of our heap, a live buffer the view lies in.
                unsafe {
                    if v.kind == 2 {
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    FirstElement: 0,
                                    NumElements: v.count,
                                    StructureByteStride: v.stride,
                                    CounterOffsetInBytes: 0,
                                    Flags: if v.raw {
                                        D3D12_BUFFER_UAV_FLAG_RAW
                                    } else {
                                        D3D12_BUFFER_UAV_FLAG_NONE
                                    },
                                },
                            },
                        };
                        gpu.device.CreateUnorderedAccessView(b, None, Some(&desc), at);
                    } else {
                        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    FirstElement: 0,
                                    NumElements: v.count,
                                    StructureByteStride: v.stride,
                                    Flags: if v.raw {
                                        D3D12_BUFFER_SRV_FLAG_RAW
                                    } else {
                                        D3D12_BUFFER_SRV_FLAG_NONE
                                    },
                                },
                            },
                        };
                        gpu.device.CreateShaderResourceView(b, Some(&desc), at);
                    }
                }
                heap_views.push((base + v.dw / 8, v.clone()));
            }
        }
    }
    // The view tables the game binds next to tables, as the game's host
    // fills them for these slots: strides, view addresses, write marks
    // (none: the harness reads its buffers back whole) and their bitmap.
    let bits_len = u64::from(victory_shader::emit::VIEW_WRITE_BITS / 8);
    let view_tables = if heap_views.is_empty() {
        None
    } else {
        let n = u64::from(victory_shader::emit::HEAP_VIEWS.max(next));
        let strides = gpu.buffer(4 * n, D3D12_HEAP_TYPE_UPLOAD, false).context("view strides")?;
        let addresses = gpu.buffer(16 * n, D3D12_HEAP_TYPE_UPLOAD, false).context("view addresses")?;
        let marks = gpu.buffer(8 * n, D3D12_HEAP_TYPE_UPLOAD, false).context("view marks")?;
        let bits = gpu.buffer(bits_len, D3D12_HEAP_TYPE_DEFAULT, true).context("view write bits")?;
        let (sp, ap, mp) = (Gpu::map(&strides)?, Gpu::map(&addresses)?, Gpu::map(&marks)?);
        // SAFETY: mapped upload buffers of n entries each.
        unsafe {
            let st = std::slice::from_raw_parts_mut(sp.cast::<u32>(), n as usize);
            let ad = std::slice::from_raw_parts_mut(ap.cast::<u32>(), 4 * n as usize);
            let mk = std::slice::from_raw_parts_mut(mp.cast::<u32>(), 2 * n as usize);
            st.fill(0);
            ad.fill(0);
            for e in mk.chunks_mut(2) {
                e[0] = victory_shader::emit::VIEW_MARK_NONE;
                e[1] = 0;
            }
            // The segments' raw pairs.
            for i in 2..(2 + 2 * segs.len()) {
                st[i] = 4;
            }
            for (i, v) in &heap_views {
                let i = *i as usize;
                st[i] = v.stride_word;
                ad[4 * i] = v.title as u32;
                ad[4 * i + 1] = ((v.title >> 32) & 0xffff) as u32;
            }
            std::ptr::write_bytes(up, 0, bits_len as usize);
        }
        gpu.run(|list| {
            // SAFETY: live buffers.
            unsafe { list.CopyBufferRegion(&bits, 0, &upload, 0, bits_len) };
            Ok(())
        }).context("harness run: view bitmap fill")?;
        Some((strides, addresses, marks, bits))
    };
    let fill = |words: &mut [u32]| words.fill(POISON);
    // SAFETY: the mapped upload buffer, STAGE bytes.
    fill(unsafe { std::slice::from_raw_parts_mut(up.cast::<u32>(), (STAGE / 4) as usize) });
    let seg_list: Vec<(u32, u64)> = segs
        .keys()
        .map(|&s| (s, c.segments[s as usize].len().next_multiple_of(4)))
        .collect();
    for (s, len) in &seg_list {
        let b = segs[s].buffer.clone();
        let mut at = 0;
        while at < *len {
            let n = (*len - at).min(STAGE);
            gpu.run(|list| {
                // SAFETY: live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&b, at, &upload, 0, n) };
                Ok(())
            }).context("harness run: segment poison")?;
            at += n;
        }
    }
    // Pages in batches of the stage size.
    let mut batch: Vec<(u32, u64, u64)> = Vec::new();
    let mut used = 0u64;
    let flush = |gpu: &mut Gpu, batch: &mut Vec<(u32, u64, u64)>| -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let jobs = std::mem::take(batch);
        gpu.run(|list| {
            for &(s, at_seg, len_off) in &jobs {
                let (len, off) = (len_off >> 32, len_off & 0xffff_ffff);
                // SAFETY: live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&segs[&s].buffer, at_seg, &upload, off, len) };
            }
            Ok(())
        }).context("harness run: page upload")?;
        Ok(())
    };
    for p in &pre {
        if !segs.contains_key(&p.seg) {
            continue;
        }
        let bytes = l.bytes(p)?;
        if used + PAGE > STAGE {
            flush(gpu, &mut batch)?;
            used = 0;
        }
        // SAFETY: inside the mapped upload buffer; the GPU is done with it.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), up.add(used as usize), bytes.len())
        };
        batch.push((p.seg, p.at_seg, (bytes.len() as u64) << 32 | used));
        used += PAGE;
    }
    flush(gpu, &mut batch)?;
    // The page table (in upload memory: read as a root SRV).
    let table = gpu.buffer((TABLE_DWORDS * 4) as u64, D3D12_HEAP_TYPE_UPLOAD, false)?;
    let tp = Gpu::map(&table)?.cast::<u32>();
    // SAFETY: the mapped table, TABLE_DWORDS dwords.
    let t = unsafe { std::slice::from_raw_parts_mut(tp, TABLE_DWORDS) };
    t.fill(0);
    for (b, &l1) in c.table.blocks.iter().enumerate().skip(1) {
        t[l1 as usize] = b as u32;
    }
    let mut missing_seg = 0;
    for (i, e) in c.table.entries.iter().enumerate() {
        let slot = LEVEL1 + 2 * i;
        let named = c.table.address_of(i).filter(|at| captured.contains(at));
        let v = match (*e, named) {
            (Entry::Copy { seg }, Some(_)) => {
                match (segs.get(&seg), c.segments.get(seg as usize)) {
                    (Some(s), Some(Segment::Copy { start, .. })) => Some((s.pair, *start as u32)),
                    _ => None,
                }
            }
            (Entry::Arena { seg, offset }, Some(at)) => segs
                .get(&seg)
                .map(|s| (s.pair, (at as u32).wrapping_sub(offset))),
            _ => None,
        };
        if v.is_none() && !matches!(e, Entry::Null) && named.is_some() {
            missing_seg += 1;
        }
        let (pair, lo) = v.unwrap_or((0, 0));
        t[slot] = pair;
        t[slot + 1] = lo;
    }
    if missing_seg > 0 {
        bail!("{missing_seg} captured pages without a segment buffer");
    }
    // The page bitmap, zeroed.
    let bitmap = gpu.buffer(BITMAP_BYTES, D3D12_HEAP_TYPE_DEFAULT, true)?;
    // SAFETY: a live buffer.
    let bitmap_gpu = unsafe { bitmap.GetGPUVirtualAddress() };
    // SAFETY: the mapped upload buffer.
    unsafe { std::ptr::write_bytes(up, 0, BITMAP_BYTES as usize) };
    gpu.run(|list| {
        // SAFETY: live buffers.
        unsafe { list.CopyBufferRegion(&bitmap, 0, &upload, 0, BITMAP_BYTES) };
        Ok(())
    }).context("harness run: page bitmap fill")?;
    // Pipelines.
    let mut psos: HashMap<u32, (ID3D12RootSignature, ID3D12PipelineState)> = HashMap::new();
    for d in c.dispatches.iter().filter(|d| o.all || d.builder) {
        if psos.contains_key(&d.pipeline) {
            continue;
        }
        let p = c
            .pipelines
            .get(d.pipeline as usize)
            .context("pipeline index")?;
        // SAFETY: serialized root signature and DXIL as the game made them.
        let pair = unsafe {
            let root: ID3D12RootSignature = gpu
                .device
                .CreateRootSignature(0, &p.root)
                .with_context(|| format!("root signature of {}", p.hash))?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: p.dxil.as_ptr().cast(),
                    BytecodeLength: p.dxil.len(),
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = gpu
                .device
                .CreateComputePipelineState(&desc)
                .with_context(|| format!("pipeline of {}", p.hash))?;
            (root, pso)
        };
        psos.insert(d.pipeline, pair);
    }
    // Indirect signatures by stride.
    let mut signatures: HashMap<u32, ID3D12CommandSignature> = HashMap::new();
    for d in &c.dispatches {
        if let Groups::Indirect { stride, .. } = d.groups
            && !signatures.contains_key(&stride)
        {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut s: Option<ID3D12CommandSignature> = None;
            // SAFETY: one argument, no root arguments.
            unsafe {
                gpu.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut s)?
            };
            signatures.insert(stride, s.context("no command signature")?);
        }
    }
    let gpu_of = |at: u64| -> Option<(u64, &Seg)> {
        let s = seg_of(at)?;
        let seg = segs.get(&s)?;
        let off = match c.segments.get(s as usize)? {
            Segment::Copy { start, .. } => at.checked_sub(*start)?,
            Segment::Arena { .. } => match c.table.entry_at(at)?.1 {
                Entry::Arena { offset, .. } => u64::from(offset) + (at & (PAGE - 1)),
                _ => return None,
            },
        };
        Some((seg.gpu + off, seg))
    };
    let uav_barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    let table_gpu = {
        // SAFETY: a live buffer.
        unsafe { table.GetGPUVirtualAddress() }
    };
    let took = gpu.run(|list| {
        // SAFETY: an open list; every object lives until the run is waited for.
        unsafe {
            list.SetDescriptorHeaps(&[Some(heap.clone()), Some(samplers.clone())]);
            let cb5 = cb5_table();
            for (dk, d) in c
                .dispatches
                .iter()
                .enumerate()
                .filter(|(_, d)| o.all || d.builder)
                .take(upto)
            {
                let (root, pso) = &psos[&d.pipeline];
                list.SetComputeRootSignature(root);
                list.SetPipelineState(pso);
                // Fix C (production draw::x_groups): the captures carry no
                // DispatchX flags; every direct dispatch of caa78b8c/28918b71
                // is a DispatchX with flags 0x20 (x in threads; PC 2 flag
                // log 27.11., 903/903 x = node count). Other dispatches of a
                // limited program pass no limit (u32::MAX).
                let limit = limits.get(&d.pipeline).map(|&(index, group)| match &d.groups {
                    Groups::Direct([x, ..]) => {
                        let (groups, threads) = crate::draw::x_groups(0x20, *x, Some(group));
                        (index as usize, groups, threads.unwrap_or(u32::MAX))
                    }
                    _ => (index as usize, 0, u32::MAX),
                });
                // Table offsets: the harness's heap ranges (senior 115).
                let overrides: Vec<(u32, u32, u32)> = d
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(ak, a)| match a {
                        RootArg::Table { param, index, .. } => {
                            table_base.get(&(dk, ak)).map(|&b| (*param, *index, b * 32))
                        }
                        _ => None,
                    })
                    .collect();
                for a in &d.args {
                    match a {
                        RootArg::Table { .. } => {}
                        RootArg::Constants { param, dwords } => {
                            let mut values = Vec::with_capacity(dwords.len());
                            for (xi, x) in dwords.iter().enumerate() {
                                values.push(match x.kind {
                                    DwordKind::RootViewSrv | DwordKind::RootViewUav => {
                                        let s = seg_of(x.address)
                                            .and_then(|s| segs.get(&s))
                                            .with_context(|| {
                                                format!(
                                                    "root view {:#x} outside the capture",
                                                    x.address
                                                )
                                            })?;
                                        s.pair + u32::from(x.kind == DwordKind::RootViewUav)
                                    }
                                    // Diag only (not Fix B's code): the title's
                                    // value from VICTORY_S82_CB5_FILE (senior 82's
                                    // reconstruction), else as captured (0).
                                    DwordKind::GroupXOffset => {
                                        cb5.get(&(dk + 1, xi)).copied().unwrap_or(0)
                                    }
                                    _ => x.value,
                                });
                            }
                            if let Some((index, _, threads)) = limit {
                                if values.len() <= index {
                                    values.resize(index + 1, 0);
                                }
                                values[index] = threads;
                            }
                            for &(p, index, value) in &overrides {
                                if p == *param
                                    && let Some(v) = values.get_mut(index as usize)
                                {
                                    *v = value;
                                }
                            }
                            list.SetComputeRoot32BitConstants(
                                *param,
                                values.len() as u32,
                                values.as_ptr().cast(),
                                0,
                            );
                        }
                        RootArg::View {
                            param,
                            kind,
                            address,
                        } => {
                            let (at, _) = gpu_of(*address).with_context(|| {
                                format!("root view {address:#x} outside the capture")
                            })?;
                            match kind {
                                0 => list.SetComputeRootConstantBufferView(*param, at),
                                1 => list.SetComputeRootShaderResourceView(*param, at),
                                _ => list.SetComputeRootUnorderedAccessView(*param, at),
                            }
                        }
                        RootArg::Special {
                            param,
                            what: Special::MemoryTable,
                        } => list.SetComputeRootShaderResourceView(*param, table_gpu),
                        RootArg::Special {
                            param,
                            what: Special::MemoryDirty,
                        } => list.SetComputeRootUnorderedAccessView(*param, bitmap_gpu),
                        RootArg::Special { param, what } => {
                            let Some((strides, addresses, marks, bits)) = &view_tables else {
                                bail!("root parameter {what:?} not replayable")
                            };
                            let (srv, buffer) = match what {
                                Special::ViewStrides => (true, strides),
                                Special::ViewAddresses => (true, addresses),
                                Special::MarksTable => (true, marks),
                                Special::MarksBitmap => (false, bits),
                                _ => bail!("root parameter {what:?} not replayable"),
                            };
                            if srv {
                                list.SetComputeRootShaderResourceView(*param, buffer.GetGPUVirtualAddress());
                            } else {
                                list.SetComputeRootUnorderedAccessView(*param, buffer.GetGPUVirtualAddress());
                            }
                        }
                    }
                }
                if d.barrier || o.barriers {
                    list.ResourceBarrier(std::slice::from_ref(&uav_barrier));
                }
                match &d.groups {
                    Groups::Direct([_, y, z]) if limit.is_some() => {
                        list.Dispatch(limit.map_or(0, |l| l.1), *y, *z)
                    }
                    Groups::Direct([x, y, z]) => list.Dispatch(*x, *y, *z),
                    Groups::Indirect {
                        stride,
                        args,
                        count,
                        most,
                        ..
                    } => {
                        let counter = (*count != 0).then(|| gpu_of(*count)).flatten();
                        // senior 115: arguments captured with their bytes.
                        let own = d.args.iter().find_map(|a| match a {
                            RootArg::Table { views, .. } => views
                                .iter()
                                .find(|v| v.kind == INDIRECT_ARGUMENTS && v.address == *args)
                                .and_then(|v| tbufs.get(&(v.address, v.len())))
                                .map(|(b, _)| b),
                            _ => None,
                        });
                        let (buffer, off) = match own {
                            Some(b) => (b, 0),
                            None => {
                                let (a, s) =
                                    gpu_of(*args).context("indirect arguments outside the capture")?;
                                (&s.buffer, a - s.gpu)
                            }
                        };
                        list.ExecuteIndirect(
                            &signatures[stride],
                            *most,
                            buffer,
                            off,
                            counter.map(|(_, cs)| &cs.buffer),
                            counter.map_or(0, |(a, cs)| a - cs.gpu),
                        );
                    }
                }
            }
        }
        Ok(())
    }).context("harness run: the dispatches")?;
    // Results: the bitmap, then every marked page's bytes.
    let readback = gpu.buffer(STAGE, D3D12_HEAP_TYPE_READBACK, false)?;
    let rb = Gpu::map(&readback)?;
    gpu.run(|list| {
        // SAFETY: live buffers.
        unsafe { list.CopyBufferRegion(&readback, 0, &bitmap, 0, BITMAP_BYTES) };
        Ok(())
    }).context("harness run: bitmap readback")?;
    let mut set = BTreeSet::new();
    // SAFETY: the mapped readback buffer, the GPU done with it.
    let words =
        unsafe { std::slice::from_raw_parts(rb.cast::<u32>(), (BITMAP_BYTES / 4) as usize) };
    for (w, &m) in words.iter().enumerate() {
        let mut bits = m;
        while bits != 0 {
            set.insert(w as u32 * 32 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }
    let marked: Vec<u64> = set
        .iter()
        .filter_map(|&e| c.table.address_of(e as usize))
        .chain(also.iter().copied())
        .filter(|a| captured.contains(a))
        .collect::<BTreeSet<u64>>()
        .into_iter()
        .collect();
    let mut pages = BTreeMap::new();
    for chunk in marked.chunks((STAGE / PAGE) as usize) {
        let mut jobs = Vec::new();
        for (k, &at) in chunk.iter().enumerate() {
            if let Some((g, s)) = gpu_of(at) {
                let seg_len = c.segments[seg_of(at).unwrap_or(0) as usize].len();
                let off = g - s.gpu;
                let len = match c.segments[seg_of(at).unwrap_or(0) as usize] {
                    Segment::Copy { .. } => {
                        seg_len.next_multiple_of(4).saturating_sub(off).min(PAGE)
                    }
                    Segment::Arena { .. } => PAGE,
                };
                jobs.push((k, at, s.buffer.clone(), off, len));
            }
        }
        gpu.run(|list| {
            for (k, _, b, off, len) in &jobs {
                // SAFETY: live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&readback, *k as u64 * PAGE, b, *off, *len) };
            }
            Ok(())
        }).context("harness run: page readback")?;
        for (k, at, _, _, len) in jobs {
            // SAFETY: the mapped readback buffer.
            let bytes =
                unsafe { std::slice::from_raw_parts(rb.add(k * PAGE as usize), len as usize) }
                    .to_vec();
            pages.insert(at, bytes);
        }
    }
    if o.sentinel {
        let uavs = tbufs.values().filter(|t| t.1).count();
        println!("sentinel: {sentinel_views} unordered table buffers started as {SENTINEL:#04x}, {} with the game's bytes", uavs.saturating_sub(sentinel_views));
    }
    // Unordered table views (reader captures).
    let mut view_bytes = BTreeMap::new();
    for (&key, (b, uav)) in &tbufs {
        if !*uav {
            continue;
        }
        let len = key.1;
        if len > STAGE {
            bail!("table view {:#x} of {len} bytes: past the readback stage", key.0);
        }
        gpu.run(|list| {
            // SAFETY: live buffers, ranges inside them.
            unsafe { list.CopyBufferRegion(&readback, 0, b, 0, len) };
            Ok(())
        }).context("harness run: table view readback")?;
        // SAFETY: the mapped readback buffer, the GPU done with it.
        view_bytes.insert(key, unsafe { std::slice::from_raw_parts(rb, len as usize) }.to_vec());
    }
    Ok(RunResult {
        views: view_bytes,
        bitmap: set,
        pages,
        gpu_ms: took.as_secs_f64() * 1000.0,
    })
}

/// senior 115, `--reader`: runs a reader capture `o.runs` times and
/// compares every unordered table view with the game's bytes after the
/// fence (all from the bytes before the program's first list). Exit 0: the
/// game's bytes reproduced (byte-equal, or within 1 with a note), 1: not,
/// 2: the capture cannot stand for the game (accesses outside it, or
/// incomplete before the run).
fn reader_main(gpu: &mut Gpu, l: &mut Loaded, o: &Options, incomplete: bool) -> anyhow::Result<i32> {
    let c = l.capture.clone();
    // The game's views: bytes before (first dispatch naming the range) and
    // after the fence (the last one).
    let mut game: BTreeMap<(u64, u64), (String, Vec<u8>, Vec<u8>)> = BTreeMap::new();
    for d in c.dispatches.iter().filter(|d| o.all || d.builder) {
        for a in &d.args {
            let RootArg::Table { views, .. } = a else { continue };
            for v in views.iter().filter(|v| v.kind == 2 && !v.post.is_empty()) {
                let e = game
                    .entry((v.address, v.len()))
                    .or_insert_with(|| (format!("u{}", v.register), v.pre.clone(), Vec::new()));
                e.2 = v.post.clone();
            }
        }
    }
    let runs: BTreeSet<u32> = c.dispatches.iter().filter(|d| o.all || d.builder).map(|d| d.list).collect();
    let game_bits: BTreeSet<u32> = c
        .bitmaps
        .iter()
        .filter(|(list, _)| runs.contains(list))
        .flat_map(|(_, s)| s.iter().copied())
        .collect();
    println!(
        "reader replay: dispatches {}, unordered views with game bytes {}, page bitmap bits in the game {}",
        c.dispatches.iter().filter(|d| o.all || d.builder).count(),
        game.len(),
        game_bits.len()
    );
    let mut first: Option<BTreeMap<(u64, u64), Vec<u8>>> = None;
    let mut deterministic = true;
    let mut worst = Vec::new();
    let mut outside_any = 0usize;
    for k in 0..o.runs.max(1) {
        let r = run_once(gpu, l, o)?;
        let out = outside(&c, &r.bitmap, &game_bits);
        outside_any += out.len();
        let mut verdicts = Vec::new();
        for (key, (name, pre, post)) in &game {
            let Some(h) = r.views.get(key) else {
                println!("run {k}: {name} at {:#x}: no harness bytes", key.0);
                continue;
            };
            let h = &h[..post.len().min(h.len())];
            let v = if o.sentinel { compare_view_sentinel(pre, post, h) } else { compare_view(pre, post, h) };
            println!(
                "run {k}: {name} at {:#x} ({} bytes): differ {} (most {}), changed by the game {}, by the harness {}, first differing {:?}, {:.1} ms",
                key.0, v.bytes, v.differ, v.most, v.game_changed, v.harness_changed, v.first, r.gpu_ms
            );
            verdicts.push(v);
        }
        println!(
            "run {k}: accesses outside the capture {} (first: {})",
            out.len(),
            out.iter().take(8).map(|&a| outside_text(a)).collect::<Vec<_>>().join(", ")
        );
        match &first {
            None => first = Some(r.views.clone()),
            Some(f) => deterministic &= *f == r.views,
        }
        worst = verdicts;
    }
    let verdict = reader_verdict(&worst);
    println!("determinism across {} runs: {}", o.runs.max(1), if deterministic { "same bytes" } else { "BYTES DIFFER" });
    let code = if incomplete || outside_any > 0 {
        println!(
            "RESULT: CAPTURE INVALID (incomplete before the run {incomplete}, accesses outside {outside_any}); output {verdict}"
        );
        2
    } else {
        println!("RESULT: {verdict}");
        match verdict {
            "BYTE-EQUAL" | "WITHIN 1" => 0,
            "NOT TESTED" => 3,
            _ => 1,
        }
    };
    Ok(code)
}

/// The validator over `pages` (title address of the page, bytes): headers
/// by address with kind, validity and class of reason.
pub fn validate(pages: &BTreeMap<u64, Vec<u8>>) -> BTreeMap<u64, (&'static str, bool, String)> {
    let list: Vec<(usize, &[u8])> = pages.iter().map(|(&a, b)| (a as usize, &b[..])).collect();
    let scan = crate::bvh_gt::scan(&list);
    let read = crate::bvh_gt::reader(&list);
    scan.headers
        .iter()
        .map(|(at, c)| {
            (
                *at,
                (
                    crate::bvh_gt::header_kind(&read, *at),
                    c.valid,
                    crate::bvh_gt::why_class(c.why.as_deref()),
                ),
            )
        })
        .collect()
}

/// "valid by kind" as the game's log line writes it.
pub fn by_kind(v: &BTreeMap<u64, (&'static str, bool, String)>) -> String {
    let mut k: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
    for (kind, valid, _) in v.values() {
        let e = k.entry(kind).or_default();
        e.0 += 1;
        e.1 += u32::from(*valid);
    }
    k.iter()
        .map(|(k, (n, v))| format!("{k} {v}/{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn hash8(b: &[u8]) -> u64 {
    u64::from_le_bytes(blake3::hash(b).as_bytes()[..8].try_into().unwrap_or([0; 8]))
}

/// The harness: loads, checks, runs, compares. Returns the exit code
/// (0 reproduced, 1 not reproduced, 2 cannot replay).
pub fn main(args: &[String]) -> anyhow::Result<i32> {
    let started = std::time::Instant::now();
    let o = Options::parse(args)?;
    let mut l = Loaded::open(&o.capture)?;
    let c = l.capture.clone();
    println!(
        "capture {}: {}",
        o.capture.display(),
        c.meta("submission").unwrap_or("?")
    );
    // T4: translate only, no device (32 lanes, as NVIDIA): the fresh DXIL
    // next to the captured one for a diff (`VICTORY_RETRANSLATE_OUT`).
    if let (Some(file), Ok(_)) = (&o.retranslate, std::env::var("VICTORY_RETRANSLATE_OUT")) {
        retranslate(&mut l, file, &o.only, 32, 1_000_000)?;
        return Ok(0);
    }
    let mut gpu = Gpu::new(o.warp)?;
    println!("harness gpu: {}", super::adapter_text(&gpu.device));
    {
        // T4: the wave sizes the device runs (a pipeline that asks for
        // WaveSize(32) outside them is refused with E_INVALIDARG).
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: a plain feature query into a struct of its size.
        let ok = unsafe {
            gpu.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&mut o1 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS1).cast(),
                std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
            )
        };
        println!(
            "harness gpu wave lanes: {} to {} (query {})",
            o1.WaveLaneCountMin,
            o1.WaveLaneCountMax,
            if ok.is_ok() { "ok" } else { "failed" }
        );
        if let Some(file) = &o.retranslate {
            let lanes = if ok.is_ok() { o1.WaveLaneCountMax } else { 32 };
            // The view heap size the game's host translates for (NVIDIA:
            // 1,000,000 slots), as `crate::host` finds it.
            let views =
                victory_shader::emit::host::Host::from_device(gpu.device.clone()).view_heap_limit();
            retranslate(&mut l, file, &o.only, lanes, views)?;
        }
    }
    println!("capture gpu: {}", c.meta("gpu").unwrap_or("?"));
    let why = unsupported(&c, o.all);
    if !why.is_empty() {
        println!("CANNOT REPLAY: {}", why.join("; "));
        return Ok(2);
    }
    println!(
        "dispatches {} (builder {}), pages pre {}, post {}, title writes between snapshot and fence {}, other commands in the lists {}",
        c.dispatches.len(),
        c.dispatches.iter().filter(|d| d.builder).count(),
        c.pages_of(Stage::Pre).count(),
        c.pages_of(Stage::Post).count(),
        c.cpu_dirty.len(),
        c.others.len()
    );
    // Completeness is a hard gate (an incomplete capture's verdict is
    // CAPTURE INVALID, never "valid in the harness").
    let before = c.incomplete_before_run();
    println!(
        "completeness before the run: pages left out {}, marked pages with a source not taken before {} (first: {}){}",
        c.left_out()
            .map_or("unknown".to_string(), |n| n.to_string()),
        c.marked_not_taken(),
        c.marked_not_taken_pages()
            .iter()
            .take(8)
            .map(|a| format!("{a:#x} {:?}", c.table.entry_at(*a).map(|(_, e)| e)))
            .collect::<Vec<_>>()
            .join(", "),
        if before.is_empty() {
            String::new()
        } else {
            format!(" -> {}", before.join("; "))
        }
    );
    if !before.is_empty() && !o.diagnose {
        println!(
            "RESULT: CAPTURE INVALID ({}); --diagnose runs it anyway",
            before.join("; ")
        );
        return Ok(2);
    }
    if o.reader {
        return reader_main(&mut gpu, &mut l, &o, !before.is_empty());
    }
    // The game's G and bitmap.
    let mut g: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    for p in c.pages_of(Stage::Post).cloned().collect::<Vec<_>>() {
        g.insert(p.address & !(PAGE - 1), l.bytes(&p)?);
    }
    // The bitmaps of the runs the harness runs: those with the builder's
    // dispatches (or all with `--all`).
    let runs: BTreeSet<u32> = c
        .dispatches
        .iter()
        .filter(|d| o.all || d.builder)
        .map(|d| d.list)
        .collect();
    let game_bits: BTreeSet<u32> = c
        .bitmaps
        .iter()
        .filter(|(list, _)| runs.contains(list))
        .flat_map(|(_, s)| s.iter().copied())
        .collect();
    println!(
        "runs replayed {:?}; others in the capture: {}",
        runs,
        c.dispatches
            .iter()
            .filter(|d| !runs.contains(&d.list))
            .count()
    );
    let gv = validate(&g);
    println!("G (game, all marked pages): valid by kind {}", by_kind(&gv));
    // As the game's GT line counts: arena pages first (64), then copies.
    let mut order: Vec<u64> = Vec::new();
    let (mut arena, mut owned) = (Vec::new(), Vec::new());
    for &e in &game_bits {
        let Some(at) = c.table.address_of(e as usize) else {
            continue;
        };
        match c.table.entries.get(e as usize) {
            Some(Entry::Arena { .. }) => arena.push(at),
            Some(Entry::Copy { .. }) => owned.push(at),
            _ => {}
        }
    }
    order.extend(arena.iter().take(64));
    order.extend(owned.iter().take(64usize.saturating_sub(order.len())));
    let g64: BTreeMap<u64, Vec<u8>> = order
        .iter()
        .filter_map(|a| g.get(a).map(|b| (*a, b.clone())))
        .collect();
    println!(
        "G (as the game's GT line, 64 pages): valid by kind {} (compare with 'BVH GT s{} format:' in the run log)",
        by_kind(&validate(&g64)),
        c.meta("submission").unwrap_or("?")
    );
    if let Some(cuts) = &o.cuts {
        return stages(&mut gpu, &mut l, &o, cuts, &g, &gv, &game_bits);
    }
    let mut results = Vec::new();
    for k in 0..o.runs.max(1) {
        let r = run_once(&mut gpu, &mut l, &o)?;
        println!(
            "run {k}: gpu {:.1} ms, bitmap bits {}, marked pages {}",
            r.gpu_ms,
            r.bitmap.len(),
            r.pages.len()
        );
        results.push(r);
    }
    let first = &results[0];
    // Self-check: the bitmap.
    let extra: Vec<u32> = first.bitmap.difference(&game_bits).copied().collect();
    let lacking: Vec<u32> = game_bits.difference(&first.bitmap).copied().collect();
    // Accesses outside the capture, per run (A2 step 1: the same set in
    // every run -> a computation; changing -> a race or undefined input).
    let outs: Vec<BTreeSet<u64>> = results
        .iter()
        .map(|r| outside(&c, &r.bitmap, &game_bits))
        .collect();
    let out_first = outs[0].len();
    let self_ok = extra.is_empty() && lacking.is_empty();
    println!(
        "self-check bitmap: {} (harness only {}, of them outside the capture {}; game only {}){}",
        if self_ok { "EQUAL" } else { "DIFFERS" },
        extra.len(),
        out_first,
        lacking.len(),
        if out_first > 0 {
            " -> capture incomplete: fix the capture/harness first"
        } else {
            ""
        }
    );
    // What the accesses outside the capture were: entry kinds, first addresses.
    let outside_first: Vec<String> = extra
        .iter()
        .take(8)
        .map(|&e| match c.table.address_of(e as usize) {
            Some(at) => format!(
                "{at:#x} {:?}",
                c.table.entries.get(e as usize).copied().unwrap_or_default()
            ),
            None => format!("null block +{:#x}", u64::from(e) << 16),
        })
        .collect();
    if !outside_first.is_empty() {
        println!("harness only, first: {}", outside_first.join(", "));
    }
    for (k, o) in outs.iter().enumerate() {
        println!(
            "run {k}: outside the capture {}: {}",
            o.len(),
            o.iter()
                .take(64)
                .map(|&a| outside_text(a))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    let union: BTreeSet<u64> = outs.iter().flatten().copied().collect();
    let common: BTreeSet<u64> = union
        .iter()
        .copied()
        .filter(|a| outs.iter().all(|o| o.contains(a)))
        .collect();
    println!(
        "outside the capture over {} runs: {} (union {}, in every run {})",
        outs.len(),
        if outs.iter().all(|o| *o == outs[0]) {
            "IDENTICAL"
        } else {
            "VARYING"
        },
        union.len(),
        common.len()
    );
    // Determinism.
    let all_pages: BTreeSet<u64> = results
        .iter()
        .flat_map(|r| r.pages.keys().copied())
        .collect();
    let mut stable = BTreeSet::new();
    let mut unstable = Vec::new();
    for &at in &all_pages {
        let hashes: BTreeSet<Option<u64>> = results
            .iter()
            .map(|r| r.pages.get(&at).map(|b| hash8(b)))
            .collect();
        if hashes.len() == 1 {
            stable.insert(at);
        } else {
            unstable.push(at);
        }
    }
    println!(
        "determinism over {} runs: pages stable {}, unstable {} (first: {})",
        results.len(),
        stable.len(),
        unstable.len(),
        unstable
            .iter()
            .take(5)
            .map(|a| format!("{a:#x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // Validator on the harness's result, against G.
    let hv = validate(&first.pages);
    println!("harness: valid by kind {}", by_kind(&hv));
    let invalid = |v: &BTreeMap<u64, (&'static str, bool, String)>| -> BTreeSet<(u64, String)> {
        v.iter()
            .filter(|(_, (k, ok, _))| *k == "bottom" && !ok)
            .map(|(a, (_, _, w))| (*a, w.clone()))
            .collect()
    };
    let (gi, hi) = (invalid(&gv), invalid(&hv));
    let head = |v: &BTreeSet<(u64, String)>| {
        v.iter()
            .take(3)
            .map(|(a, w)| format!("{a:#x} {w}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("invalid bottom-level in G, first: {}", head(&gi));
    println!("invalid bottom-level in the harness, first: {}", head(&hi));
    let same_invalid = gi == hi;
    println!(
        "invalid bottom-level structures by address and class: G {}, harness {}, both {} -> {}",
        gi.len(),
        hi.len(),
        gi.intersection(&hi).count(),
        if same_invalid { "SAME" } else { "DIFFERENT" }
    );
    // Bytes on stable pages, but those runs the harness does not replay
    // also wrote (G holds their writes too).
    let foreign: BTreeSet<u64> = c
        .bitmaps
        .iter()
        .filter(|(list, _)| !runs.contains(list))
        .flat_map(|(_, s)| s.iter().filter_map(|&e| c.table.address_of(e as usize)))
        .collect();
    let (mut equal, mut differ) = (0, Vec::new());
    let mut skipped = 0;
    for &at in &stable {
        if foreign.contains(&at) {
            skipped += 1;
            continue;
        }
        match (first.pages.get(&at), g.get(&at)) {
            (Some(h), Some(gb)) if h == gb => equal += 1,
            (Some(_), Some(_)) => differ.push(at),
            _ => {}
        }
    }
    println!(
        "bytes on stable pages against G: equal {equal}, different {} (first: {}); left out, also written by runs not replayed: {skipped}",
        differ.len(),
        differ
            .iter()
            .take(5)
            .map(|a| format!("{a:#x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if let Some(dir) = &o.dump {
        std::fs::create_dir_all(dir)?;
        for (k, r) in results.iter().enumerate() {
            let mut index = String::new();
            let mut bytes = Vec::new();
            for (at, b) in &r.pages {
                index.push_str(&format!("{at:#x} {} {}\n", bytes.len(), b.len()));
                bytes.extend_from_slice(b);
            }
            std::fs::write(dir.join(format!("run{k}.index")), index)?;
            std::fs::write(dir.join(format!("run{k}.pages")), bytes)?;
        }
    }
    let invalid = incomplete(c.left_out(), c.marked_not_taken(), out_first);
    if !invalid.is_empty() {
        println!(
            "RESULT: CAPTURE INVALID ({}) ({:.1} s)",
            invalid.join("; "),
            started.elapsed().as_secs_f64()
        );
        return Ok(2);
    }
    let reproduced = self_ok && same_invalid && differ.is_empty();
    println!(
        "RESULT: {} ({:.1} s)",
        if reproduced {
            "REPRODUCED (self-check equal, same invalid structures, bytes equal on stable pages)"
        } else if self_ok && hi.is_empty() {
            "VALID IN HARNESS (self-check equal): an environment hypothesis, not a harness fault"
        } else {
            "NOT REPRODUCED"
        },
        started.elapsed().as_secs_f64()
    );
    Ok(if reproduced { 0 } else { 1 })
}

/// What one structure's check says, with each invariant on its own (T4):
/// version 6 header, root at +48, bounds, the walk (children in their
/// parent's box and inside the size), the parent table.
pub fn invariants(read: &dyn Fn(u64) -> Option<u32>, at: u64) -> String {
    let Some(h) = (0..16)
        .map(|i| read(at + 4 * i))
        .collect::<Option<Vec<u32>>>()
    else {
        return "header not read".into();
    };
    let c = crate::bvh_gt::check(read, at);
    let v6 = crate::bvh_gt::is_header(h[0], h[1]);
    let bytes = u64::from(h[1]) * 64 + 64;
    let root = h[12];
    let rat = u64::from(root & !7) << 3;
    let root_ok = !(root == u32::MAX || root & 7 == 7 || rat == 0 || rat >= bytes);
    let f = |i: usize| f32::from_bits(h[i]);
    let bounds_ok = (4..10).all(|i| f(i).is_finite() && f(i).abs() != f32::MAX)
        && (0..3).all(|a| f(4 + a) <= f(7 + a));
    let mark = |b: bool| if b { "ok" } else { "BAD" };
    format!(
        "{} v6 {} root {} ({root:#x}) bounds {} boxes {} outside {} missing {} parents {}/{}/{} hdr [{:08x} {:08x} {:08x}] root children [{}] why: {}",
        if c.valid { "VALID" } else { "INVALID" },
        mark(v6),
        mark(root_ok),
        mark(bounds_ok),
        c.boxes,
        c.outside,
        c.missing,
        c.parents.0,
        c.parents.1,
        c.parents.2,
        h[0],
        h[1],
        h[3],
        (0..4)
            .map(|i| read(at + rat + 4 * i).map_or("-".to_string(), |w| format!("{w:08x}")))
            .collect::<Vec<_>>()
            .join(" "),
        c.why.as_deref().unwrap_or("-"),
    )
}

/// Stage dumps (T4): for each cut, `runs` replays of the first k builder
/// dispatches; the structures G has (and any new header) checked after it,
/// invariant by invariant, the pages that changed against the cut before,
/// and the pages that differ across the runs.
fn stages(
    gpu: &mut Gpu,
    l: &mut Loaded,
    o: &Options,
    cuts: &Cuts,
    g: &BTreeMap<u64, Vec<u8>>,
    gv: &BTreeMap<u64, (&'static str, bool, String)>,
    game_bits: &BTreeSet<u32>,
) -> anyhow::Result<i32> {
    let c = l.capture.clone();
    let order: Vec<&Dispatch> = c.dispatches.iter().filter(|d| o.all || d.builder).collect();
    let pipes: Vec<u32> = order.iter().map(|d| d.pipeline).collect();
    let hash = |p: u32| {
        c.pipelines
            .get(p as usize)
            .map_or("?".to_string(), |p| p.hash.chars().take(8).collect())
    };
    let points = cuts.points(&pipes);
    // Pages read back at every cut: those the game marked (G's pages).
    let also: BTreeSet<u64> = game_bits
        .iter()
        .filter_map(|&e| c.table.address_of(e as usize))
        .collect();
    let targets: Vec<u64> = gv.keys().copied().collect();
    println!(
        "stages: {} builder dispatches, cuts {}, runs per cut {}, barriers {}; G structures: {}",
        order.len(),
        points.len(),
        o.runs.max(1),
        o.barriers,
        gv.iter()
            .map(|(a, (k, ok, w))| format!(
                "{a:#x} {k} {} {w}",
                if *ok { "valid" } else { "invalid" }
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    {
        let list: Vec<(usize, &[u8])> = g.iter().map(|(&a, b)| (a as usize, &b[..])).collect();
        let read = crate::bvh_gt::reader(&list);
        for &t in &targets {
            println!("G {t:#x}: {}", invariants(&read, t));
        }
    }
    let mut before: BTreeMap<u64, u64> = BTreeMap::new();
    for (i, &k) in points.iter().enumerate() {
        let d = order[k - 1];
        let mut results = Vec::new();
        for _ in 0..o.runs.max(1) {
            results.push(run_prefix(gpu, l, o, k, &also)?);
        }
        let first = &results[0];
        let now: BTreeMap<u64, u64> = first.pages.iter().map(|(a, b)| (*a, hash8(b))).collect();
        let changed: Vec<u64> = now
            .iter()
            .filter(|(a, h)| before.get(a) != Some(h))
            .map(|(a, _)| *a)
            .collect();
        let unstable: Vec<u64> = now
            .keys()
            .copied()
            .filter(|a| {
                results
                    .iter()
                    .map(|r| r.pages.get(a).map(|b| hash8(b)))
                    .collect::<BTreeSet<_>>()
                    .len()
                    > 1
            })
            .collect();
        let hexes = |v: &[u64]| {
            v.iter()
                .take(6)
                .map(|a| format!("{a:#x}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "cut {i}: after dispatch {} of {} ({}, groups {:?}), bitmap bits {}, pages changed {} ({}), unstable over runs {} ({})",
            k,
            order.len(),
            hash(d.pipeline),
            d.groups,
            first.bitmap.len(),
            changed.len(),
            hexes(&changed),
            unstable.len(),
            hexes(&unstable),
        );
        let mut heads: BTreeSet<u64> = targets.iter().copied().collect();
        let lists: Vec<Vec<(usize, &[u8])>> = results
            .iter()
            .map(|res| {
                res.pages
                    .iter()
                    .map(|(&a, b)| (a as usize, &b[..]))
                    .collect()
            })
            .collect();
        for list in &lists {
            heads.extend(crate::bvh_gt::scan(list).headers.iter().map(|(a, _)| *a));
        }
        for (r, list) in lists.iter().enumerate() {
            let read = crate::bvh_gt::reader(list);
            for &t in &heads {
                println!("  run {r} {t:#x}: {}", invariants(&read, t));
            }
        }
        // The pages after this cut, every run (`--dump`): cut<k>-run<r>.
        if let Some(dir) = &o.dump {
            std::fs::create_dir_all(dir)?;
            for (r, res) in results.iter().enumerate() {
                let mut index = String::new();
                let mut bytes = Vec::new();
                for (at, b) in &res.pages {
                    index.push_str(&format!("{at:#x} {} {}\n", bytes.len(), b.len()));
                    bytes.extend_from_slice(b);
                }
                std::fs::write(dir.join(format!("cut{k}-run{r}.index")), index)?;
                std::fs::write(dir.join(format!("cut{k}-run{r}.pages")), bytes)?;
            }
        }
        before = now;
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::Cuts;

    #[test]
    fn stage_cuts_end_every_run_of_one_pipeline_and_the_last_cut_runs_all() {
        let p = [7, 7, 3, 3, 3, 7, 9];
        assert_eq!(Cuts::Stages.points(&p), vec![2, 5, 6, 7]);
        assert_eq!(Cuts::Each.points(&p), vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(Cuts::List(vec![4, 0, 99, 4]).points(&p), vec![1, 4, 7]);
    }

    use super::{compare_view, compare_view_sentinel, reader_verdict, unsupported, view_unusable, SENTINEL};
    use crate::bvh_replay::format::*;

    fn view(kind: u8, register: u32, dw: u32, bytes: usize) -> TableView {
        TableView {
            kind,
            register,
            dw,
            title: 0x1000,
            address: 0x1000,
            format: 61,
            count: bytes as u32,
            stride: 0,
            raw: false,
            stride_word: 1,
            pre: vec![255; bytes],
            post: Vec::new(),
        }
    }

    /// A reader dispatch: the table offset in root constant dword 1 of
    /// parameter 0, the table with `views`, the view tables beside it.
    fn reader(views: Vec<TableView>, covered: bool) -> Capture {
        let mut c = Capture::default();
        c.pipelines.push(Pipeline { name: "r".into(), hash: "f607780a".into(), ..Pipeline::default() });
        let mut args = vec![
            RootArg::Constants {
                param: 0,
                dwords: vec![
                    Dword::default(),
                    Dword { kind: DwordKind::TableOffset, value: 0x4000, address: 0 },
                ],
            },
            RootArg::Special { param: 1, what: Special::ViewStrides },
            RootArg::Special { param: 2, what: Special::MarksTable },
            RootArg::Special { param: 3, what: Special::MarksBitmap },
        ];
        if covered {
            args.push(RootArg::Table { param: 0, index: 1, slot: 5, value: 0x4000, views });
        }
        c.dispatches.push(Dispatch {
            list: 0,
            pipeline: 0,
            builder: true,
            barrier: false,
            groups: Groups::Direct([1, 1, 1]),
            args,
        });
        c
    }

    /// senior 115: a table recorded with its views makes the table offset
    /// and the view tables replayable; without it they stay refused, and a
    /// view with too few bytes or no host view refuses the capture.
    #[test]
    fn tables_with_their_views_are_replayable_and_only_whole_ones() {
        assert_eq!(unsupported(&reader(vec![view(2, 52, 48, 16)], true), false), Vec::<String>::new());
        let refused = unsupported(&reader(vec![], false), false);
        assert!(refused.iter().any(|w| w.contains("descriptor tables")), "{refused:?}");
        assert!(refused.iter().any(|w| w.contains("ViewStrides")), "{refused:?}");
        let mut short = view(1, 10, 8, 16);
        short.pre.truncate(15);
        let refused = unsupported(&reader(vec![short], true), false);
        assert!(refused.iter().any(|w| w.contains("t10") && w.contains("15 of 16")), "{refused:?}");
        let none = TableView { count: 0, title: 0, ..view(0, 3, 56, 0) };
        assert!(view_unusable(&none).is_none());
        let address_only = TableView { count: 0, format: 0, title: 0xd5f2_0cd00, ..view(1, 355, 32, 0) };
        assert!(view_unusable(&address_only).is_none());
        assert!(view_unusable(&TableView { format: 0, ..view(1, 1, 0, 4) }).is_some());
        assert!(view_unusable(&view(1, 1, 0, 4)).is_none());
    }

    /// senior 115: the u52 comparison counts every differing byte, the
    /// bytes each side changed, and the verdict follows the largest gap.
    #[test]
    fn reader_views_compare_byte_by_byte() {
        let pre = vec![255u8; 8];
        let mut game = pre.clone();
        game[2] = 127;
        game[5] = 127;
        let v = compare_view(&pre, &game, &game);
        assert_eq!((v.differ, v.game_changed, v.harness_changed), (0, 2, 2));
        assert_eq!(reader_verdict(&[v]), "BYTE-EQUAL");
        let mut near = game.clone();
        near[5] = 128;
        let v = compare_view(&pre, &game, &near);
        assert_eq!((v.differ, v.most, v.first.clone()), (1, 1, vec![5]));
        assert_eq!(reader_verdict(&[v]), "WITHIN 1");
        let v = compare_view(&pre, &game, &pre);
        assert_eq!((v.differ, v.most, v.harness_changed), (2, 128, 0));
        assert_eq!(reader_verdict(&[v]), "DIFFERENT");
        let v = compare_view(&pre, &game, &game[..7]);
        assert_eq!(v.differ, 1);
        assert_eq!(reader_verdict(&[]), "NO VIEW");
    }

    /// senior 122 (Fable audit 3.3): equal bytes where neither side changed
    /// anything are "NOT TESTED", never "BYTE-EQUAL" (onCap3: u52 before =
    /// after = 127 on the game side).
    #[test]
    fn reader_verdict_without_any_change_is_not_tested() {
        let pre = vec![127u8; 8];
        let v = compare_view(&pre, &pre, &pre);
        assert_eq!((v.differ, v.game_changed, v.harness_changed), (0, 0, 0));
        assert_eq!(reader_verdict(&[v.clone()]), "NOT TESTED");
        // One view with a change is enough to test.
        let mut game = pre.clone();
        game[3] = 0;
        let w = compare_view(&pre, &game, &game);
        assert_eq!(reader_verdict(&[v, w]), "BYTE-EQUAL");
    }

    /// senior 122: with the sentinel the harness's writes show even when it
    /// writes the value the game's memory held; an untouched sentinel byte
    /// stands for that value.
    #[test]
    fn sentinel_counts_writes_of_the_old_value() {
        // The game rewrote probes 2 and 5 with 127 (= what was there).
        let pre = vec![127u8; 8];
        let game = pre.clone();
        let mut h = vec![SENTINEL; 8];
        h[2] = 127;
        h[5] = 127;
        let v = compare_view_sentinel(&pre, &game, &h);
        assert_eq!((v.differ, v.game_changed, v.harness_changed), (0, 0, 2));
        assert_eq!(reader_verdict(&[v]), "BYTE-EQUAL");
        // A program that writes nothing is not tested, not equal.
        let v = compare_view_sentinel(&pre, &game, &[SENTINEL; 8]);
        assert_eq!((v.differ, v.harness_changed), (0, 0));
        assert_eq!(reader_verdict(&[v]), "NOT TESTED");
        // A wrong write shows as a difference against the game.
        h[5] = 0;
        let v = compare_view_sentinel(&pre, &game, &h);
        assert_eq!((v.differ, v.most, v.first.clone(), v.harness_changed), (1, 127, vec![5], 2));
        assert_eq!(reader_verdict(&[v]), "DIFFERENT");
        // The game wrote a byte the harness left alone: different.
        let mut game2 = pre.clone();
        game2[1] = 0;
        let v = compare_view_sentinel(&pre, &game2, &[SENTINEL; 8]);
        assert_eq!((v.differ, v.game_changed, v.harness_changed), (1, 1, 0));
        assert_eq!(reader_verdict(&[v]), "DIFFERENT");
    }
}

/// T4 coherence test: translates the pipelines whose shader hash starts
/// with one of `only` afresh from the containers in `file`, as the game
/// does for the BVH builder (write marks, loop limit 4096 with the arena,
/// the device's wave width), with the coherence diagnostics when
/// `VICTORY_SHADER_COHERENT=1`; prints whether the result is the captured
/// DXIL byte for byte, and replays with it.
fn retranslate(
    l: &mut Loaded,
    file: &Path,
    only: &[String],
    lanes: u32,
    views: u32,
) -> anyhow::Result<()> {
    // H-05 separation: 1/all (UAV, globally coherent, fence), uav, uavgc,
    // fence; unset or 0: off.
    let parts = match std::env::var("VICTORY_SHADER_COHERENT")
        .unwrap_or_default()
        .trim()
    {
        "" | "0" => 0,
        "1" | "all" => 7,
        "uav" => 1,
        "uavgc" => 3,
        "fence" => 4,
        v => bail!("VICTORY_SHADER_COHERENT={v}: 0, 1/all, uav, uavgc or fence"),
    };
    let coherent = format!(
        "{}{}{}",
        if parts & 1 != 0 { "U" } else { "-" },
        if parts & 2 != 0 { "G" } else { "-" },
        if parts & 4 != 0 { "F" } else { "-" }
    );
    let data = std::fs::read(file).with_context(|| format!("{}", file.display()))?;
    let mut found: HashMap<String, Vec<u8>> = HashMap::new();
    let mut at = 0;
    while let Some(k) = data[at..].windows(4).position(|w| w == b"DXBC") {
        let start = at + k;
        at = start + 4;
        let Some(total) = data
            .get(start + 0x18..start + 0x1c)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
        else {
            break;
        };
        let Some(bytes) = data.get(start..start + total) else {
            continue;
        };
        if total < 32 {
            continue;
        }
        let hash = blake3::hash(bytes).to_hex()[..16].to_string();
        if only.iter().any(|h| hash.starts_with(h.as_str())) {
            found.insert(hash, bytes.to_vec());
        }
    }
    victory_shader::emit::set_host_wave_max(lanes);
    crate::descriptor::host::host_views(views);
    victory_shader::emit::set_coherent_parts(parts);
    let options = victory_shader::emit::Options {
        view_marks: true,
        group_x_offset: false,
        loop_limit: crate::pipeline::builder_loop_limit().or(Some(4096)),
        thread_limit: false,
    };
    let mut done = 0;
    let mut limits: Vec<(u32, (u32, u32))> = Vec::new();
    for (pi, p) in l.capture.pipelines.iter_mut().enumerate() {
        if !only.iter().any(|h| p.hash.starts_with(h.as_str())) {
            continue;
        }
        let Some(container) = found.get(&p.hash) else {
            println!(
                "retranslate {}: container not found in {}",
                p.hash,
                file.display()
            );
            continue;
        };
        // Fix C as production chooses it (shader_cache::host_shader_with):
        // the variant from the hash list of VICTORY_DISPATCHX_THREADS, and
        // without the limit where it cannot be put before the program.
        let options = victory_shader::emit::Options {
            thread_limit: crate::shader_cache::dispatch_threads(container),
            ..options
        };
        let host = match victory_shader::emit::translate_with(container, options) {
            Err(error) if options.thread_limit => {
                println!(
                    "retranslate {}: VICTORY_DISPATCHX_THREADS: no thread limit for this program; translated without ({error})",
                    p.hash
                );
                victory_shader::emit::translate_with(
                    container,
                    victory_shader::emit::Options { thread_limit: false, ..options },
                )
            }
            made => made,
        }
        .map_err(|e| anyhow::anyhow!("translating {}: {e}", p.hash))?;
        // As pipeline::with_resources: the limit from the translation's own
        // entry and the program's group size in x.
        let group_x = victory_shader::Container::parse(container)
            .ok()
            .and_then(|c| {
                let part = c
                    .part(victory_shader::FourCC::DXIL)
                    .or(c.part(victory_shader::FourCC::ACCC))?;
                victory_shader::native_programs(part.data).ok()?.into_iter().find_map(|n| match &n.io {
                    victory_shader::native::StageIo::Cs(cs) => Some(cs.threads[0]),
                    _ => None,
                })
            });
        if let Some(group) = crate::pipeline::thread_limit_of(&host.user_data, group_x) {
            let entry = host
                .user_data
                .iter()
                .find(|u| victory_shader::emit::is_thread_limit(u))
                .expect("thread_limit_of found the entry");
            let captured = p
                .root
                .len();
            println!(
                "retranslate {}: thread limit: entry user-data dword {} (slot {}), group size {group}, user-data dwords {} (captured root signature {captured} bytes)",
                p.hash,
                entry.index,
                entry.slot,
                host.user_data.iter().map(|u| u.index + 1).max().unwrap_or(0),
            );
            limits.push((pi as u32, (entry.index, group)));
        } else if options.thread_limit {
            println!("retranslate {}: listed for the thread limit, translation has no limit entry", p.hash);
        }
        println!(
            "retranslate {}: coherent {coherent}, wave lanes {lanes}, view heap {views}, {} bytes, captured {} bytes, byte-equal to the captured DXIL: {}",
            p.hash,
            host.dxil.len(),
            p.dxil.len(),
            if host.dxil == p.dxil { "YES" } else { "NO" }
        );
        if let Ok(dir) = std::env::var("VICTORY_RETRANSLATE_OUT") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("{}-captured.dxil", p.hash)), &p.dxil)?;
            std::fs::write(
                dir.join(format!("{}-fresh-coherent-{coherent}.dxil", p.hash)),
                &host.dxil,
            )?;
        }
        p.dxil = host.dxil;
        done += 1;
    }
    l.thread_limits.extend(limits);
    victory_shader::emit::set_coherent(false);
    if done == 0 {
        bail!("no pipeline of {only:?} retranslated");
    }
    Ok(())
}

/// Diag (senior 82's Fix-B emulation, not production code): the title's
/// values of root-constant dwords old captures recorded zeroed as "group x
/// offset", from `VICTORY_S82_CB5_FILE`: lines `<dispatch number, 1-based>
/// <dword index> <value hex>`.
fn cb5_table() -> HashMap<(usize, usize), u32> {
    let Ok(path) = std::env::var("VICTORY_S82_CB5_FILE") else {
        return HashMap::new();
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let t: HashMap<(usize, usize), u32> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let k = it.next()?.parse().ok()?;
            let i = it.next()?.parse().ok()?;
            let v = u32::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
            Some(((k, i), v))
        })
        .collect();
    println!("diag: {} root-constant values from {path} (Fix-B emulation, senior 82)", t.len());
    t
}
