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
                "--dump" => {
                    o.dump = Some(PathBuf::from(it.next().context("--dump needs a folder")?))
                }
                s if s.starts_with("--") => bail!("unknown option {s}"),
                s => o.capture = PathBuf::from(s),
            }
        }
        if o.capture.as_os_str().is_empty() {
            bail!(
                "usage: bvh-harness <capture folder> [--runs N] [--all] [--barriers] [--card] [--warp] [--dump <folder>] [--diagnose]"
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
        for a in &d.args {
            match a {
                RootArg::Constants { dwords, .. } => {
                    if dwords.iter().any(|d| d.kind == DwordKind::TableOffset) {
                        why.insert(
                            "descriptor tables (table offsets in root constants)".to_string(),
                        );
                    }
                }
                RootArg::Special { what, .. } => match what {
                    Special::MemoryTable | Special::MemoryDirty => {}
                    w => {
                        why.insert(format!("root parameter {w:?}"));
                    }
                },
                RootArg::View { .. } => {}
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
            self.list.Close()?;
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
    pub gpu_ms: f64,
}

/// Runs the capture once on `gpu`.
fn run_once(gpu: &mut Gpu, l: &mut Loaded, o: &Options) -> anyhow::Result<RunResult> {
    let c = l.capture.clone();
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
                RootArg::Special { .. } => {}
            }
        }
        if let Groups::Indirect { args, .. } = &d.groups {
            wanted.extend(seg_of(*args));
        }
    }
    // Heap: null pair, a pair per segment.
    let slots = 2 + 2 * wanted.len() as u32 + 16;
    // SAFETY: plain descriptor heap creation.
    let heap: ID3D12DescriptorHeap = unsafe {
        gpu.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: slots,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
    };
    // SAFETY: as above.
    let samplers: ID3D12DescriptorHeap = unsafe {
        gpu.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 16,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
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
            })?;
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
        })?;
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
    })?;
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
            for d in c.dispatches.iter().filter(|d| o.all || d.builder) {
                let (root, pso) = &psos[&d.pipeline];
                list.SetComputeRootSignature(root);
                list.SetPipelineState(pso);
                for a in &d.args {
                    match a {
                        RootArg::Constants { param, dwords } => {
                            let mut values = Vec::with_capacity(dwords.len());
                            for x in dwords {
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
                                    DwordKind::GroupXOffset => 0,
                                    _ => x.value,
                                });
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
                        RootArg::Special { what, .. } => {
                            bail!("root parameter {what:?} not replayable")
                        }
                    }
                }
                if d.barrier || o.barriers {
                    list.ResourceBarrier(std::slice::from_ref(&uav_barrier));
                }
                match &d.groups {
                    Groups::Direct([x, y, z]) => list.Dispatch(*x, *y, *z),
                    Groups::Indirect {
                        stride,
                        args,
                        count,
                        most,
                        ..
                    } => {
                        let (_, s) =
                            gpu_of(*args).context("indirect arguments outside the capture")?;
                        let off = gpu_of(*args).map(|(a, s)| a - s.gpu).unwrap_or(0);
                        let counter = (*count != 0).then(|| gpu_of(*count)).flatten();
                        list.ExecuteIndirect(
                            &signatures[stride],
                            *most,
                            &s.buffer,
                            off,
                            counter.map(|(_, cs)| &cs.buffer),
                            counter.map_or(0, |(a, cs)| a - cs.gpu),
                        );
                    }
                }
            }
        }
        Ok(())
    })?;
    // Results: the bitmap, then every marked page's bytes.
    let readback = gpu.buffer(STAGE, D3D12_HEAP_TYPE_READBACK, false)?;
    let rb = Gpu::map(&readback)?;
    gpu.run(|list| {
        // SAFETY: live buffers.
        unsafe { list.CopyBufferRegion(&readback, 0, &bitmap, 0, BITMAP_BYTES) };
        Ok(())
    })?;
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
        .filter(|a| captured.contains(a))
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
        })?;
        for (k, at, _, _, len) in jobs {
            // SAFETY: the mapped readback buffer.
            let bytes =
                unsafe { std::slice::from_raw_parts(rb.add(k * PAGE as usize), len as usize) }
                    .to_vec();
            pages.insert(at, bytes);
        }
    }
    Ok(RunResult {
        bitmap: set,
        pages,
        gpu_ms: took.as_secs_f64() * 1000.0,
    })
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
    let mut gpu = Gpu::new(o.warp)?;
    println!("harness gpu: {}", super::adapter_text(&gpu.device));
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
        c.left_out().map_or("unknown".to_string(), |n| n.to_string()),
        c.marked_not_taken(),
        c.marked_not_taken_pages()
            .iter()
            .take(8)
            .map(|a| format!("{a:#x} {:?}", c.table.entry_at(*a).map(|(_, e)| e)))
            .collect::<Vec<_>>()
            .join(", "),
        if before.is_empty() { String::new() } else { format!(" -> {}", before.join("; ")) }
    );
    if !before.is_empty() && !o.diagnose {
        println!("RESULT: CAPTURE INVALID ({}); --diagnose runs it anyway", before.join("; "));
        return Ok(2);
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
    let misses = |v: &[u32]| {
        v.iter()
            .filter(|&&e| {
                (e as usize) < BLOCK
                    || c.table.address_of(e as usize).is_some_and(|a| {
                        !c.pages_of(Stage::Pre).any(|p| p.address & !(PAGE - 1) == a)
                    })
            })
            .count()
    };
    let self_ok = extra.is_empty() && lacking.is_empty();
    println!(
        "self-check bitmap: {} (harness only {}, of them outside the capture {}; game only {}){}",
        if self_ok { "EQUAL" } else { "DIFFERS" },
        extra.len(),
        misses(&extra),
        lacking.len(),
        if misses(&extra) > 0 {
            " -> capture incomplete: fix the capture/harness first"
        } else {
            ""
        }
    );
    // What the accesses outside the capture were: entry kinds, first addresses.
    let outside: Vec<String> = extra
        .iter()
        .take(8)
        .map(|&e| match c.table.address_of(e as usize) {
            Some(at) => format!(
                "{at:#x} {:?}",
                c.table.entries.get(e as usize).copied().unwrap_or_default()
            ),
            None => format!("region {:#x}", u64::from(e) << 28),
        })
        .collect();
    if !outside.is_empty() {
        println!("harness only, first: {}", outside.join(", "));
    }
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
    let invalid = incomplete(c.left_out(), c.marked_not_taken(), misses(&extra));
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
