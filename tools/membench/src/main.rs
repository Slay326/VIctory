//! Micro-benchmark of memory costs on the host GPU and in the OS, without
//! game data: GPU reads/writes/atomics per heap type, buffer copies between
//! heaps, CPU access to mapped heaps, and page-protection costs
//! (VirtualProtect, guard faults, write watch).
//!
//! Every number is measured: each test runs `REPS` times after `WARM`
//! warm-up runs and reports median, p10 and p90 over the repetitions.
//! GPU times come from timestamp queries; CPU times from `Instant`.
//!
//! Usage: membench [out.json]

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct3D::Fxc::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Diagnostics::Debug::*;
use windows::Win32::System::Memory::*;
use windows::Win32::System::Threading::*;
use windows::core::{Interface, PCSTR, s};

const REPS: usize = 15;
const WARM: usize = 2;
const KIB: u64 = 1024;
const MIB: u64 = 1024 * 1024;
const PAGE: usize = 4096;

// ---------------------------------------------------------------- results

#[derive(Clone, Copy)]
struct Stat {
    median: f64,
    p10: f64,
    p90: f64,
    n: usize,
}

fn stat(v: &[f64]) -> Stat {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let x = p * (s.len() - 1) as f64;
        let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
        s[lo] + (s[hi] - s[lo]) * (x - lo as f64)
    };
    Stat { median: q(0.5), p10: q(0.1), p90: q(0.9), n: s.len() }
}

struct Results {
    rows: Vec<(String, Stat, &'static str, String)>,
    info: Vec<(String, String)>,
}

impl Results {
    fn add(&mut self, key: impl Into<String>, values: &[f64], unit: &'static str, note: impl Into<String>) {
        let key = key.into();
        let note: String = note.into();
        let st = stat(values);
        println!(
            "| {:<40} | {:>12.4} | {:>12.4} | {:>12.4} | {:>3} | {:<6} | {}",
            key,
            st.median,
            st.p10,
            st.p90,
            st.n,
            unit,
            note
        );
        self.rows.push((key, st, unit, String::new()));
    }
    fn info(&mut self, k: &str, v: impl Into<String>) {
        let v = v.into();
        println!("# {k}: {v}");
        self.info.push((k.to_string(), v));
    }
    fn json(&self) -> String {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let mut o = String::from("{\n  \"info\": {\n");
        let info: Vec<String> = self.info.iter().map(|(k, v)| format!("    \"{}\": \"{}\"", esc(k), esc(v))).collect();
        o += &info.join(",\n");
        o += "\n  },\n  \"results\": {\n";
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|(k, s, u, _)| {
                format!(
                    "    \"{}\": {{\"median\": {:e}, \"p10\": {:e}, \"p90\": {:e}, \"n\": {}, \"unit\": \"{}\"}}",
                    esc(k),
                    s.median,
                    s.p10,
                    s.p90,
                    s.n,
                    u
                )
            })
            .collect();
        o += &rows.join(",\n");
        o += "\n  }\n}\n";
        o
    }
}


fn size_name(b: u64) -> String {
    if b >= MIB { format!("{}MiB", b / MIB) } else { format!("{}KiB", b / KIB) }
}

// ---------------------------------------------------------------- GPU

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    value: u64,
    qheap: ID3D12QueryHeap,
    qread: ID3D12Resource,
    qptr: *const u64,
    freq: f64,
}

const QUERIES: u32 = 256;

impl Gpu {
    fn new() -> windows::core::Result<Self> {
        // SAFETY: plain D3D12 object creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            let device = device.unwrap();
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let mut qheap: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: QUERIES, NodeMask: 0 },
                &mut qheap,
            )?;
            let qread = buffer(&device, props(D3D12_HEAP_TYPE_READBACK), QUERIES as u64 * 8, false, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let mut p = std::ptr::null_mut();
            qread.Map(0, None, Some(&mut p))?;
            let freq = queue.GetTimestampFrequency()? as f64;
            Ok(Gpu { device, queue, alloc, list, fence, event, value: 0, qheap: qheap.unwrap(), qread, qptr: p as *const u64, freq })
        }
    }

    fn begin(&self) {
        // SAFETY: the previous submission was waited for.
        unsafe {
            self.alloc.Reset().unwrap();
            self.list.Reset(&self.alloc, None).unwrap();
        }
    }

    fn ts(&self, i: u32) {
        // SAFETY: an open list; i < QUERIES.
        unsafe { self.list.EndQuery(&self.qheap, D3D12_QUERY_TYPE_TIMESTAMP, i) };
    }

    /// Closes, runs and waits; returns the CPU wall time in µs from
    /// ExecuteCommandLists to the fence's event.
    fn run(&mut self, queries: u32) -> f64 {
        // SAFETY: our own list, fence and event.
        unsafe {
            if queries > 0 {
                self.list.ResolveQueryData(&self.qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, queries, &self.qread, 0);
            }
            self.list.Close().unwrap();
            let l: ID3D12CommandList = self.list.cast().unwrap();
            self.value += 1;
            let t = Instant::now();
            self.queue.ExecuteCommandLists(&[Some(l)]);
            self.queue.Signal(&self.fence, self.value).unwrap();
            if self.fence.GetCompletedValue() < self.value {
                self.fence.SetEventOnCompletion(self.value, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
            }
            let us = t.elapsed().as_secs_f64() * 1e6;
            let removed = self.device.GetDeviceRemovedReason();
            assert!(removed.is_ok(), "device removed: {removed:?}");
            us
        }
    }

    /// µs between timestamps a and b of the last run.
    fn dt(&self, a: u32, b: u32) -> f64 {
        // SAFETY: the readback buffer stays mapped; the run was waited for.
        unsafe {
            let ta = *self.qptr.add(a as usize);
            let tb = *self.qptr.add(b as usize);
            (tb.wrapping_sub(ta)) as f64 / self.freq * 1e6
        }
    }

    fn uav_barrier(&self) {
        let all = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
            },
        };
        // SAFETY: an open list.
        unsafe { self.list.ResourceBarrier(&[all]) };
    }
}

fn props(t: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() }
}

fn custom_wb_l0() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    }
}

fn buffer_desc(size: u64, uav: bool, cross: bool) -> D3D12_RESOURCE_DESC {
    let mut flags = D3D12_RESOURCE_FLAG_NONE;
    if uav {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if cross {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    size: u64,
    uav: bool,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, uav, false), state, None, &mut r)?;
    }
    Ok(r.unwrap())
}

const HLSL: &str = r#"
#define RS "SRV(t0), UAV(u0), UAV(u1), RootConstants(num32BitConstants=4, b0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer sink : register(u1);
cbuffer c : register(b0) { uint total; uint chunkShift; uint chunkMask; uint mode; }
#define ITER 8
// Element (16 bytes) index -> element index. mode 0: sequential; mode 1:
// chunks of (1 << chunkShift) elements in a permuted order (odd multiplier
// mod a power of two is a bijection).
uint map(uint e) {
    if (mode == 0) return e;
    uint chunk = e >> chunkShift;
    uint within = e & ((1u << chunkShift) - 1u);
    return (((chunk * 2654435761u) & chunkMask) << chunkShift) | within;
}
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void read_main(uint3 id : SV_DispatchThreadID) {
    uint4 acc = 0;
    [unroll] for (uint k = 0; k < ITER; k++) {
        acc ^= src.Load4(map(id.x + k * total) * 16);
    }
    if (acc.x == 0x9e3779b9u && acc.y == 1u) sink.Store(0, acc.z);
}
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void write_main(uint3 id : SV_DispatchThreadID) {
    [unroll] for (uint k = 0; k < ITER; k++) {
        uint e = id.x + k * total;
        dst.Store4(map(e) * 16, uint4(e, e, e, e));
    }
}
// mode 0: random dword over the buffer (chunkMask = dwords - 1);
// mode 1: 64 hot dwords.
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void atomic_main(uint3 id : SV_DispatchThreadID) {
    [unroll] for (uint k = 0; k < ITER; k++) {
        uint h = (id.x + k * total) * 2654435761u;
        h ^= h >> 15;
        uint a = h & chunkMask;
        if (mode == 1) a &= 63u;
        dst.InterlockedAdd(a * 4, 1u);
    }
}
"#;

struct Kernels {
    root: ID3D12RootSignature,
    read: ID3D12PipelineState,
    write: ID3D12PipelineState,
    atomic: ID3D12PipelineState,
}

fn kernels(device: &ID3D12Device) -> Kernels {
    let compile = |entry: PCSTR| -> Vec<u8> {
        let (mut code, mut errors) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            let made = D3DCompile(HLSL.as_ptr().cast(), HLSL.len(), s!("membench"), None, None, entry, s!("cs_5_1"), D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &mut code, Some(&mut errors));
            if made.is_err() {
                let e: ID3DBlob = errors.unwrap();
                panic!("{}", String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())));
            }
            let code: ID3DBlob = code.unwrap();
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()).to_vec()
        }
    };
    let r = compile(s!("read_main"));
    let w = compile(s!("write_main"));
    let a = compile(s!("atomic_main"));
    // SAFETY: compiled shaders with their root signature inside.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &r).unwrap();
        let pso = |bytes: &[u8]| -> ID3D12PipelineState {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc).unwrap()
        };
        Kernels { read: pso(&r), write: pso(&w), atomic: pso(&a), root: root.clone() }
    }
}

struct Heap {
    name: &'static str,
    res: ID3D12Resource,
    size: u64,
    writable: bool,
    _keep: Option<ID3D12Heap>,
}

/// One dispatch per repetition, bracketed by timestamps; returns µs.
fn time_dispatch(
    gpu: &mut Gpu,
    k: &Kernels,
    pso: &ID3D12PipelineState,
    src: Option<&ID3D12Resource>,
    dst: Option<&ID3D12Resource>,
    sink: &ID3D12Resource,
    consts: [u32; 4],
    groups: u32,
) -> Vec<f64> {
    gpu.begin();
    // SAFETY: an open list; resources live until the run is waited for.
    unsafe {
        let l = &gpu.list;
        l.SetComputeRootSignature(&k.root);
        l.SetPipelineState(pso);
        let any = src.or(dst).unwrap();
        l.SetComputeRootShaderResourceView(0, src.unwrap_or(sink).GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(1, dst.unwrap_or(sink).GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(2, sink.GetGPUVirtualAddress());
        l.SetComputeRoot32BitConstants(3, 4, consts.as_ptr().cast(), 0);
        let _ = any;
    }
    let n = (REPS + WARM) as u32;
    for i in 0..n {
        gpu.ts(2 * i);
        // SAFETY: an open list.
        unsafe { gpu.list.Dispatch(groups, 1, 1) };
        gpu.uav_barrier();
        gpu.ts(2 * i + 1);
    }
    gpu.run(2 * n);
    (WARM as u32..n).map(|i| gpu.dt(2 * i, 2 * i + 1)).collect()
}

fn gpu_access(gpu: &mut Gpu, k: &Kernels, heaps: &[Heap], sink: &ID3D12Resource, out: &mut Results) {
    const ITER: u64 = 8;
    for h in heaps {
        let elems = h.size / 16;
        let threads = elems / ITER;
        let groups = (threads / 256) as u32;
        let total = groups * 256;
        let bytes = total as u64 * ITER * 16;
        let gbps = |us: &[f64]| us.iter().map(|t| bytes as f64 / (t * 1e3)).collect::<Vec<f64>>();
        // Sequential and scattered reads/writes.
        let mut patterns: Vec<(String, u32, u32, u32)> = vec![("seq".into(), 0, 0, 0)];
        for chunk in [4 * KIB, 64 * KIB, MIB] {
            let shift = (chunk / 16).trailing_zeros();
            let nchunks = (total as u64 * ITER * 16 / chunk) as u32;
            patterns.push((format!("scat{}", size_name(chunk)), shift, nchunks - 1, 1));
        }
        for (pname, shift, mask, mode) in &patterns {
            let t = time_dispatch(gpu, k, &k.read, Some(&h.res), None, sink, [total, *shift, *mask, *mode], groups);
            out.add(format!("gpu_read_GBps_{}_{}", pname, h.name), &gbps(&t), "GBps", format!("{} MiB", bytes / MIB));
            if h.writable {
                let t = time_dispatch(gpu, k, &k.write, None, Some(&h.res), sink, [total, *shift, *mask, *mode], groups);
                out.add(format!("gpu_write_GBps_{}_{}", pname, h.name), &gbps(&t), "GBps", format!("{} MiB", bytes / MIB));
            }
        }
        if h.writable {
            // Atomics: fewer ops in system memory (keeps a dispatch far from TDR).
            let ops: u64 = if h.name == "default" { 16 << 20 } else { 1 << 18 };
            let agroups = (ops / ITER / 256) as u32;
            let atotal = agroups * 256;
            let dwords = (h.size / 4) as u32;
            for (mname, mode) in [("random", 0u32), ("hot64", 1u32)] {
                let t = time_dispatch(gpu, k, &k.atomic, None, Some(&h.res), sink, [atotal, 0, dwords - 1, mode], agroups);
                let rate: Vec<f64> = t.iter().map(|us| (atotal as u64 * ITER) as f64 / (us * 1e3)).collect();
                out.add(format!("gpu_atomic_Gops_{}_{}", mname, h.name), &rate, "Gop/s", format!("{} ops", atotal as u64 * ITER));
            }
        }
    }
}

/// Batches of CopyBufferRegion per repetition; per-copy µs and GB/s.
fn copies(gpu: &mut Gpu, pairs: &[(String, &ID3D12Resource, u64, &ID3D12Resource, u64)], out: &mut Results) {
    for (name, src, ssize, dst, dsize) in pairs {
        for chunk in [4 * KIB, 64 * KIB, MIB, 16 * MIB] {
            let span = (*ssize).min(*dsize).min(64 * MIB);
            let n = (span / chunk).min(2048).max(1);
            gpu.begin();
            let reps = (REPS + WARM) as u32;
            for r in 0..reps {
                gpu.ts(2 * r);
                for i in 0..n {
                    let off = (i * chunk) % span;
                    // SAFETY: an open list; both ranges inside the buffers.
                    unsafe { gpu.list.CopyBufferRegion(*dst, off, *src, off, chunk) };
                }
                gpu.ts(2 * r + 1);
            }
            gpu.run(2 * reps);
            let per: Vec<f64> = (WARM as u32..reps).map(|r| gpu.dt(2 * r, 2 * r + 1) / n as f64).collect();
            let gbps: Vec<f64> = per.iter().map(|us| chunk as f64 / (us * 1e3)).collect();
            let sn = size_name(chunk);
            out.add(format!("copy_{name}_us_{sn}"), &per, "us", format!("{n} copies/batch"));
            out.add(format!("copy_{name}_GBps_{sn}"), &gbps, "GBps", String::new());
        }
    }
}

// ---------------------------------------------------------------- CPU

fn cpu_rw(ptr: *mut u8, len: usize, name: &str, read: bool, write: bool, out: &mut Results) {
    let mut local = vec![0u8; len];
    if write {
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            // SAFETY: ptr..ptr+len is mapped and writable.
            unsafe { std::ptr::copy_nonoverlapping(local.as_ptr(), ptr, len) };
            v.push(len as f64 / (t.elapsed().as_secs_f64() * 1e9));
        }
        out.add(format!("cpu_write_GBps_{name}"), &v[WARM..], "GBps", format!("{} MiB memcpy", len as u64 / MIB));
    }
    if read {
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            // SAFETY: ptr..ptr+len is mapped and readable.
            unsafe { std::ptr::copy_nonoverlapping(ptr, local.as_mut_ptr(), len) };
            std::hint::black_box(&local);
            v.push(len as f64 / (t.elapsed().as_secs_f64() * 1e9));
        }
        out.add(format!("cpu_read_GBps_{name}"), &v[WARM..], "GBps", format!("{} MiB memcpy", len as u64 / MIB));
    }
}

fn cpu_mem(out: &mut Results) {
    let len = 256 * MIB as usize;
    let a: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
    let mut b = a.clone();
    let c = a.clone();
    for chunk in [4 * KIB as usize, 64 * KIB as usize, MIB as usize] {
        let n = len / chunk;
        let mut cp = Vec::new();
        let mut cm = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            for i in 0..n {
                let r = i * chunk..(i + 1) * chunk;
                b[r.clone()].copy_from_slice(&a[r]);
            }
            std::hint::black_box(&b);
            cp.push(len as f64 / (t.elapsed().as_secs_f64() * 1e9));
            let t = Instant::now();
            let mut eq = 0usize;
            for i in 0..n {
                let r = i * chunk..(i + 1) * chunk;
                eq += (std::hint::black_box(&a[r.clone()]) == &c[r]) as usize;
            }
            std::hint::black_box(eq);
            cm.push(len as f64 / (t.elapsed().as_secs_f64() * 1e9));
        }
        let sn = size_name(chunk as u64);
        out.add(format!("cpu_memcpy_GBps_{sn}"), &cp[WARM..], "GBps", "256 MiB working set".to_string());
        out.add(format!("cpu_memcmp_GBps_{sn}"), &cm[WARM..], "GBps", "256 MiB, equal".to_string());
    }
}

static REGION_LO: AtomicUsize = AtomicUsize::new(0);
static REGION_HI: AtomicUsize = AtomicUsize::new(0);
static FAULTS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the OS passes valid exception pointers.
    let rec = unsafe { &*(*info).ExceptionRecord };
    let code = rec.ExceptionCode;
    let addr = rec.ExceptionInformation[1];
    let inside = addr >= REGION_LO.load(Ordering::Relaxed) && addr < REGION_HI.load(Ordering::Relaxed);
    if !inside {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    if code == STATUS_GUARD_PAGE_VIOLATION {
        FAULTS.fetch_add(1, Ordering::Relaxed);
        return -1; // EXCEPTION_CONTINUE_EXECUTION; the guard is already gone.
    }
    if code == STATUS_ACCESS_VIOLATION {
        FAULTS.fetch_add(1, Ordering::Relaxed);
        let page = addr & !(PAGE - 1);
        let mut old = PAGE_PROTECTION_FLAGS(0);
        // SAFETY: a page of our own region.
        let ok = unsafe { VirtualProtect(page as *const c_void, PAGE, PAGE_READWRITE, &mut old) };
        return if ok.is_ok() { -1 } else { 0 };
    }
    0
}

fn pages(out: &mut Results) {
    let size = 64 * MIB as usize;
    let npages = size / PAGE;
    // SAFETY: plain allocations and protection changes of our own memory;
    // the handler only acts on faults inside the region.
    unsafe {
        let base = VirtualAlloc(None, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as *mut u8;
        assert!(!base.is_null());
        for i in 0..npages {
            *base.add(i * PAGE) = 1;
        }
        let mut old = PAGE_PROTECTION_FLAGS(0);
        // VirtualProtect, one page per call.
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            let calls = 1000;
            let t = Instant::now();
            for i in 0..calls {
                let p = base.add(i * PAGE) as *const c_void;
                VirtualProtect(p, PAGE, PAGE_READONLY, &mut old).unwrap();
                VirtualProtect(p, PAGE, PAGE_READWRITE, &mut old).unwrap();
            }
            v.push(t.elapsed().as_secs_f64() * 1e6 / (2 * calls) as f64);
        }
        out.add("vprotect_us_per_call", &v[WARM..], "us", "1 page, committed+touched".to_string());
        // VirtualProtect over 4096 pages per call.
        let span = 4096usize;
        let mut v = Vec::new();
        let mut vc = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            VirtualProtect(base as *const c_void, span * PAGE, PAGE_READONLY, &mut old).unwrap();
            VirtualProtect(base as *const c_void, span * PAGE, PAGE_READWRITE, &mut old).unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / 2.0;
            vc.push(us);
            v.push(us / span as f64);
        }
        out.add("vprotect_us_per_page", &v[WARM..], "us", "4096-page range per call".to_string());
        out.add("vprotect_us_4096pages_call", &vc[WARM..], "us", String::new());

        REGION_LO.store(base as usize, Ordering::Relaxed);
        REGION_HI.store(base as usize + size, Ordering::Relaxed);
        let vh = AddVectoredExceptionHandler(1, Some(handler));
        assert!(!vh.is_null());

        // Baseline: touching plain pages.
        let touch = |n: usize| {
            let t = Instant::now();
            for i in 0..n {
                std::ptr::write_volatile(base.add(i * PAGE), 2);
            }
            t.elapsed().as_secs_f64() * 1e6
        };
        let mut base_us = Vec::new();
        for _ in 0..REPS + WARM {
            base_us.push(touch(span) / span as f64);
        }
        let b = stat(&base_us[WARM..]).median;
        // Guard faults.
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            VirtualProtect(base as *const c_void, span * PAGE, PAGE_READWRITE | PAGE_GUARD, &mut old).unwrap();
            FAULTS.store(0, Ordering::Relaxed);
            let us = touch(span);
            assert_eq!(FAULTS.load(Ordering::Relaxed), span, "guard faults");
            v.push(us / span as f64 - b);
        }
        out.add("guard_fault_us", &v[WARM..], "us", "per fault, vectored handler, minus plain touch".to_string());
        // Read-only page + handler that makes it writable (write tracking).
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            VirtualProtect(base as *const c_void, span * PAGE, PAGE_READONLY, &mut old).unwrap();
            FAULTS.store(0, Ordering::Relaxed);
            let us = touch(span);
            assert_eq!(FAULTS.load(Ordering::Relaxed), span, "access faults");
            v.push(us / span as f64 - b);
        }
        out.add("ro_fault_reprotect_us", &v[WARM..], "us", "per fault: AV + VirtualProtect(RW) in handler".to_string());
        RemoveVectoredExceptionHandler(vh);
        let _ = VirtualFree(base as *mut c_void, 0, MEM_RELEASE);

        // Write watch.
        let ww = VirtualAlloc(None, size, MEM_RESERVE | MEM_COMMIT | VIRTUAL_ALLOCATION_TYPE(0x0020_0000), PAGE_READWRITE) as *mut u8;
        assert!(!ww.is_null());
        let mut addrs: Vec<*mut c_void> = vec![std::ptr::null_mut(); npages];
        let mut gran = 0u32;
        let mut get = |flags: u32| -> (f64, usize) {
            let mut count = npages;
            let t = Instant::now();
            let r = GetWriteWatch(flags, ww as *const c_void, size, Some(addrs.as_mut_ptr()), Some(&mut count), Some(&mut gran));
            let us = t.elapsed().as_secs_f64() * 1e6;
            assert_eq!(r, 0, "GetWriteWatch");
            (us, count)
        };
        ResetWriteWatch(ww as *const c_void, size);
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            let (us, c) = get(0);
            assert_eq!(c, 0);
            v.push(us);
        }
        out.add("gww_us_per_call", &v[WARM..], "us", "64 MiB region, 0 dirty".to_string());
        let empty = stat(&v[WARM..]).median;
        let mut vp = Vec::new();
        let mut vf = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            for i in 0..npages {
                std::ptr::write_volatile(ww.add(i * PAGE), 3);
            }
            vf.push(t.elapsed().as_secs_f64() * 1e6 / npages as f64);
            let (us, c) = get(1); // WRITE_WATCH_FLAG_RESET
            assert_eq!(c, npages);
            vp.push((us - empty) / npages as f64);
        }
        out.add("gww_us_per_page", &vp[WARM..], "us", format!("{npages} dirty pages, with reset"));
        out.add("ww_first_write_us_per_page", &vf[WARM..], "us", "touch after reset (incl. plain store)".to_string());
        let mut v = Vec::new();
        for _ in 0..REPS + WARM {
            let t = Instant::now();
            ResetWriteWatch(ww as *const c_void, size);
            v.push(t.elapsed().as_secs_f64() * 1e6);
        }
        out.add("resetww_us_per_call", &v[WARM..], "us", "64 MiB region, clean".to_string());
        let _ = VirtualFree(ww as *mut c_void, 0, MEM_RELEASE);
    }
}

// ---------------------------------------------------------------- main

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "membench.json".into());
    let mut out = Results { rows: Vec::new(), info: Vec::new() };
    let mut gpu = Gpu::new().expect("D3D12 device");
    // SAFETY: plain device and DXGI queries.
    unsafe {
        let luid = gpu.device.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).unwrap();
        let d = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&d.Description).trim_end_matches('\0').to_string();
        out.info("adapter", name);
        out.info("dedicated_video_MiB", format!("{}", d.DedicatedVideoMemory as u64 / MIB));
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
        let _ = gpu.device.CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, (&raw mut arch).cast(), size_of_val(&arch) as u32);
        out.info("uma", format!("{}", arch.UMA.as_bool()));
        out.info("cache_coherent_uma", format!("{}", arch.CacheCoherentUMA.as_bool()));
    }
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the query writes its own structure.
    let gpu_upload = unsafe {
        gpu.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32).is_ok()
            && o16.GPUUploadHeapSupported.as_bool()
    };
    out.info("gpu_upload_heap_supported", format!("{gpu_upload}"));
    out.info("timestamp_freq_hz", format!("{}", gpu.freq));
    out.info("reps", format!("{REPS} (+{WARM} warm-up)"));
    println!("| {:<40} | {:>12} | {:>12} | {:>12} | {:>3} | {:<6} | note", "key", "median", "p10", "p90", "n", "unit");

    let dev = gpu.device.clone();
    let k = kernels(&dev);
    let sink = buffer(&dev, props(D3D12_HEAP_TYPE_DEFAULT), 64 * KIB, true, D3D12_RESOURCE_STATE_COMMON).unwrap();
    let host_size = 64 * MIB;
    let mut heaps = vec![
        Heap { name: "default", res: buffer(&dev, props(D3D12_HEAP_TYPE_DEFAULT), 256 * MIB, true, D3D12_RESOURCE_STATE_COMMON).unwrap(), size: 256 * MIB, writable: true, _keep: None },
        Heap { name: "upload", res: buffer(&dev, props(D3D12_HEAP_TYPE_UPLOAD), host_size, false, D3D12_RESOURCE_STATE_GENERIC_READ).unwrap(), size: host_size, writable: false, _keep: None },
        Heap { name: "custom_wb_l0", res: buffer(&dev, custom_wb_l0(), host_size, true, D3D12_RESOURCE_STATE_COMMON).unwrap(), size: host_size, writable: true, _keep: None },
    ];
    if gpu_upload {
        match buffer(&dev, props(D3D12_HEAP_TYPE_GPU_UPLOAD), host_size, true, D3D12_RESOURCE_STATE_COMMON) {
            Ok(r) => heaps.push(Heap { name: "gpu_upload", res: r, size: host_size, writable: true, _keep: None }),
            Err(e) => out.info("gpu_upload_create", format!("failed: {e}")),
        }
    }
    // A host allocation opened as a heap.
    // SAFETY: our own committed allocation, kept alive for the whole run.
    let host_ptr = unsafe { VirtualAlloc(None, host_size as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let opened: windows::core::Result<(ID3D12Heap, ID3D12Resource)> = (|| unsafe {
        let d3: ID3D12Device3 = dev.cast()?;
        let heap: ID3D12Heap = d3.OpenExistingHeapFromAddress(host_ptr)?;
        let mut r: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&heap, 0, &buffer_desc(host_size, true, true), D3D12_RESOURCE_STATE_COMMON, None, &mut r)?;
        Ok((heap, r.unwrap()))
    })();
    match opened {
        Ok((h, r)) => heaps.push(Heap { name: "host_opened", res: r, size: host_size, writable: true, _keep: Some(h) }),
        Err(e) => out.info("open_existing_heap", format!("failed: {e}")),
    }
    let readback = buffer(&dev, props(D3D12_HEAP_TYPE_READBACK), host_size, false, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();

    println!("## GPU access (GPU timestamps, per dispatch)");
    gpu_access(&mut gpu, &k, &heaps, &sink, &mut out);

    println!("## copies (GPU timestamps, per copy inside a batch)");
    // Plain empty submission roundtrip (CPU wall).
    let mut v = Vec::new();
    for _ in 0..30 + WARM {
        gpu.begin();
        v.push(gpu.run(0));
    }
    out.add("copy_batch_fixed_us", &v[WARM..], "us", "empty list: Execute+Signal+wait, CPU wall".to_string());
    let def = &heaps[0].res;
    let mut v = Vec::new();
    for _ in 0..30 + WARM {
        gpu.begin();
        // SAFETY: an open list.
        unsafe { gpu.list.CopyBufferRegion(def, 0, &heaps[1].res, 0, 64 * KIB) };
        v.push(gpu.run(0));
    }
    out.add("copy_single_64KiB_roundtrip_us", &v[WARM..], "us", "one 64 KiB upload copy, CPU wall".to_string());
    let mut pairs: Vec<(String, &ID3D12Resource, u64, &ID3D12Resource, u64)> = Vec::new();
    for h in &heaps[1..] {
        let n = if h.name == "upload" { "h2d".to_string() } else { format!("h2d_{}", h.name) };
        pairs.push((n, &h.res, h.size, def, 256 * MIB));
    }
    pairs.push(("d2h".into(), def, 256 * MIB, &readback, host_size));
    for h in &heaps[2..] {
        pairs.push((format!("d2h_{}", h.name), def, 256 * MIB, &h.res, h.size));
    }
    pairs.push(("d2d".into(), def, 128 * MIB, def, 128 * MIB));
    copies(&mut gpu, &pairs, &mut out);

    println!("## CPU access to mapped heaps");
    let len = 16 * MIB as usize;
    for h in &heaps[1..] {
        if h.name == "host_opened" {
            cpu_rw(host_ptr as *mut u8, len, h.name, true, true, &mut out);
            continue;
        }
        let mut p = std::ptr::null_mut();
        // SAFETY: CPU-visible heaps map.
        unsafe { h.res.Map(0, None, Some(&mut p)).unwrap() };
        let read = h.name != "upload" && h.name != "gpu_upload";
        cpu_rw(p as *mut u8, len, h.name, read, true, &mut out);
        if !read {
            // Write-combined: reading is very slow, measure 1 MiB only.
            cpu_rw(p as *mut u8, MIB as usize, &format!("{}_1MiB", h.name), true, false, &mut out);
        }
    }
    {
        let mut p = std::ptr::null_mut();
        // SAFETY: a readback heap maps.
        unsafe { readback.Map(0, None, Some(&mut p)).unwrap() };
        cpu_rw(p as *mut u8, len, "readback", true, false, &mut out);
    }
    cpu_mem(&mut out);

    println!("## page protection");
    pages(&mut out);

    std::fs::write(&path, out.json()).unwrap();
    println!("# wrote {path}");
}
