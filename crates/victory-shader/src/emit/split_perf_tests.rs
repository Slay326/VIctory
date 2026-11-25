//! Timing (ignored by default; `cargo test -p victory-shader split_perf --
//! --ignored --nocapture`): a synthetic wave64 program shaped like the
//! XboxBvhBuilder's split programs (64 threads, a few exchanges up front,
//! a divergent per-lane loop writing the lane's own LDS dwords, a join,
//! each lane reading its own dwords back, 128 bytes stored per thread),
//! many groups, timed with GPU timestamps: split with and without the
//! thread-group barriers, and with the buffer in GPU memory or in system
//! memory (custom heap, write-back, L0: how the runtime keeps title buffers
//! that have no copy in GPU memory).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::host::Host;
use super::{HostShader, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC: u8 = 106;
const EXEC: u8 = 126;
const LITERAL: u16 = 255;

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
const fn sopp(op: u8, imm: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | imm as u16 as u32
}
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}
/// MUBUF `op vdata, vaddr, s[srsrc:+3], 0 offen offset:byte_off`.
const fn mubuf(op: u8, vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32) << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}
/// DS `op vdst, addr, data0 offset0`.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8, offset: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18 | offset as u32,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}
const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}
const fn int(n: u8) -> u16 {
    128 + n as u16
}

/// The program, for `groups` groups (the output's base follows the input).
/// `lanes`: the cross-lane reads up front (the exchanges).
fn program(groups: u32, lanes: bool) -> Vec<u32> {
    let out_base = (groups * 256 + 256).next_multiple_of(256);
    let mut w = Vec::new();
    // s7 = group id; s[4:5] the table; s[8:11] the UAV.
    w.push(sop1(0x03, 7, 5));
    w.push(sop1(0x03, 5, 0x80));
    w.extend(s_load_dwordx4(8, 4));
    // v1 = global id; v3 = out_base + 128 * id; v1 *= 4.
    w.extend(vop3(0x143, 1, 7, int(64), vgpr(0)));
    w.push(vop2(0x1A, 3, int(7), 1));
    w.push(vop2(0x25, 3, LITERAL, 3));
    w.push(out_base);
    w.push(vop2(0x1A, 1, int(2), 1));
    // v2 = (id & 7) + 1: the lane's trip count.
    w.push(vop2(0x1B, 2, int(7), 0));
    w.push(vop2(0x25, 2, int(1), 2));
    if lanes {
        // s21 = popcount(id < 40) + readlane(v0, 32) + readlane(v2, 33)
        //       + readfirstlane(v1).
        w.push(vopc(0xC4, int(40), 0));
        w.push(sop1(0x10, 21, VCC));
        w.extend(vop3(0x360, 22, vgpr(0), int(32), 0));
        w.extend(vop3(0x360, 23, vgpr(2), int(33), 0));
        w.push(vop1(0x02, 24, vgpr(1)));
        w.push(sop2(0x02, 21, 21, 22));
        w.push(sop2(0x02, 21, 21, 23));
        w.push(sop2(0x02, 21, 21, 24));
    } else {
        w.push(sop1(0x03, 21, 0x80));
    }
    // The divergent loop: lane i writes its own LDS dwords 8 * id + 0..v2.
    w.push(sop1(0x04, 12, EXEC)); // s_mov_b64 s[12:13], exec
    w.push(sop1(0x04, 16, EXEC)); // s_mov_b64 s[16:17], exec
    w.push(sop1(0x03, 20, 0x80)); // s_mov_b32 s20, 0
    w.push(sopp(0x02, 1)); // s_branch body
    let top = w.len();
    w.push(sop2(0x0F, EXEC, 18, 16)); // s_and_b64 exec, s[18:19], s[16:17]
    // body
    w.extend(vop3(0x143, 12, 20, int(4), vgpr(1))); // v12 = s20 * 4 + v1
    w.extend(mubuf(0x0C, 13, 12, 8, 0)); // buffer_load_dword v13
    w.push(vop2(0x25, 13, 21, 13)); // v13 += s21
    w.extend(vop3(0x143, 14, vgpr(0), int(8), 20)); // v14 = id * 8 + s20
    w.push(vop2(0x1A, 14, int(2), 14));
    w.extend(ds(0x0D, 0, 14, 13, 0)); // ds_write_b32 v14, v13
    w.push(sop2(0x02, 20, 20, 129)); // s_add_i32 s20, s20, 1
    w.push(sop1(0x04, 18, EXEC)); // s_mov_b64 s[18:19], exec
    w.push(vopc(0xD2, 20, 2)); // v_cmpx_eq_u32 exec, s20, v2
    w.push(sop2(0x15, 16, 16, EXEC)); // s_andn2_b64 s[16:17], s[16:17], exec
    let back = top as i32 - (w.len() as i32 + 1);
    w.push(sopp(0x05, back as i16)); // s_cbranch_scc1 top
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    // Each lane reads its own 8 dwords back and stores 32 dwords.
    w.push(vop2(0x1A, 14, int(5), 0));
    for k in 0..8u8 {
        w.extend(ds(0x36, 16 + k, 14, 0, 4 * k));
    }
    for k in 0..32u8 {
        w.extend(mubuf(0x1C, 16 + k % 8, 3, 8, 4 * u16::from(k)));
    }
    w.push(S_ENDPGM);
    w
}

fn lower_words(words: &[u32]) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        // 5 user SGPRs, the group id (TGID_X) after them, 2 KiB LDS.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | 1 << 7 | 4 << 15,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        host_wave_max: 32,
        view_marks: false,
    })
    .expect("lower")
}

fn shader(words: &[u32], barriers: bool) -> HostShader {
    lower::NO_GROUP_BARRIERS.with(|c| c.set(!barriers));
    let l = lower_words(words);
    lower::NO_GROUP_BARRIERS.with(|c| c.set(false));
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: false,
    }
}

/// Milliseconds per dispatch (median of `reps`), `groups` groups each, the
/// buffer in system memory when `system`; and the output's first dwords.
fn time(host: &Host, s: &HostShader, groups: u32, system: bool, reps: u32) -> (f64, Vec<u32>) {
    let (root, pso) = host.compute_pipeline(s).expect("pipeline");
    let dev = &host.device;
    let dwords = (groups * 64 + 64) + groups * 64 * 32 + 64;
    let bytes = u64::from(dwords) * 4;
    // SAFETY: D3D12 calls; resources outlive the GPU's use (fence wait).
    unsafe {
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let alloc: ID3D12CommandAllocator =
            dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .unwrap();
        let desc = |size: u64, flags| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let props = if system {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            }
        } else {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
        };
        let mut buf: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &props,
            D3D12_HEAP_FLAG_NONE,
            &desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buf,
        )
        .unwrap();
        let buf = buf.unwrap();
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &desc(4096 + 8 * u64::from(reps + 1), D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .unwrap();
        let rb = rb.unwrap();
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .unwrap();
        dev.CreateUnorderedAccessView(
            &buf,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: dwords,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            heap.GetCPUDescriptorHandleForHeapStart(),
        );
        let qh: ID3D12QueryHeap = {
            let mut q: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: reps + 1,
                    NodeMask: 0,
                },
                &mut q,
            )
            .unwrap();
            q.unwrap()
        };
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(&root);
        let ud = vec![0u32; s.user_data_dwords().max(1) as usize];
        list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::transmute_copy(&buf),
                }),
            },
        };
        list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for r in 0..reps {
            list.Dispatch(groups, 1, 1);
            list.ResourceBarrier(&[uav.clone()]);
            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, r + 1);
        }
        list.ResolveQueryData(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, reps + 1, &rb, 0);
        let out_at = u64::from((groups * 256 + 256).next_multiple_of(256));
        if !system {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&buf),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }]);
        }
        list.CopyBufferRegion(&rb, 2048, &buf, out_at, 2048);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        let start = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                panic!("device removed: {e}");
            }
            assert!(start.elapsed().as_secs() < 50, "GPU still running after 50 s");
            std::thread::yield_now();
        }
        let freq = queue.GetTimestampFrequency().unwrap() as f64;
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        let ts = std::slice::from_raw_parts(p.cast::<u64>(), reps as usize + 1);
        let mut ms: Vec<f64> = ts.windows(2).map(|t| (t[1] - t[0]) as f64 * 1e3 / freq).collect();
        let out = std::slice::from_raw_parts(p.cast::<u8>().add(2048).cast::<u32>(), 512).to_vec();
        rb.Unmap(0, None);
        ms.sort_by(f64::total_cmp);
        (ms[ms.len() / 2], out)
    }
}

#[test]
#[ignore = "timing on the hardware GPU"]
fn split_perf_barriers_and_memory() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no hardware device");
        return;
    };
    for groups in [1u32, 71, 9000] {
        let words = program(groups, true);
        let plain = program(groups, false);
        let variants = [
            ("split, exchanges + barriers", shader(&words, true)),
            ("split, no barriers", shader(&words, false)),
            ("split, no exchanges (LDS only)", shader(&plain, true)),
        ];
        for (name, s) in &variants {
            for system in [false, true] {
                let (ms, out) = time(&host, s, groups, system, 7);
                let nonzero = out.iter().filter(|&&x| x != 0).count();
                eprintln!(
                    "groups {groups:5}  {name:32} {:6}  {ms:9.3} ms/dispatch  {:8.4} us/group  out nonzero {nonzero}/512",
                    if system { "system" } else { "card" },
                    ms * 1e3 / f64::from(groups),
                );
            }
        }
    }
}

// ---- title memory: stores through a pointer, each marking its pages ----

/// The title address of the memory the program writes.
const TITLE: u64 = 0x12_3450_0000;

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// 64 threads (not split: no lane reads, no LDS), each storing 128 bytes
/// through the pointer in s[0:1] at `128 * global id` (eight
/// `global_store_dwordx4`), as 28918b71 stores its nodes.
fn memory_shader() -> HostShader {
    let mut w = Vec::new();
    w.extend(vop3(0x143, 1, 4, int(64), vgpr(0))); // v1 = s4 * 64 + v0
    w.push(vop2(0x1A, 1, int(7), 1)); // v1 <<= 7
    for k in 2..6u8 {
        w.push(vop1(0x01, k, vgpr(0)));
    }
    for k in 0..8u32 {
        w.extend(global(30, 0, 1, 2, 0, 16 * k));
    }
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 4 << 1 | 1 << 7, threads: [64, 1, 1] }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        host_wave_max: 32,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes && !l.view_addresses, "title memory only");
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    }
}

/// Where a buffer lives.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// GPU memory (default heap).
    Card,
    /// System memory the GPU writes (custom heap, write-back, L0), as the
    /// runtime's shared buffers (the written-pages bitmap).
    System,
    /// Upload heap (read only), as the runtime's page table.
    Upload,
}

/// Milliseconds per dispatch of [`memory_shader`] (median of `reps`), with
/// the copy of title memory, the written-pages bitmap and the page table
/// where given; and how many bitmap dwords are set.
fn time_memory(
    host: &Host,
    s: &HostShader,
    groups: u32,
    copy_at: Place,
    bits_at: Place,
    table_at: Place,
    reps: u32,
) -> (f64, usize) {
    let (root, pso) = host.compute_pipeline(s).expect("pipeline");
    let dev = &host.device;
    let copy_bytes = u64::from(groups) * 64 * 128;
    let pages = copy_bytes.div_ceil(1 << 16) as u32;
    let lo = TITLE as u32;
    assert!((lo >> 16 & 0xFFF) + pages <= 4096, "one block");
    // The page table: level 1, then the null block and block 1.
    let mut table = vec![0u32; 4096];
    table[((TITLE >> 32) as u32) as usize * 16 + (lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        table.extend([2u32, 0]);
    }
    for p in 0..pages {
        let e = 4096 + 2 * (4096 + ((lo >> 16 & 0xFFF) + p) as usize);
        table[e..e + 2].copy_from_slice(&[0, lo]);
    }
    let bitmap_dwords: u32 = 2 * 4096 / 32;
    // SAFETY: D3D12 calls; resources outlive the GPU's use (fence wait).
    unsafe {
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let alloc: ID3D12CommandAllocator =
            dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .unwrap();
        let make = |place: Place, size: u64| -> ID3D12Resource {
            let (props, flags, state) = match place {
                Place::Card => (
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                Place::System => (
                    D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_CUSTOM,
                        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                Place::Upload => (
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                ),
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size.next_multiple_of(256),
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut r,
            )
            .unwrap();
            r.unwrap()
        };
        let copy = make(copy_at, copy_bytes);
        let bitmap = make(bits_at, u64::from(bitmap_dwords) * 4);
        // The table: written through an upload buffer, copied when on the card.
        let table_bytes = table.len() as u64 * 4;
        let staging = make(Place::Upload, table_bytes);
        let mut p = std::ptr::null_mut();
        staging.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(table.as_ptr(), p.cast::<u32>(), table.len());
        staging.Unmap(0, None);
        let table_buf = match table_at {
            Place::Upload => staging.clone(),
            other => {
                let t = make(other, table_bytes);
                list.CopyBufferRegion(&t, 0, &staging, 0, table_bytes);
                t
            }
        };
        let readback = {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 8192,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut r,
            )
            .unwrap();
            r.unwrap()
        };
        // Views: the copy's raw SRV and UAV at 0 and 1, a null pair at 2, 3.
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .unwrap();
        let size = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let n = (copy_bytes / 4) as u32;
        for (i, r) in [(0usize, Some(&copy)), (2, None)] {
            let mut at = heap.GetCPUDescriptorHandleForHeapStart();
            at.ptr += i * size;
            let count = if r.is_some() { n } else { 1 };
            dev.CreateShaderResourceView(
                r,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            NumElements: count,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                }),
                at,
            );
            at.ptr += size;
            dev.CreateUnorderedAccessView(
                r,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: count,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                }),
                at,
            );
        }
        let qh: ID3D12QueryHeap = {
            let mut q: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: reps + 1, NodeMask: 0 },
                &mut q,
            )
            .unwrap();
            q.unwrap()
        };
        if !matches!(table_at, Place::Upload) {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&table_buf),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    }),
                },
            }]);
        }
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(&root);
        let mut ud = vec![0u32; s.user_data_dwords().max(1) as usize];
        for u in &s.user_data {
            if u.root == RootType::Constants && u.dword < 2 {
                ud[u.index as usize] = (TITLE >> (32 * u.dword)) as u32;
            }
        }
        list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        list.SetComputeRootShaderResourceView(1, table_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, bitmap.GetGPUVirtualAddress());
        list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for r in 0..reps {
            list.Dispatch(groups, 1, 1);
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, r + 1);
        }
        list.ResolveQueryData(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, reps + 1, &readback, 0);
        if !matches!(bits_at, Place::Upload) {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&bitmap),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }]);
        }
        list.CopyBufferRegion(&readback, 4096, &bitmap, 0, u64::from(bitmap_dwords) * 4);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        let start = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                panic!("device removed: {e}");
            }
            assert!(start.elapsed().as_secs() < 50, "GPU still running after 50 s");
            std::thread::yield_now();
        }
        let freq = queue.GetTimestampFrequency().unwrap() as f64;
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let ts = std::slice::from_raw_parts(p.cast::<u64>(), reps as usize + 1);
        let mut ms: Vec<f64> = ts.windows(2).map(|t| (t[1] - t[0]) as f64 * 1e3 / freq).collect();
        let bits = std::slice::from_raw_parts(p.cast::<u8>().add(4096).cast::<u32>(), bitmap_dwords as usize);
        let set = bits.iter().map(|b| b.count_ones() as usize).sum();
        readback.Unmap(0, None);
        ms.sort_by(f64::total_cmp);
        (ms[ms.len() / 2], set)
    }
}

#[test]
#[ignore = "timing on the hardware GPU"]
fn split_perf_title_memory_marks() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no hardware device");
        return;
    };
    let s = memory_shader();
    for groups in [71u32, 1000, 9000] {
        for (copy, bits, table) in [
            (Place::Card, Place::Card, Place::Card),
            (Place::Card, Place::Card, Place::Upload),
            (Place::Card, Place::System, Place::Upload),
            (Place::System, Place::Card, Place::Upload),
            (Place::System, Place::System, Place::Upload),
        ] {
            let (ms, set) = time_memory(&host, &s, groups, copy, bits, table, 5);
            eprintln!(
                "groups {groups:5}  copy {copy:?} bitmap {bits:?} table {table:?}: {ms:10.3} ms/dispatch  {:9.4} us/group  pages marked {set}",
                ms * 1e3 / f64::from(groups)
            );
        }
    }
}