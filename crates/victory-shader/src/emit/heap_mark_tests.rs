//! The heap-usage marks (`Lower::mark_heap_use`): a program reading views
//! from the resource heap itself sets each slot's bit in the bitmap (root
//! UAV `u3` of the memory space), once a block, with a load first and the
//! atomic only where the bit is still clear. Synthetic GFX10 programs
//! lowered to DXIL (validated by `dxil.dll` when present) and run on the
//! host GPU and WARP; the timing (ignored by default; `cargo test -p
//! victory-shader --release heap_mark -- --ignored --nocapture`) compares
//! with the unguarded atomics.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HEAP_VIEWS, HostShader, VIEW_STRIDES, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 the heap index of the first of `k` structured buffers (a
/// root constant), s1 the resource heap, s2 a table with a raw UAV. Each
/// lane sums dword 1 of element `lane` of each buffer (one descriptor load
/// and one buffer load each) and stores the sum to dword `lane`; with
/// `branch`, a uniform branch on s0 between the loads and the store adds 1
/// where s0 is not 0.
fn program(k: u32, branch: bool) -> Vec<u32> {
    let mut w = vec![
        0x8F04_8500, // s_lshl_b32 s4, s0, 5
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
    ];
    for n in 0..k {
        // s_load_dwordx4 s[8+4n:], s[6:7], s4 offset:32n
        w.extend([0xF408_0003 | (8 + 4 * n) << 6, 0x0800_0000 | 32 * n]);
    }
    w.extend([
        0xBEAC_0302, // s_mov_b32 s44, s2
        0xBEAD_0380, // s_mov_b32 s45, 0
        0xF408_0C16,
        0xFA00_0000, // s_load_dwordx4 s[48:51], s[44:45], 0x0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]);
    for n in 0..k {
        // buffer_load_dword v(2+n), v0, s[8+4n:], 0 idxen offset:4
        w.extend([0xE030_2004, 0x8000_0000 | (2 + n) << 16 | (2 + n) << 8]);
    }
    w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    for n in 1..k {
        // v_add_nc_u32 v2, v(2+n), v2
        w.push(0x25 << 25 | 2 << 17 | 2 << 9 | (256 + 2 + n));
    }
    if branch {
        w.extend([
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        ]);
    }
    w.extend([
        0xE070_1000,
        0x800C_0201, // buffer_store_dword v2, v1, s[48:51], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

/// [`program`] lowered with marks, guarded unless `unguarded`.
fn build(words: &[u32], threads: u32, unguarded: bool) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::TEST_HEAP_MARKS.with(|c| c.set((true, !unguarded)));
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        host_wave_max: 64,
    });
    lower::TEST_HEAP_MARKS.with(|c| c.set((false, true)));
    let lowered = lowered.expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: Vec::new(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        heap_usage: lowered.heap_usage,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Element `e` of buffer `n`: (100 + e, 200 + e + 1000 n, 300 + e).
fn buffer(n: u32, elements: u32) -> Vec<u32> {
    (0..elements)
        .flat_map(|e| [100 + e, 200 + e + 1000 * n, 300 + e])
        .collect()
}

/// Runs `shader` (`k` buffers at heap slots 1..=k, the output at 0) over
/// `groups` groups; the output and the bitmap.
fn run(host: &Host, shader: &HostShader, k: u32, groups: u32, data: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
    let mut views = vec![View::RawUav(256)];
    views.extend(data.iter().map(|d| View::Buffer {
        data: d,
        stride: Stride::Structured(12),
    }));
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Constants => 1,
            RootType::Table => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    // The view-stride table: the output raw, the buffers 12-byte elements.
    let strides: Vec<u32> = std::iter::once(4).chain((0..k).map(|_| 12)).collect();
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![
                    (0, Root::Constants(ud)),
                    (1, Root::View { srv: true, data: strides }),
                    (2, Root::Uav(HEAP_VIEWS / 32)),
                ],
                views: &views,
                samplers: &[],
                groups: [groups, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    let bitmap = out.last().cloned().unwrap_or_default();
    (out[0].clone(), bitmap)
}

/// Reads through heap slots 1..=4 mark exactly those slots, with and
/// without a branch after the reads (the marks are set at the end of the
/// block), and read the right values.
#[test]
fn heap_reads_mark_their_slots_once_guarded() {
    let k = 4;
    let data: Vec<Vec<u32>> = (0..k).map(|n| buffer(n, 32)).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for branch in [false, true] {
            let shader = build(&program(k, branch), 32, false);
            assert!(shader.heap_usage, "the program marks heap slots");
            assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
            let (out, bitmap) = run(&host, &shader, k, 1, &data);
            let want: Vec<u32> = (0..32)
                .map(|l| (0..k).map(|n| 200 + l + 1000 * n).sum::<u32>() + u32::from(branch))
                .collect();
            assert_eq!(out[..32], want[..], "{label} branch {branch}");
            assert_eq!(bitmap.len() as u32, HEAP_VIEWS / 32);
            assert_eq!(bitmap[0], 0b11110, "{label} branch {branch}: slots 1-4 marked");
            assert!(bitmap[1..].iter().all(|&w| w == 0), "{label} branch {branch}: nothing else");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Milliseconds of a dispatch of `groups` groups beyond one of a single
/// group (wall time, median of `reps`).
fn time(host: &Host, shader: &HostShader, k: u32, groups: u32, data: &[Vec<u32>], reps: usize) -> (f64, Vec<u32>) {
    let mut ms = Vec::new();
    let mut bitmap = Vec::new();
    for _ in 0..reps {
        let t = std::time::Instant::now();
        run(host, shader, k, 1, data);
        let one = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        bitmap = run(host, shader, k, groups, data).1;
        ms.push((t.elapsed().as_secs_f64() - one) * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    (ms[ms.len() / 2], bitmap)
}

#[test]
#[ignore = "timing on the hardware GPU"]
fn heap_mark_timing() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no hardware device");
        return;
    };
    let k = 8;
    let threads = 256;
    let data: Vec<Vec<u32>> = (0..k).map(|n| buffer(n, threads)).collect();
    for groups in [32000u32, 128000, 512000] {
        for (name, unguarded) in [("unguarded atomics", true), ("load, then atomic if clear", false)] {
            let shader = build(&program(k, false), threads, unguarded);
            let (ms, bitmap) = time(&host, &shader, k, groups, &data, 9);
            eprintln!(
                "groups {groups:6} x {threads} threads, {k} heap reads each: {name:28} {ms:9.3} ms  bitmap word 0 {:#x}",
                bitmap[0]
            );
        }
    }
}
