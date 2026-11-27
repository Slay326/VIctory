//! senior 82 (diagnostics, template for Fable's Fix C; not a production
//! path): the thread-limit prologue `v_cmpx_gt_u32 exec, x, global id` at a
//! wave64 program's start, as the BVH harness puts it in front of
//! caa78b8c/28918b71. A synthetic program stores `s_bcnt1(exec)` at its
//! global thread id; lanes >= x must not write, the last wave counts the
//! rest. Run on the host GPU and WARP with the host's 32-lane waves.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const GROUP: u32 = 64;

/// x in user SGPR s1 (root constant), group id in s2, the raw UAV at
/// table entry 2 (s0 the table). `prologue`: the lanes >= x leave EXEC.
fn program(prologue: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xD746_0004,
        0x0401_0C02, // v_lshl_add_u32 v4, s2, 6, v0 (global id)
    ];
    if prologue {
        w.push(0x7DA8_0801); // v_cmpx_gt_u32 exec, s1, v4
    }
    w.extend([
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E06_0206, // v_mov_b32 v3, s6
        0x3420_0882, // v_lshlrev_b32 v16, 2, v4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(words: &[u32]) -> HostShader {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 1, slot: 1 },
        ],
        used: vec![range(ResourceType::Srv, 0, 8, 0), range(ResourceType::Srv, 8, 4, 16), range(ResourceType::Uav, 16, 4, 0)],
        // 2 user SGPRs, group id in x.
        io: StageIo::Cs(ComputeIo { rsrc2: (2 << 1) | (1 << 7), threads: [GROUP, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
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
        view_marks: false,
        // As the BVH builder runs on a 32-lane adapter.
        host_wave_max: 32,
        group_x_offset: false,
        loop_limit: super::loop_limit(),
    })
    .expect("lower");
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l.samplers.iter().map(|&w| super::StaticSampler::from_words(w)).collect(),
        samplers: l.samplers,
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

/// What each global thread id holds after ceil(x / 64) groups: per adapter.
fn run(prologue: bool, x: u32) -> Vec<(&'static str, Vec<u32>)> {
    let shader = build(&program(prologue));
    let groups = x.div_ceil(GROUP);
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer { data: &[0; 4], stride: super::host::Stride::Structured(16) },
        View::RawUav(groups * GROUP),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: pipeline: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(4) as usize];
        ud[1] = x;
        let r = host
            .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [groups, 1, 1] })
            .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(!out.is_empty(), "no adapter ran the test");
    out
}

/// Lanes < x hold the active-lane count of their wave (64, the last wave
/// the rest), lanes >= x keep 0 (not written).
fn want(x: u32) -> Vec<u32> {
    let n = x.div_ceil(GROUP) * GROUP;
    (0..n).map(|g| if g < x { (x - g / GROUP * GROUP).min(GROUP) } else { 0 }).collect()
}

fn check(x: u32) {
    for (label, got) in run(true, x) {
        assert_eq!(got, want(x), "{label}: x = {x}");
    }
}

#[test]
fn thread_limit_x_1() {
    check(1);
}

#[test]
fn thread_limit_x_group() {
    check(GROUP);
}

#[test]
fn thread_limit_x_group_plus_1() {
    check(GROUP + 1);
}

/// Red proof: without the prologue the same expectation fails (lanes >= x
/// write, the last wave counts 64), for every x that is not a multiple of
/// the group.
#[test]
fn without_the_prologue_lanes_past_x_write() {
    for x in [1, GROUP + 1] {
        for (label, got) in run(false, x) {
            assert_ne!(got, want(x), "{label}: x = {x} passes without the prologue: the test cannot fail");
        }
    }
}
