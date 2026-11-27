//! [`super::Options::thread_limit`]: a dispatch that counts x in threads.
//! A synthetic wave64 program stores `s_bcnt1(exec)` at its global thread
//! id; with the limit, threads at or past it must not write and the last
//! wave counts only the rest, as the console's partly filled last wave
//! does. Run on the host GPU (and WARP where it takes the pipeline) with
//! 32-lane host waves, as the BVH builder's stages run on such adapters.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, UserData, lower, thread_limit};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const GROUP: u32 = 64;

/// The group id in s2, the raw UAV at table entry 2 (s0 the table); s1 is
/// a root constant the program does not read.
fn native() -> NativeProgram {
    let words: [u32; 13] = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xD746_0004,
        0x0401_0C02, // v_lshl_add_u32 v4, s2, 6, v0 (global id)
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E06_0206, // v_mov_b32 v3, s6
        0x3420_0882, // v_lshlrev_b32 v16, 2, v4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
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
    NativeProgram {
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
    }
}

/// The program translated with the thread limit (`limit`) or as it is, by
/// the steps of `translate_with`.
fn build(limit: bool) -> HostShader {
    let mut native = native();
    let regs = limit.then(|| thread_limit::prepend(&mut native).expect("prepend"));
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let mut io = io::raise(&p, &native).expect("io::raise");
    let lowered = |io: &io::Io| {
        lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io,
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
        .expect("lower")
    };
    let l = match regs {
        None => lowered(&io),
        Some(regs) => {
            let first = thread_limit::past_the_layout(&native);
            thread_limit::inputs(&p, regs, first, &mut io).expect("inputs");
            let index = thread_limit::free_index(&lowered(&io).user_data, false);
            thread_limit::inputs(&p, regs, index, &mut io).expect("inputs");
            lowered(&io)
        }
    };
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

/// What each global thread id holds after `groups` groups ran with the
/// limit entry (if the translation has one) set to `value`: per adapter.
fn run(limit: bool, value: u32, groups: u32) -> Vec<(&'static str, Vec<u32>)> {
    let shader = build(limit);
    let entry = shader.user_data.iter().find(|u| super::is_thread_limit(u));
    assert_eq!(entry.is_some(), limit, "the limit's entry exists exactly with the option");
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
        // WARP refuses fixed WaveSize pipelines (as in synthetic_tests); the
        // hardware adapter must run it.
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) if warp => {
                eprintln!("warp: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: pipeline: {e}"),
        };
        let mut ud = vec![0u32; shader.user_data_dwords().max(4) as usize];
        if let Some(u) = entry {
            ud[u.index as usize] = value;
        }
        let r = host
            .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [groups, 1, 1] })
            .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(!out.is_empty(), "no adapter ran the test");
    out
}

/// Threads below x hold the active-lane count of their wave (64, the last
/// wave the rest), threads from x on keep 0 (not written).
fn want(x: u32, groups: u32) -> Vec<u32> {
    (0..groups * GROUP).map(|g| if g < x { (x - g / GROUP * GROUP).min(GROUP) } else { 0 }).collect()
}

/// x threads in the groups that hold them, as the host dispatches them.
fn check(x: u32) {
    let groups = x.div_ceil(GROUP);
    for (label, got) in run(true, x, groups) {
        assert_eq!(got, want(x, groups), "{label}: x = {x}");
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

/// A dispatch counted in groups passes `u32::MAX`: every thread runs.
#[test]
fn no_limit_runs_every_thread() {
    for (label, got) in run(true, u32::MAX, 2) {
        assert_eq!(got, vec![GROUP; 2 * GROUP as usize], "{label}");
    }
}

/// Red proof: translated without the limit, the same dispatches fail the
/// same expectation (threads past x write, the last wave counts 64) for
/// every x that is not a multiple of the group.
#[test]
fn without_the_limit_threads_past_x_write() {
    for x in [1, GROUP + 1] {
        let groups = x.div_ceil(GROUP);
        for (label, got) in run(false, x, groups) {
            assert_ne!(got, want(x, groups), "{label}: x = {x} passes without the limit: the test cannot fail");
        }
    }
}

/// The entry is told from a title's by its dword, which no root parameter
/// has, and sits at the lowest user-data dword the program's entries leave
/// free: here dword 1, the root constant the program does not read.
#[test]
fn the_entry_is_the_programs_first_free_dword() {
    let shader = build(true);
    let limits: Vec<&UserData> = shader.user_data.iter().filter(|u| super::is_thread_limit(u)).collect();
    assert_eq!(limits.len(), 1);
    assert_eq!((limits[0].index, limits[0].dword, limits[0].root), (1, super::THREAD_LIMIT, RootType::Constants));
    assert_eq!(shader.user_data_dwords(), build(false).user_data_dwords(), "the user data did not grow");
    // A title's root constant at the same user-data dword is no limit.
    let title = UserData { index: 1, slot: 1, root: RootType::Constants, dword: 0 };
    assert!(!super::is_thread_limit(&title));
    assert!(!super::is_thread_limit(&UserData { root: RootType::Descriptor, ..*limits[0] }));
    // The group offset's dword stays free for it; a taken dword is passed.
    let taken = |index| UserData { index, slot: 0, root: RootType::Constants, dword: index };
    assert_eq!(thread_limit::free_index(&[taken(0), taken(1), taken(3)], false), 2);
    let run: Vec<UserData> = (0..super::GROUP_X_OFFSET_INDEX).map(taken).collect();
    assert_eq!(thread_limit::free_index(&run, true), super::GROUP_X_OFFSET_INDEX + 1);
    assert_eq!(thread_limit::free_index(&run, false), super::GROUP_X_OFFSET_INDEX);
}

/// Only a compute program whose start nothing jumps to takes the limit;
/// the code of one that does not is left as it was.
#[test]
fn programs_the_limit_cannot_be_put_before_are_refused() {
    let mut vertex = NativeProgram { stage: HwStage::Vs, ..native() };
    let before = vertex.clone();
    assert!(thread_limit::prepend(&mut vertex).is_err());
    assert_eq!(vertex, before);
    // s_branch back to the first instruction: the compare would run again.
    let mut looping = native();
    looping.code = [0xBF80_0000u32, 0xBF82_FFFE].iter().flat_map(|w| w.to_le_bytes()).collect();
    let before = looping.clone();
    assert!(thread_limit::prepend(&mut looping).is_err());
    assert_eq!(looping, before);
    // The compute program takes it: one word in front, one VGPR more.
    let mut taken = native();
    let (s, v) = thread_limit::prepend(&mut taken).expect("prepend");
    assert_eq!((s, v), (105, 32));
    assert_eq!(taken.code.len(), native().code.len() + 4);
    assert_eq!(taken.code[4..], native().code[..]);
    assert_eq!(taken.vgprs, 33);
}
