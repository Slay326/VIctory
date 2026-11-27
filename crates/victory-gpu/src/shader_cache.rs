//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

/// The folder `use_shader_cache` names, and the cache in it, opened on first
/// use ([`cache`]).
static BASE: OnceLock<PathBuf> = OnceLock::new();
static CACHE: OnceLock<Option<(Cache, PathBuf)>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder of
/// this translator and its settings. It is opened on the first translation,
/// not here: the key names the device's widest wave, which a run learns
/// only when it creates its device, after this.
pub fn use_shader_cache(dir: PathBuf) {
    let _ = BASE.set(dir);
}

/// The cache's folder, opening it now (`None`: no cache, or it cannot be
/// opened). After the device, if one is to set the wave width.
pub fn shader_cache_dir() -> Option<PathBuf> {
    cache().map(|(_, dir)| dir.clone())
}

fn cache() -> Option<&'static (Cache, PathBuf)> {
    CACHE
        .get_or_init(|| {
            let dir = BASE
                .get()?
                .join(&blake3::Hash::from(*build()).to_hex()[..16]);
            match Cache::open(&dir) {
                Ok(cache) => {
                    tracing::info!(dir = %dir.display(), "shader cache");
                    Some((cache, dir))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no shader cache");
                    None
                }
            }
        })
        .as_ref()
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings (the loop limit, which a diagnostics build can
/// change at run time, and the host's widest wave, which decides whether
/// console wave64 programs run split). Taken on first use, after the device
/// has set the wave width.
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(
            victory_shader::SOURCE_HASH,
            ENCODING,
            limit,
            victory_shader::emit::host_wave_max(),
        )
    })
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(translator: &str, encoding: &[u8], limit: u32, wave: u32) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    h.update(&wave.to_le_bytes());
    *h.finalize().as_bytes()
}

/// [`build`] as this host's entries are keyed: a translation also depends
/// on the host's view heaps where they are smaller than translations
/// assume (`victory_shader::emit::set_view_heap`: `views` slots, `title` of
/// them the title's). Elsewhere the build itself.
fn host_build(build: &[u8; 32], views: u32, title: u32) -> [u8; 32] {
    if (views, title)
        == (
            victory_shader::emit::HEAP_VIEWS,
            victory_shader::emit::HEAP_VIEWS,
        )
    {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&views.to_le_bytes());
    h.update(&title.to_le_bytes());
    *h.finalize().as_bytes()
}

/// Whether the compute program in `container` is one the title dispatches
/// with more than the host's 65,535 groups in x: its translation takes a
/// group offset and the host splits its dispatches
/// (`victory_shader::emit::Options::group_x_offset`). Listed by content
/// hash (16 hex digits, as the logs name shaders) in `VICTORY_WIDE_GROUPS`,
/// by default Grand Theft Auto VI's screen-probe final gather (72,432
/// groups in x at 2560x1440).
// shortcut: a list, not found at run time; a program missing from it keeps
// its dispatches cut at 65,535 groups (the clamp, `crate::indirect`).
pub fn wide_groups(container: &[u8]) -> bool {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_WIDE_GROUPS").unwrap_or_else(|_| "2dc364df5d3278b8".into());
        list.split(',').map(|h| h.trim().to_ascii_lowercase()).filter(|h| !h.is_empty()).collect()
    });
    if list.is_empty() {
        return false;
    }
    let hash = blake3::hash(container).to_hex();
    list.iter().any(|h| hash.starts_with(h.as_str()))
}

/// Whether the compute program in `container` is translated to take a
/// thread limit (`victory_shader::emit::Options::thread_limit`), so that
/// the host can run a `DispatchX` that counts x in threads as the console
/// does (`crate::draw::State::dispatch_x`). `VICTORY_DISPATCHX_THREADS`:
/// unset or `0` none (as before the switch existed); `1` Grand Theft Auto
/// VI's two BVH builder passes that are dispatched so (their x is nodes:
/// 1..11,632 with groups of 64); else a list of content hashes as
/// `VICTORY_WIDE_GROUPS` takes.
// shortcut: a list, not found at run time (a program's dispatch flags are
// known only after it is translated); a program missing from it is
// dispatched as before and named once in the log.
pub fn dispatch_threads(container: &[u8]) -> bool {
    let list = DISPATCH_THREADS.get_or_init(dispatch_threads_env);
    if list.is_empty() {
        return false;
    }
    let hash = blake3::hash(container).to_hex();
    list.iter().any(|h| hash.starts_with(h.as_str()))
}

/// Whether `VICTORY_DISPATCHX_THREADS` names any program.
pub fn dispatch_threads_on() -> bool {
    !DISPATCH_THREADS.get_or_init(dispatch_threads_env).is_empty()
}

static DISPATCH_THREADS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

fn dispatch_threads_env() -> Vec<String> {
    dispatch_threads_list(std::env::var("VICTORY_DISPATCHX_THREADS").ok().as_deref())
}

/// The hashes `VICTORY_DISPATCHX_THREADS` names ([`dispatch_threads`]).
fn dispatch_threads_list(value: Option<&str>) -> Vec<String> {
    let value = match value.map(str::trim) {
        None | Some("" | "0") => return Vec::new(),
        Some("1") => "caa78b8ca62bc3d6,28918b7150a7aad5",
        Some(list) => list,
    };
    value.split(',').map(|h| h.trim().to_ascii_lowercase()).filter(|h| !h.is_empty()).collect()
}

/// The translation of a console shader container, from the cache when
/// there is one, and whether it came from there.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    host_shader_with(container, None, (0, 0))
}

/// Translations made with coherent title memory since the start
/// ([`host_shader_with`]'s `coherent`; translated or from the cache).
static COHERENT: AtomicU64 = AtomicU64::new(0);

/// How many shaders were taken with coherent title memory since the start
/// (`VICTORY_BVH_COHERENT`, the BVH builder's stages).
pub fn coherent_count() -> u64 {
    COHERENT.load(Ordering::Relaxed)
}

/// The translation of `container` as `options` say, with the coherence
/// parts `coherent` (`victory_shader::emit::with_coherent`, on the thread
/// that translates). 0: exactly `victory_shader::emit::translate_with`.
pub(crate) fn translate_container(
    container: &[u8],
    options: victory_shader::emit::Options,
    coherent: u8,
) -> Result<HostShader, String> {
    let translate = || victory_shader::emit::translate_with(container, options).map_err(|e| e.to_string());
    if coherent != 0 {
        victory_shader::emit::with_coherent(coherent, translate)
    } else {
        translate()
    }
}

/// The key part of an entry: `build` ([`host_build`]), then a builder
/// stage's own loop limit, its coherence (parts, `VICTORY_BVH_COHERENT`
/// stage) and the thread limit ([`dispatch_threads`]), each only when set,
/// so every other entry keeps its key.
fn entry_build(build: [u8; 32], loop_limit: Option<u32>, coherent: (u8, u8), thread_limit: bool) -> [u8; 32] {
    let mut build = build;
    if thread_limit {
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"thread limit: a dispatch's x in threads");
        build = *h.finalize().as_bytes();
    }
    if let Some(limit) = loop_limit {
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"loop limit of the program's own");
        h.update(&limit.to_le_bytes());
        build = *h.finalize().as_bytes();
    }
    if coherent.0 != 0 {
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"coherent title memory where the console bypasses its caches: parts, stage");
        h.update(&[coherent.0, coherent.1]);
        build = *h.finalize().as_bytes();
    }
    build
}

/// [`host_shader`] with loops left after `loop_limit` iterations if set
/// (`victory_shader::emit::Options::loop_limit`; the BVH builder's stages,
/// `crate::pipeline`), and with the coherence parts `coherent.0` if not 0
/// (the builder's stages with `VICTORY_BVH_COHERENT`, stage `coherent.1`,
/// [`translate_container`]):
/// such entries are keyed apart, every other entry stays as it is.
pub fn host_shader_with(
    container: &[u8],
    loop_limit: Option<u32>,
    coherent: (u8, u8),
) -> (Result<HostShader, String>, bool) {
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    let options = victory_shader::emit::Options {
        view_marks: true,
        group_x_offset: wide_groups(container),
        loop_limit,
        thread_limit: dispatch_threads(container),
    };
    let translate = |c: &[u8]| match translate_container(c, options, coherent.0) {
        // A program the limit cannot be put before stays as it is without
        // the switch: its translation has no limit entry, so its dispatches
        // run as before (`crate::draw::State::dispatch_x` names it).
        Err(error) if options.thread_limit => {
            tracing::warn!(
                container = %&blake3::hash(c).to_hex()[..16],
                %error,
                "VICTORY_DISPATCHX_THREADS: no thread limit for this program; translated without"
            );
            let options = victory_shader::emit::Options { thread_limit: false, ..options };
            translate_container(c, options, coherent.0)
        }
        made => made,
    };
    // The host first: making it says what its view heaps are like, which
    // translations depend on. Without a host they are as for most hosts.
    let _ = crate::host::host();
    let (views, title) = (
        victory_shader::emit::heap_views(),
        victory_shader::emit::title_views(),
    );
    let build = entry_build(host_build(build(), views, title), loop_limit, coherent, options.thread_limit);
    if coherent.0 != 0 {
        let n = COHERENT.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::info!(
            n,
            parts = coherent.0,
            stage = coherent.1,
            container = %&blake3::hash(container).to_hex()[..16],
            "VICTORY_BVH_COHERENT: a BVH builder stage taken with coherent title memory"
        );
    }
    match cache() {
        Some((cache, _)) => cached(
            cache,
            &build,
            container,
            translate,
        ),
        None => (translate(container), false),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit;
/// `made` keeps a miss's translation, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms here).
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let made = (self.translate)(&input[32..]);
        let bytes = encode(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return (shader, false);
        }
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    (
        made().unwrap_or_else(|| (converter.translate)(container)),
        false,
    )
}

const MAGIC: &[u8; 4] = b"VSH1";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let memory = r.word()?;
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        view_marks: memory & 32 != 0,
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A compute program in an `ACCC` container (the `FSPX` layout
    /// `victory_shader::native::fspx_program` reads), wave32, 32 lanes, a
    /// root UAV whose address is in `s[4:5]`: a pointer loaded from it, a
    /// `glc dlc` load of title memory where it points (as the BVH builder's
    /// refit loads), a store of the value 4 bytes on,
    /// `s_waitcnt_vscnt null, 0`. Made here, so no title data is needed.
    pub(crate) fn coherent_container() -> Vec<u8> {
        const GLC: u32 = 1 << 16;
        const DLC: u32 = 1 << 12;
        let global = |op: u32, vdst: u32, vaddr: u32, data: u32, saddr: u32, offset: u32| {
            [0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF), vaddr | data << 8 | saddr << 16 | vdst << 24]
        };
        // A pointer read from the root view, then title memory where it
        // points (an address the program computes, as the builder's are).
        let mut w = vec![
            0x7E00_0000 | 1 << 17 | 0x01 << 9 | 4, // v_mov_b32 v1, s4
            0x7E00_0000 | 2 << 17 | 0x01 << 9 | 5, // v_mov_b32 v2, s5
        ];
        w.extend(global(0x0D, 3, 1, 0, 0x7D, 0)); // global_load_dwordx2 v[3:4], v[1:2], off: a pointer
        w.push(0xBF8C_0000); // s_waitcnt 0
        let [a, b] = global(0x0C, 5, 3, 0, 0x7D, 0); // global_load_dword v5, v[3:4], off glc dlc
        w.extend([a | GLC | DLC, b, 0xBF8C_0000]); // s_waitcnt 0
        w.extend(global(0x1C, 0, 3, 5, 0x7D, 4)); // global_store_dword v[3:4], v5, off offset:4
        w.extend([0xBBFD_0000, 0xBF81_0000]); // s_waitcnt_vscnt null, 0; s_endpgm
        let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
        let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // Registers: threads 32x1x1, wave32, 16 VGPRs, 6 user SGPRs.
        let packet = words(&[0, 0, 0, 32 | 1 << 16, 1 | (1 << 4) << 16, 1, 6 << 1]);
        // A root UAV (resource type 3, root type 1) in user data 4, slot 0.
        let mut uav = [0u8; 32];
        uav[0] = 0x41;
        uav[4..6].copy_from_slice(&[3, 1]);
        uav[20..23].copy_from_slice(&[2, 4, 0]);
        uav[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut hash = [0u8; 32];
        hash[0] = 0x43;
        let mut meta = words(&[0x403, 0, code.len() as u32 / 4, 0, 0]);
        meta.extend_from_slice(&[0x40, 4, 1, 0]);
        meta.extend_from_slice(&uav);
        meta.extend_from_slice(&hash);
        let len = meta.len() as u32 - 0x14;
        meta[4..8].copy_from_slice(&len.to_le_bytes());
        let blobs: [(u32, &[u8]); 4] = [(0, &packet), (1, &[0; 0x14]), (6, &code), (7, &meta)];
        let mut part = vec![0u8; 0x10 + 16 * blobs.len()];
        for (i, (kind, blob)) in blobs.iter().enumerate() {
            let at = part.len().next_multiple_of(16);
            part.resize(at, 0);
            part[0x10 + 16 * i..0x10 + 16 * (i + 1)].copy_from_slice(&words(&[*kind, at as u32, blob.len() as u32, 0]));
            part.extend_from_slice(blob);
        }
        let header = words(&[u32::from_le_bytes(*b"FSPX"), 1, 4, part.len() as u32]);
        part[..16].copy_from_slice(&header);
        // A container of that one part (as `victory_shader::dxbc` reads one).
        let total = 0x20 + 4 + 8 + part.len();
        let mut out = b"DXBC".to_vec();
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&words(&[1, total as u32, 1, 0x24]));
        out.extend_from_slice(&victory_shader::FourCC::ACCC.0);
        out.extend_from_slice(&(part.len() as u32).to_le_bytes());
        out.extend_from_slice(&part);
        out
    }

    /// The options the host translates with ([`host_shader_with`]).
    fn host_options() -> victory_shader::emit::Options {
        victory_shader::emit::Options { view_marks: true, group_x_offset: false, loop_limit: None, thread_limit: false }
    }

    /// `VICTORY_BVH_COHERENT` off (and for every stage not the builder's
    /// with it on): the translation is the one made without the switch,
    /// byte for byte, and its cache entry keeps its key; on, the container
    /// above (glc dlc, vscnt) translates differently and is keyed apart.
    #[test]
    fn coherent_translations_change_only_what_asks_for_them() {
        let container = coherent_container();
        let today = victory_shader::emit::translate_with(&container, host_options()).expect("translates");
        let plain = translate_container(&container, host_options(), 0).expect("translates");
        assert_eq!(plain.dxil, today.dxil, "off: byte for byte as without the switch");
        let b = victory_shader::emit::COHERENT_UAV | victory_shader::emit::COHERENT_GLOBAL;
        let coherent = translate_container(&container, host_options(), b).expect("translates");
        assert_ne!(coherent.dxil, today.dxil, "on: the glc dlc load through the coherent UAV changes the DXIL");
        assert!(!victory_shader::emit::coherent(), "the thread's switch is reset");
        let again = translate_container(&container, host_options(), 0).expect("translates");
        assert_eq!(again.dxil, today.dxil, "a coherent translation leaves later ones as they were");
        // Keys: as before the switch existed when off (the old derivation,
        // written out), apart when on.
        let build = [7u8; 32];
        let old = |limit: Option<u32>| match limit {
            None => build,
            Some(limit) => {
                let mut h = blake3::Hasher::new();
                h.update(&build);
                h.update(b"loop limit of the program's own");
                h.update(&limit.to_le_bytes());
                *h.finalize().as_bytes()
            }
        };
        for limit in [None, Some(4096)] {
            assert_eq!(entry_build(build, limit, (0, 0), false), old(limit), "off: the key of today, {limit:?}");
            // No parts is off whatever the stage says.
            assert_eq!(entry_build(build, limit, (0, 2), false), old(limit), "no parts: the key of today, {limit:?}");
            assert_ne!(entry_build(build, limit, (b, 1), false), old(limit), "on: keyed apart, {limit:?}");
            assert_ne!(
                entry_build(build, limit, (b, 1), false),
                entry_build(build, limit, (b, 2), false),
                "each stage its own key"
            );
        }
        assert_ne!(entry_build(build, None, (b, 1), false), entry_build(build, Some(4096), (b, 1), false));
    }

    /// `VICTORY_DISPATCHX_THREADS`: unset or 0 names no program, 1 the two
    /// BVH builder passes, anything else is the list; a translation with
    /// the thread limit is keyed apart from the one without (with every
    /// other key part), and without it the key is the one of before.
    #[test]
    fn thread_limit_translations_are_listed_and_keyed_apart() {
        assert!(dispatch_threads_list(None).is_empty());
        assert!(dispatch_threads_list(Some("")).is_empty());
        assert!(dispatch_threads_list(Some(" 0 ")).is_empty());
        assert_eq!(dispatch_threads_list(Some("1")), ["caa78b8ca62bc3d6", "28918b7150a7aad5"]);
        assert_eq!(dispatch_threads_list(Some("ABCD1234, 5062bc7a48e842f8,")), ["abcd1234", "5062bc7a48e842f8"]);
        let build = [7u8; 32];
        let b = victory_shader::emit::COHERENT_UAV;
        for limit in [None, Some(4096)] {
            for coherent in [(0, 0), (b, 1)] {
                let without = entry_build(build, limit, coherent, false);
                let with = entry_build(build, limit, coherent, true);
                assert_ne!(with, without, "keyed apart: {limit:?} {coherent:?}");
            }
        }
        assert_eq!(entry_build(build, None, (0, 0), false), build, "off: the key of before");
        assert_ne!(entry_build(build, None, (0, 0), true), entry_build(build, Some(4096), (0, 0), true));
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &[7; 32], b"a container", translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the host's wave width) makes
    /// another key, so no entry
    /// written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let k = key("0123456789abcdef", b"encoding", 1 << 20, 64);
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20, 64));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20, 64));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20, 64));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19, 64));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 20, 32));
        // A host with smaller view heaps has entries of its own.
        let views = victory_shader::emit::HEAP_VIEWS;
        assert_eq!(host_build(&k, views, views), k);
        assert_ne!(host_build(&k, 1_000_000, 721_469), k);
        assert_ne!(
            host_build(&k, 1_000_000, 721_469),
            host_build(&k, 1_000_000, 1_000_000)
        );
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                victory_shader::emit::host_wave_max()
            )
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
