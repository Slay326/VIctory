//! Static samplers created in one order in every process, so a pipeline
//! kept in the library names the same driver samplers when it is loaded.
//!
//! NVIDIA's driver gives each distinct static sampler a place in a table
//! of the process's own, and a compiled pipeline refers to its samplers by
//! those places. Measured (RTX 3070, driver 616.56, 3 of 3 runs each, the
//! two-process test in `library`): a place belongs to a sampler's
//! description alone (its register and visibility do not count), places
//! are given in creation order among the samplers alive at the time, and
//! a released place goes to the next new sampler. The workers make host
//! pipelines in no fixed order, so without this a pipeline loaded in
//! another run sampled through other samplers (linear reads came back as
//! point reads), and the load still succeeded.
//!
//! So every distinct static sampler (as given to the host, after
//! `filter_for_host`; without register, space and visibility) is listed in
//! [`FILE`] beside the library, in the order first met. When the host
//! device is made ([`prime`]), each listed one is created in list order, in
//! a root signature of its own that lives as long as the process: its
//! place is then its line, whoever asks first, and no place is ever given
//! back. A sampler not listed is appended (and the list saved), and its own
//! root signature made, under one lock, before the root signature that
//! asked for it ([`admit`]). The presenter's sampler is admitted too.
//!
//! The guard: a pipeline's library name holds its samplers' lines and a
//! hash of the list up to the last of them ([`admit`]'s answer). The
//! places follow from the samplers created before, and those are the list
//! up to there, so a name found again means the same places; a lost, cut
//! or reordered list gives other names, a miss and a compile, never a load
//! through other samplers. Where the order cannot be vouched for, `admit`
//! gives no guard and `library::pipeline` makes such a pipeline on NVIDIA
//! (neither loaded nor stored): after a listed sampler was refused, or for
//! a sampler first met after code outside this crate got the device
//! ([`outside`]: the host's FidelityFX may create samplers of its own at
//! any time). Elsewhere such a pipeline uses the library as before.
//!
//! Design after trustno1-coder/VIctory-NVIDIA's `pipeline/samplers.rs`,
//! whose header counts 28 distinct static samplers in the title (the
//! driver's table holds 2,032).
//!
//! `VICTORY_SAMPLER_ORDER=0`: no list; on NVIDIA every pipeline with
//! static samplers is made every run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use victory_shader::emit::SAMPLER_SPACE;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

/// The list, beside the pipeline library: one sampler a line.
pub(super) const FILE: &str = "static-samplers.txt";

/// A sampler as the host is given it, without register, space and
/// visibility: filter, the three address modes, LOD bias (bits), maximum
/// anisotropy, comparison, border, least and greatest LOD (bits).
type Key = [u32; 10];

struct Order {
    /// The samplers in their places.
    list: Vec<Key>,
    /// Each one's place.
    places: HashMap<Key, u32>,
    /// After each place: a hash of the list up to it.
    prefix: Vec<String>,
    hasher: blake3::Hasher,
    /// The root signatures that hold the samplers: never released, so the
    /// driver keeps their places.
    kept: Vec<ID3D12RootSignature>,
    /// Where the list is saved, if anywhere.
    file: Option<PathBuf>,
    /// The first place made after code outside the crate had the device.
    outside_from: Option<u32>,
    /// A sampler was refused: what is created from now on may be out of
    /// the list's order.
    broken: bool,
}

// SAFETY: D3D12 objects are free-threaded; the root signatures are only held.
unsafe impl Send for Order {}

static ORDER: Mutex<Option<Order>> = Mutex::new(None);
static OUTSIDE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
pub(super) static UNPRIMED: AtomicBool = AtomicBool::new(false);

/// Whether the order is kept (`VICTORY_SAMPLER_ORDER=0` turns it off).
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SAMPLER_ORDER").is_none_or(|v| v != "0"))
}

/// Creates the listed samplers in list order: called when the host
/// device is made, before anything else has a static sampler.
pub(crate) fn prime(device: &ID3D12Device) {
    if on() {
        let mut order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
        order.get_or_insert_with(|| Order::read(device));
    }
}

/// Code outside the crate gets the host device (and may create static
/// samplers of its own): samplers first met from now on are made, not
/// kept, on NVIDIA for the rest of the run.
pub(crate) fn outside() {
    OUTSIDE.store(true, Ordering::Relaxed);
    let mut order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(o) = order.as_mut() {
        o.outside_from.get_or_insert(o.list.len() as u32);
    }
}

/// Places `samplers` (a root signature about to be created on `device`):
/// those not listed are appended, saved and created first. The text a
/// pipeline under that root signature adds to its library name, or `None`
/// where their places cannot be vouched for (see the module's notes). Empty
/// for no samplers.
pub(crate) fn admit(device: &ID3D12Device, samplers: &[D3D12_STATIC_SAMPLER_DESC]) -> Option<String> {
    if samplers.is_empty() {
        return Some(String::new());
    }
    if !on() {
        return None;
    }
    let mut order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    let order = order.get_or_insert_with(|| Order::read(device));
    let mut added = false;
    for s in samplers {
        let key = key(s);
        if !order.places.contains_key(&key) {
            if order.broken || !order.push(device, key) {
                return None;
            }
            added = true;
        }
    }
    if added {
        order.save();
    }
    order.guard(samplers.iter().map(|s| order.places[&key(s)]))
}

/// The list as saved (for the pre-build's list, which holds only for the
/// names it was saved with).
pub(super) fn listed() -> String {
    let order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    order.as_ref().map(|o| o.list.iter().map(line).collect::<Vec<_>>().join("\n")).unwrap_or_default()
}

impl Order {
    /// The list in the cache folder, its samplers created in order.
    fn read(device: &ID3D12Device) -> Order {
        let file = super::library::dir().map(|d| d.join(FILE));
        let mut order = Order {
            list: Vec::new(),
            places: HashMap::new(),
            prefix: Vec::new(),
            hasher: blake3::Hasher::new(),
            kept: Vec::new(),
            file: file.clone(),
            outside_from: OUTSIDE.load(Ordering::Relaxed).then_some(0),
            broken: false,
        };
        let text = file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()).unwrap_or_default();
        // A line that does not read ends the list: the guard then misses
        // for every place after it.
        for key in text.lines().map_while(parse) {
            if !order.places.contains_key(&key) && !order.push(device, key) {
                break;
            }
        }
        tracing::info!(
            samplers = order.list.len(),
            saved = order.file.is_some(),
            "static samplers created in the order of the cache's list"
        );
        order
    }

    /// Creates `key`'s sampler in the next place. False if refused (then
    /// nothing later is vouched for).
    fn push(&mut self, device: &ID3D12Device, key: Key) -> bool {
        // The control test's list read but not created.
        #[cfg(test)]
        let create = !UNPRIMED.load(Ordering::Relaxed);
        #[cfg(not(test))]
        let create = true;
        match create.then(|| holder(device, &desc(&key))) {
            Some(Ok(root)) => self.kept.push(root),
            None => {}
            Some(Err(e)) => {
                tracing::warn!(error = %e, sampler = line(&key), "a static sampler was refused; later ones are made every run");
                self.broken = true;
                return false;
            }
        }
        self.places.insert(key, self.list.len() as u32);
        self.list.push(key);
        self.hasher.update(line(&key).as_bytes());
        self.hasher.update(b"\n");
        self.prefix.push(self.hasher.finalize().to_hex()[..16].to_owned());
        true
    }

    /// Writes the list (all of it, so a reader never sees a part).
    fn save(&self) {
        let Some(file) = &self.file else { return };
        let mut text = self.list.iter().map(line).collect::<Vec<_>>().join("\n");
        text.push('\n');
        let written = file
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| super::library::write_atomically(file, text.as_bytes()));
        if let Err(e) = written {
            // The guard still holds: the next run's names for these miss.
            tracing::warn!(error = %e, "the list of static samplers was not saved");
        }
    }

    /// What a pipeline whose samplers have `places` adds to its name.
    fn guard(&self, places: impl Iterator<Item = u32>) -> Option<String> {
        let places: Vec<u32> = places.collect();
        let last = *places.iter().max()?;
        if self.outside_from.is_some_and(|from| last >= from) {
            return None;
        }
        let places = places.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        Some(format!("static samplers {places} of {}", self.prefix[last as usize]))
    }
}

fn key(s: &D3D12_STATIC_SAMPLER_DESC) -> Key {
    [
        s.Filter.0 as u32,
        s.AddressU.0 as u32,
        s.AddressV.0 as u32,
        s.AddressW.0 as u32,
        s.MipLODBias.to_bits(),
        s.MaxAnisotropy,
        s.ComparisonFunc.0 as u32,
        s.BorderColor.0 as u32,
        s.MinLOD.to_bits(),
        s.MaxLOD.to_bits(),
    ]
}

fn line(key: &Key) -> String {
    key.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")
}

fn parse(line: &str) -> Option<Key> {
    let mut key = [0; 10];
    let mut words = line.split_ascii_whitespace();
    for w in &mut key {
        *w = u32::from_str_radix(words.next()?, 16).ok()?;
    }
    words.next().is_none().then_some(key)
}

/// The sampler `key` stands for, at register 0 of the space the host's
/// pipelines use, for all stages (neither counts for its place).
fn desc(key: &Key) -> D3D12_STATIC_SAMPLER_DESC {
    let &[filter, u, v, w, bias, anisotropy, comparison, border, min_lod, max_lod] = key;
    D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER(filter as i32),
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(u as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(v as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(w as i32),
        MipLODBias: f32::from_bits(bias),
        MaxAnisotropy: anisotropy,
        ComparisonFunc: D3D12_COMPARISON_FUNC(comparison as i32),
        BorderColor: D3D12_STATIC_BORDER_COLOR(border as i32),
        MinLOD: f32::from_bits(min_lod),
        MaxLOD: f32::from_bits(max_lod),
        ShaderRegister: 0,
        RegisterSpace: SAMPLER_SPACE,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// A root signature that holds `sampler` and nothing else.
fn holder(device: &ID3D12Device, sampler: &D3D12_STATIC_SAMPLER_DESC) -> windows::core::Result<ID3D12RootSignature> {
    let root = D3D12_ROOT_SIGNATURE_DESC {
        NumStaticSamplers: 1,
        pStaticSamplers: sampler,
        ..Default::default()
    };
    // SAFETY: the description points at `sampler`, borrowed through the
    // call; the serialised bytes are read while their blob lives.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A saved list reads back as written; a damaged line ends it.
    #[test]
    fn the_list_reads_back_until_a_damaged_line() {
        let desc = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_ANISOTROPIC,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_BORDER,
            MipLODBias: -0.5,
            MaxAnisotropy: 16,
            ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            MinLOD: 1.0,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let written = key(&desc);
        let text = format!("{}\n{}\n", line(&written), line(&[7; 10]));
        let read: Vec<Key> = text.lines().map_while(parse).collect();
        assert_eq!(read, [written, [7; 10]]);
        for damaged in ["", "1 2 3", "1 2 3 4 5 6 7 8 9 a b", "1 2 3 4 5 6 7 8 9 g", "1 2 3 4 5 6 7 8 9 100000000"] {
            assert_eq!(parse(damaged), None, "{damaged:?}");
        }
        let cut = format!("{}\n1 2\n{}\n", line(&written), line(&[7; 10]));
        assert_eq!(cut.lines().map_while(parse).count(), 1);
    }

    /// Register, space and visibility are not part of a sampler's key.
    #[test]
    fn the_key_is_the_description_alone() {
        let a = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            MaxLOD: 12.0,
            ..Default::default()
        };
        let b = D3D12_STATIC_SAMPLER_DESC {
            ShaderRegister: 5,
            RegisterSpace: 2,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..a
        };
        assert_eq!(key(&a), key(&b));
        assert_ne!(key(&a), key(&D3D12_STATIC_SAMPLER_DESC { MaxLOD: 11.0, ..a }));
        // The sampler primed for a key has that key.
        assert_eq!(key(&desc(&key(&b))), key(&b));
    }

    /// The guard names a pipeline's places and the list up to the last of
    /// them: the same list gives the same text, another (cut, reordered,
    /// grown before them) another one; growth after them does not count.
    #[test]
    fn the_guard_follows_the_list_up_to_the_last_place() {
        let order = |keys: &[u32]| {
            let mut o = Order {
                list: Vec::new(),
                places: HashMap::new(),
                prefix: Vec::new(),
                hasher: blake3::Hasher::new(),
                kept: Vec::new(),
                file: None,
                outside_from: None,
                broken: false,
            };
            for &k in keys {
                let key = [k, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                // Bookkeeping only, as `push` does after creating the root signature.
                o.places.insert(key, o.list.len() as u32);
                o.list.push(key);
                o.hasher.update(line(&key).as_bytes());
                o.hasher.update(b"\n");
                o.prefix.push(o.hasher.finalize().to_hex()[..16].to_owned());
            }
            o
        };
        let guard = |o: &Order, places: &[u32]| o.guard(places.iter().copied());
        let first = guard(&order(&[1, 2, 3]), &[0, 1, 2]);
        assert!(first.is_some());
        assert_eq!(first, guard(&order(&[1, 2, 3, 4, 5]), &[0, 1, 2]), "later growth");
        assert_ne!(first, guard(&order(&[2, 1, 3]), &[0, 1, 2]), "reordered");
        assert_ne!(guard(&order(&[1, 2, 3]), &[2]), guard(&order(&[1, 4, 3]), &[2]), "same place, another list");
        assert_ne!(first, guard(&order(&[1, 2, 3]), &[1, 0, 2]), "other registers");
        let mut late = order(&[1, 2, 3]);
        late.outside_from = Some(2);
        assert!(guard(&late, &[0, 1]).is_some());
        assert!(guard(&late, &[0, 2]).is_none(), "made after outside code had the device");
    }
}
