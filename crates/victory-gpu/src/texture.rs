//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    /// The upload buffer and its mapping. Let go once the host texture
    /// holds its contents and the memory stayed as it was for a while
    /// ([`release_idle`]), made again by the next detiling: kept for good,
    /// every sampled texture took its size a second time in system memory.
    buffer: Mutex<Option<(ID3D12Resource, usize)>>,
    /// Its bytes.
    size: u64,
    /// When it was last made or detiled, in ms since the first one.
    touched: std::sync::atomic::AtomicU64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// copied it into the host texture, and whether one has copied it at
    /// all: a list records the copy only then. (Every list that sampled a
    /// title texture copied all of it again, changed or not: about 3,000
    /// subresource copies in a world frame of Grand Theft Auto VI, 23 ms
    /// of its GPU time.)
    held: Held,
}

/// Whether a host texture holds its staging buffer's contents.
struct Held {
    /// The staging buffer was detiled since a submitted list last copied it.
    dirty: std::sync::atomic::AtomicBool,
    /// A submitted list has copied it.
    uploaded: std::sync::atomic::AtomicBool,
}

impl Held {
    fn new() -> Held {
        Held { dirty: false.into(), uploaded: false.into() }
    }

    /// Whether the host texture has the staging buffer's contents: a list
    /// need not copy them.
    fn current(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        self.uploaded.load(Relaxed) && !self.dirty.load(Relaxed)
    }

    /// Whether the staging buffer was detiled and no list has copied it
    /// since.
    fn waiting(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The staging buffer was detiled.
    fn detiled(&self) {
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// A list being submitted copies the staging buffer into the host
    /// texture.
    fn copied(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.uploaded.store(true, Relaxed);
        self.dirty.store(false, Relaxed);
    }
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
}

/// Shared scratch buffer for reading tracked title-texture memory before
/// detiling, so each `Staging` does not retain a full-size `Vec<u8>`.
static READ_SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Records, once per list, the detiling that refreshes `resource`'s
/// staging buffer at submission if the memory changed, and the copy of the
/// staging buffer into the host copy if that does not hold its contents
/// yet (the texture is then left in the copy destination state). What the
/// check at submission finds changed, the next list recorded that samples
/// the texture copies: a frame late where every list of the frame was
/// recorded before that check.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    if staging.held.current() {
        // The host texture has the staging buffer's contents: only the
        // check at submission is left.
        #[cfg(feature = "diagnostics")]
        COPIES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    COPIES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    record(list, resource, texture, &staging);
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then copies once more).
    {
        let staging = staging.clone();
        list.defer_host(move || staging.held.copied());
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// Lists that recorded a texture's copy from its staging buffer, and lists
/// that did not need to, since the last frame-rate line (diagnostics).
#[cfg(feature = "diagnostics")]
static COPIES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Logs [`COPIES`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_copies(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [copied, skipped] = [0, 1].map(|i| COPIES[i].swap(0, Relaxed) as f64 / frames.max(1.0));
    if copied + skipped > 0.0 {
        tracing::info!(
            "sampled title textures per frame: {copied:.1} copied from their staging buffers by a list, {skipped:.1} uses by lists that found them current"
        );
    }
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    let Some(staging) = staging(resource, texture) else {
        return false;
    };
    // Also what an earlier check detiled and no list has copied since.
    let detiled = staging.refresh();
    if !detiled && !staging.held.waiting() {
        return false;
    }
    record(list, resource, texture, &staging);
    // The list runs in this submission, ahead of its other lists.
    staging.held.copied();
    true
}

/// Records the copy of every subresource of `staging` into `texture`,
/// moving it to the copy destination state first.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging) {
    const COPY_DEST: u32 = 0x400;
    // The list holds the buffer until it is reset: the staging state may
    // let go of it once a submitted list has copied it.
    let buffer = staging.buffer.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|b| b.0.clone());
    let Some(buffer) = buffer else {
        // Let go of: the host texture holds its contents.
        return;
    };
    list.keep(&buffer);
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    /// Returns the bytes of title memory read, and whether it detiled.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            if t.take(start, start + size).is_empty() {
                return (0, false);
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            if scratch.len() < size {
                scratch.resize(size, 0);
            }
            let tiled = &mut scratch[..size];
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            self.detile(tiled);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.is_none() {
            // Let go of while idle: made again (zeroed).
            let Some((b, data)) = host().and_then(|h| h.upload_buffer(self.size)) else {
                // Stays changed for the next check only where tracked
                // pages are taken again; the host texture keeps what it has.
                return;
            };
            STAGING_BYTES.fetch_add(self.size, std::sync::atomic::Ordering::Relaxed);
            REMADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *buffer = Some((b, data as usize));
        }
        let Some(data) = buffer.as_ref().map(|b| b.1 as *mut u8) else {
            return;
        };
        self.touched.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
        self.held.detiled();
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    STAGING_BYTES.fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let s = Arc::new(Staging {
        buffer: Mutex::new(Some((buffer, data as usize))),
        size: total,
        touched: now_ms().into(),
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        last: Mutex::default(),
        held: Held::new(),
    });
    *held = Some(s.clone());
    let mut all = STAGINGS.lock().unwrap_or_else(|e| e.into_inner());
    if all.len().is_power_of_two() {
        all.retain(|w| w.strong_count() > 0);
    }
    all.push(Arc::downgrade(&s));
    Some(s)
}

/// Every live staging state, for [`release_idle`].
static STAGINGS: Mutex<Vec<std::sync::Weak<Staging>>> = Mutex::new(Vec::new());
/// Bytes of staging buffers alive; staging buffers let go of while idle,
/// and made again.
static STAGING_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RELEASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REMADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether staging buffers are kept only while needed
/// (`VICTORY_TEXTURE_LEAN=0`: one per sampled texture, for good).
fn lean() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_LEAN").is_none_or(|v| v != "0"))
}

/// Milliseconds since the first call.
fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.buffer.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
            STAGING_BYTES.fetch_sub(self.size, Relaxed);
        }
    }
}

/// Lets go of the staging buffers whose contents their host textures hold
/// and whose memory was not detiled for `VICTORY_STAGING_IDLE_MS` (3000):
/// lists that copy from one hold it themselves ([`record`]). Called now and
/// then (per frame-rate line).
pub(crate) fn release_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    static IDLE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let idle = *IDLE.get_or_init(|| {
        std::env::var("VICTORY_STAGING_IDLE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000)
    });
    if !lean() {
        return;
    }
    let all: Vec<Arc<Staging>> = {
        let mut all = STAGINGS.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|w| w.strong_count() > 0);
        all.iter().filter_map(std::sync::Weak::upgrade).collect()
    };
    let now = now_ms();
    for s in all {
        if now.saturating_sub(s.touched.load(Relaxed)) < idle {
            continue;
        }
        // Under the lock detiling takes: not while one is filled.
        let _last = s.last.lock().unwrap_or_else(|e| e.into_inner());
        if !s.held.current() {
            continue;
        }
        if s.buffer.lock().unwrap_or_else(|e| e.into_inner()).take().is_some() {
            STAGING_BYTES.fetch_sub(s.size, Relaxed);
            RELEASED.fetch_add(1, Relaxed);
        }
    }
}

/// Upload memory textures hold: staging buffers' bytes, read rooms' bytes,
/// staging buffers let go of and made again since the last call.
pub(crate) fn upload_memory() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let read = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner()).capacity() as u64;
    (STAGING_BYTES.load(Relaxed), read, RELEASED.swap(0, Relaxed), REMADE.swap(0, Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A list copies a texture's staging buffer only while the host
    /// texture does not hold its contents: before the first copy, and after
    /// a check at submission detiled the memory again.
    #[test]
    fn a_texture_is_copied_again_only_after_it_was_detiled() {
        let held = Held::new();
        assert!(!held.current(), "never copied");
        // The first list's submission: its copy is noted, then the check
        // detiles (everything is changed at first).
        held.copied();
        held.detiled();
        assert!(!held.current(), "the next list copies once more");
        held.copied();
        assert!(held.current(), "later lists do not");
        // The title writes the memory: found at a submission.
        held.detiled();
        assert!(!held.current() && held.waiting());
        held.copied();
        assert!(held.current() && !held.waiting());
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
