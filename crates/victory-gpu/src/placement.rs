//! Host textures placed in heaps made ahead instead of committed one by
//! one: a new shot brings hundreds of textures, and the host spent most of
//! a frame making them (300 textures, 430 MB on the RX 6700 XT: committed
//! 255 ms; one heap 32 ms, then placed 3 ms).
//!
//! Textures without render-target, depth or unordered-access use, up to a
//! quarter of a heap, go into heaps of [`HEAP`] bytes, first fit. A
//! texture's range is free again once nothing else holds its resource
//! (heap slots keep views' resources until the running submission has run,
//! lists until they are reset), so no view or list sees its memory reused.
//! Empty heaps past [`SPARE`] are released.
//!
//! `VICTORY_TEXTURE_HEAPS_LEAN=1` ([`lean`]): no empty heap is kept, new
//! textures take the smallest free range that fits (best fit) instead of
//! the first, and past the card's budget the eviction pass empties the
//! least used heap whose textures all fit elsewhere by making them again
//! there ([`choose_drain`], `crate::resource::compact`). The free room in
//! the heaps (~480 MB of 27 heaps on the RTX 3070, mostly left by the
//! loading screens) then goes back to the card.

use std::collections::BTreeMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Bytes of one heap.
const HEAP: u64 = 64 << 20;
/// Empty heaps kept for the next textures (none when [`lean`]).
const SPARE: usize = 2;
/// Passes a heap may take to empty out ([`choose_drain`]) before another
/// is chosen (a list or a slot still holding a texture moved off it).
const DRAIN_PASSES: u64 = 600;

/// `VICTORY_TEXTURE_HEAPS_LEAN=1`: no spare heaps, best fit, heaps emptied
/// past the budget (module notes). Off by default.
pub(crate) fn lean() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_TEXTURE_HEAPS_LEAN");
        if on {
            tracing::warn!("heaps of textures kept lean: no spare heap, best fit, the least used emptied past the budget (VICTORY_TEXTURE_HEAPS_LEAN)");
        }
        on
    })
}
/// Placement alignment of textures that are not render targets.
const ALIGN: u64 = 64 << 10;

/// Where a texture lies: heap, offset and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    heap: u64,
    offset: u64,
    size: u64,
}

/// A heap and its free ranges (offset → bytes).
struct Heap {
    heap: ID3D12Heap,
    id: u64,
    free: BTreeMap<u64, u64>,
}

#[derive(Default)]
struct Heaps {
    heaps: Vec<Heap>,
    next: u64,
    /// Ranges whose texture may still be held elsewhere, with a holder of
    /// our own.
    pending: Vec<(ID3D12Resource, Placement)>,
    /// Lean ([`lean`]): the heap being emptied (no new texture goes there)
    /// and the pass it was chosen at.
    draining: Option<(u64, u64)>,
    /// Heaps emptied out and released (lean).
    drained: u64,
}

// SAFETY: D3D12 heaps and resources are free-threaded; used under the lock.
unsafe impl Send for Heaps {}

static HEAPS: Mutex<Option<Heaps>> = Mutex::new(None);

/// The first range of `free` that fits `size` bytes at `align`: its
/// offset, taken out of `free`.
fn take(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let (&start, &len, at) = free.iter().find_map(|(s, l)| {
        let at = s.next_multiple_of(align);
        (at + size <= s + l).then_some((s, l, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    Some(at)
}

/// The smallest range of `free` that fits `size` bytes at `align`: its
/// length and start (best fit), not taken out.
fn best_fit(free: &BTreeMap<u64, u64>, size: u64, align: u64) -> Option<(u64, u64)> {
    free.iter()
        .filter(|&(&s, &l)| s.next_multiple_of(align) + size <= s + l)
        .map(|(&s, &l)| (l, s))
        .min()
}

/// Takes `size` bytes at `align` out of the free range starting at
/// `start` (which fits them): their offset.
fn take_in(free: &mut BTreeMap<u64, u64>, start: u64, size: u64, align: u64) -> u64 {
    let len = free.remove(&start).expect("a free range");
    let at = start.next_multiple_of(align);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    at
}

/// Whether textures of `sizes` bytes (at [`ALIGN`]) all fit into the free
/// ranges `frees` (one map a heap), placed largest first by best fit.
fn all_fit(mut frees: Vec<BTreeMap<u64, u64>>, sizes: &[u64]) -> bool {
    let mut sizes = sizes.to_vec();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.iter().all(|&size| {
        let best = frees.iter().enumerate().filter_map(|(i, f)| best_fit(f, size, ALIGN).map(|(l, s)| (l, i, s))).min();
        best.map(|(_, i, s)| take_in(&mut frees[i], s, size, ALIGN)).is_some()
    })
}

/// Gives `[offset, offset + size)` back to `free`, joined with its
/// neighbours.
fn give(free: &mut BTreeMap<u64, u64>, offset: u64, size: u64) {
    let (mut start, mut end) = (offset, offset + size);
    if let Some((&s, &l)) = free.range(..start).next_back()
        && s + l == start
    {
        free.remove(&s);
        start = s;
    }
    if let Some(&l) = free.get(&end) {
        free.remove(&end);
        end += l;
    }
    free.insert(start, end - start);
}

/// Whether a resource has no holder but the one passed.
fn last_holder(resource: &ID3D12Resource) -> bool {
    // SAFETY: a live COM object; one reference taken and given back.
    unsafe {
        let raw = resource.as_raw();
        let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
        ((**unknown).AddRef)(raw);
        ((**unknown).Release)(raw) == 1
    }
}

impl Heaps {
    /// Frees the pending ranges nothing else holds now, and releases empty
    /// heaps past [`SPARE`].
    fn reap(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if last_holder(&self.pending[i].0) {
                let (_, p) = self.pending.swap_remove(i);
                if let Some(h) = self.heaps.iter_mut().find(|h| h.id == p.heap) {
                    give(&mut h.free, p.offset, p.size);
                }
            } else {
                i += 1;
            }
        }
        let empty = |h: &Heap| h.free.get(&0) == Some(&HEAP);
        let kept = if lean() { 0 } else { SPARE };
        let mut spare = 0;
        let draining = self.draining.map(|d| d.0);
        let mut drained = 0;
        self.heaps.retain(|h| {
            if !empty(h) {
                return true;
            }
            spare += 1;
            let keep = spare <= kept;
            if !keep && draining == Some(h.id) {
                drained += 1;
            }
            keep
        });
        if drained > 0 {
            self.draining = None;
            self.drained += drained;
        }
    }

    /// A range of `size` bytes: in a heap with room, else in a new heap.
    fn allocate(&mut self, device: &ID3D12Device, size: u64, align: u64) -> Option<(ID3D12Heap, Placement)> {
        if lean() {
            let draining = self.draining.map(|d| d.0);
            let best = self
                .heaps
                .iter()
                .enumerate()
                .filter(|(_, h)| Some(h.id) != draining)
                .filter_map(|(i, h)| best_fit(&h.free, size, align).map(|(l, s)| (l, i, s)))
                .min();
            if let Some((_, i, start)) = best {
                let h = &mut self.heaps[i];
                let offset = take_in(&mut h.free, start, size, align);
                return Some((h.heap.clone(), Placement { heap: h.id, offset, size }));
            }
        } else {
            for h in &mut self.heaps {
                if let Some(offset) = take(&mut h.free, size, align) {
                    return Some((h.heap.clone(), Placement { heap: h.id, offset, size }));
                }
            }
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: HEAP,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: ALIGN,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation, our out pointer.
        unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
        let heap = heap?;
        crate::host::name(&heap, || "texture heap".into());
        let id = self.next;
        self.next += 1;
        let mut free = BTreeMap::new();
        free.insert(0, HEAP);
        let offset = take(&mut free, size, align)?;
        self.heaps.push(Heap { heap: heap.clone(), id, free });
        Some((heap, Placement { heap: id, offset, size }))
    }
}

/// Whether a texture of `desc` goes into a heap: no render-target, depth or
/// unordered-access use, and at most a quarter of a heap.
pub(crate) fn fits(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    const USES: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Flags.0 & USES != 0 || desc.SampleDesc.Count > 1 {
        return false;
    }
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    info.SizeInBytes != u64::MAX && info.SizeInBytes <= HEAP / 4 && info.Alignment <= ALIGN
}

/// Room for a texture of `desc` in a heap ([`fits`] first): the heap and
/// the placement, which the texture gives back with [`free`].
pub(crate) fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Heap, Placement)> {
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    let size = info.SizeInBytes.next_multiple_of(ALIGN);
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(Heaps::default);
    heaps.reap();
    heaps.allocate(device, size, info.Alignment.max(ALIGN))
}

/// A range [`place`] gave whose texture could not be made there.
pub(crate) fn unplace(p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut().and_then(|hs| hs.heaps.iter_mut().find(|h| h.id == p.heap)) {
        give(&mut h.free, p.offset, p.size);
    }
}

/// The texture at `p` is let go of: its range is free again once nothing
/// else holds `resource` (checked at the next placements and passes).
pub(crate) fn free(resource: ID3D12Resource, p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut() {
        h.pending.push((resource, p));
        h.reap();
    }
}

/// Bytes of heap space free for textures (after freeing what nothing
/// holds now): the card's memory they take is not in use.
pub(crate) fn free_bytes() -> u64 {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return 0 };
    h.reap();
    h.heaps.iter().flat_map(|h| h.free.values()).sum()
}

/// Heaps of textures held, and their used and free bytes (after freeing
/// what nothing holds now), for the statistics line.
pub(crate) fn stats() -> (usize, u64, u64) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return (0, 0, 0) };
    h.reap();
    let free: u64 = h.heaps.iter().flat_map(|h| h.free.values()).sum();
    (h.heaps.len(), h.heaps.len() as u64 * HEAP - free, free)
}

impl Placement {
    /// Its offset in its heap.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// Its heap (an id of ours).
    pub(crate) fn heap(&self) -> u64 {
        self.heap
    }

    /// Its bytes.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}

impl Heaps {
    /// [`choose_drain`] on these heaps.
    fn choose_drain(&mut self, movable: &std::collections::HashMap<u64, Vec<u64>>, pass: u64) -> Option<u64> {
        self.reap();
        if let Some((id, at)) = self.draining {
            if pass.saturating_sub(at) < DRAIN_PASSES && self.heaps.iter().any(|h| h.id == id) {
                return None;
            }
            self.draining = None;
        }
        let used = |h: &Heap| HEAP - h.free.values().sum::<u64>();
        let mut order: Vec<&Heap> = self.heaps.iter().filter(|h| used(h) > 0 && used(h) <= HEAP / 2).collect();
        order.sort_by_key(|h| used(h));
        let id = order.into_iter().find_map(|h| {
            let sizes = movable.get(&h.id)?;
            // Every range in use there is a texture that may move (none
            // pending, none of another kind), and they all fit elsewhere.
            if sizes.iter().sum::<u64>() != used(h) {
                return None;
            }
            let others: Vec<BTreeMap<u64, u64>> = self.heaps.iter().filter(|o| o.id != h.id).map(|o| o.free.clone()).collect();
            all_fit(others, sizes).then_some(h.id)
        })?;
        self.draining = Some((id, pass));
        Some(id)
    }
}

/// Lean ([`lean`]), at pass `pass`: with `movable` the bytes of the
/// textures in each heap that may be made again elsewhere (by heap id),
/// the heap to empty now: the least used one, at most half full, whose
/// ranges in use are all movable textures that fit into the other heaps'
/// free ranges. No new texture goes there from now on; none while the last
/// heap chosen is still emptying (up to [`DRAIN_PASSES`] passes).
pub(crate) fn choose_drain(movable: &std::collections::HashMap<u64, Vec<u64>>, pass: u64) -> Option<u64> {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.get_or_insert_with(Heaps::default).choose_drain(movable, pass)
}

/// The heap being emptied, and the heaps emptied and released so far.
#[cfg(feature = "diagnostics")]
pub(crate) fn draining() -> (Option<u64>, u64) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.as_ref().map_or((None, 0), |h| (h.draining.map(|d| d.0), h.drained))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture's range is not handed out again while anything else holds
    /// its resource (a view's slot, a list), and is once nothing does.
    #[test]
    fn a_range_is_reused_only_once_nothing_holds_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 9,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(fits(&host.device, &desc));
        let mut heaps = Heaps::default();
        let (heap, a) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a range of the heap, our out pointer.
        unsafe { host.device.CreatePlacedResource(&heap, a.offset(), &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture) }.unwrap();
        let texture = texture.unwrap();
        let held = texture.clone();
        heaps.pending.push((texture, a));
        heaps.reap();
        let (_, b) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_ne!(b, a, "still held: not handed out");
        drop(held);
        heaps.reap();
        let (_, c) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_eq!(c, a, "nothing holds it: free again");
    }

    /// Best fit takes the smallest range that holds the texture; a heap is
    /// chosen to empty only if all its ranges in use are movable textures
    /// that fit into the others, the least used first.
    #[test]
    fn best_fit_and_the_heap_to_empty() {
        let k = ALIGN;
        let free = BTreeMap::from([(0, 8 * k), (10 * k, 3 * k), (20 * k, 2 * k)]);
        assert_eq!(best_fit(&free, 2 * k, k), Some((2 * k, 20 * k)), "the smallest that fits");
        assert_eq!(best_fit(&free, 3 * k, k), Some((3 * k, 10 * k)));
        assert_eq!(best_fit(&free, 9 * k, k), None);
        let mut f = free.clone();
        assert_eq!(take_in(&mut f, 10 * k, 2 * k, k), 10 * k);
        assert_eq!(f, BTreeMap::from([(0, 8 * k), (12 * k, k), (20 * k, 2 * k)]));
        // Two holes of 3 and 2: textures of 3 and 2 fit, of 3 and 3 not.
        let holes = vec![BTreeMap::from([(0, 3 * k)]), BTreeMap::from([(5 * k, 2 * k)])];
        assert!(all_fit(holes.clone(), &[2 * k, 3 * k]));
        assert!(!all_fit(holes, &[3 * k, 3 * k]));

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let mut heaps = Heaps::default();
        // Heap a: 3 MiB in use, b: 2 MiB, c: 40 MiB (more than half).
        let mib = 1 << 20;
        let (_, a) = heaps.allocate(&host.device, 3 * mib, k).unwrap();
        heaps.heaps.last_mut().unwrap().free = BTreeMap::from([(3 * mib, HEAP - 3 * mib)]);
        let mut next = |used: u64| {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: HEAP,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: ALIGN,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: plain creation, our out pointer.
            unsafe { host.device.CreateHeap(&desc, &mut heap) }.unwrap();
            let id = heaps.next;
            heaps.next += 1;
            heaps.heaps.push(Heap { heap: heap.unwrap(), id, free: BTreeMap::from([(used, HEAP - used)]) });
            id
        };
        let b = next(2 * mib);
        let c = next(40 * mib);
        let mut movable = std::collections::HashMap::new();
        movable.insert(a.heap, vec![mib, 2 * mib]);
        movable.insert(c, vec![40 * mib]);
        assert_eq!(heaps.choose_drain(&movable, 1), Some(a.heap), "b is less used but not movable");
        assert_eq!(heaps.choose_drain(&movable, 2), None, "a still emptying");
        movable.insert(b, vec![mib]);
        assert_eq!(heaps.choose_drain(&movable, 2 + DRAIN_PASSES), Some(a.heap), "b: 1 of 2 MiB movable, a chosen again after its passes");
        heaps.draining = None;
        movable.insert(b, vec![mib, mib]);
        assert_eq!(heaps.choose_drain(&movable, 3 + DRAIN_PASSES), Some(b), "the least used movable");
        // No new texture goes into the heap being emptied.
        if lean() {
            let (_, p) = heaps.allocate(&host.device, k, k).unwrap();
            assert_ne!(p.heap, b);
        }
    }

    #[test]
    fn ranges_are_taken_first_fit_and_joined_when_given_back() {
        let k = 64 << 10;
        let mut free = BTreeMap::from([(0, 10 * k)]);
        let a = take(&mut free, 3 * k, k).unwrap();
        let b = take(&mut free, 2 * k, k).unwrap();
        let c = take(&mut free, 4 * k, k).unwrap();
        assert_eq!((a, b, c), (0, 3 * k, 5 * k));
        assert_eq!(take(&mut free, 2 * k, k), None, "one 64 KiB range left");
        give(&mut free, b, 2 * k);
        assert_eq!(take(&mut free, 3 * k, k), None, "two ranges of 2 and 1, not joined");
        give(&mut free, c, 4 * k);
        assert_eq!(free, BTreeMap::from([(3 * k, 7 * k)]), "joined with both neighbours");
        give(&mut free, a, 3 * k);
        assert_eq!(free, BTreeMap::from([(0, 10 * k)]));
        // An aligned range inside a free one keeps the padding before it.
        let mut free = BTreeMap::from([(k, 10 * k)]);
        assert_eq!(take(&mut free, k, 4 * k), Some(4 * k));
        assert_eq!(free, BTreeMap::from([(k, 3 * k), (5 * k, 6 * k)]));
    }
}
