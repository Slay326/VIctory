//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Each view heap has one page table, which every list using title memory
//! with that heap bound reads, and each host copy has a pair of views in
//! slots of the heap kept for it while it lives ([`HostHeap::pair`], past
//! the per-draw dynamic slots). At each submission with such a list the
//! table is rewritten if copies came or went since, and every copy it names
//! is brought up to date (page by page, as every refresh; once per
//! submission until a step writes title memory). Shaders mark the pages
//! they write in their list's bitmap; after the submission each marked page
//! goes back into title memory ([`crate::shadow::Shadow::copy_back_page`])
//! and the bitmap is cleared.
//!
//! Cost per submission with such a list: one change check of every host
//! copy (the refresh), copies of the pages the title wrote, a compare of
//! each 64 KiB page shaders wrote, and the table's rewrite when copies came
//! or went (two views per new copy); per list a bitmap of 256 KiB.
//!
//! Memory no copy holds reads 0 and takes no writes (null views,
//! bounds-checked); shaders mark such pages (or 256 MiB regions with no
//! block) in the bitmap and [`log_frame`] counts them.
//! With `VICTORY_BVH_ARENA=1` pages no copy holds get pages of their own in
//! the GPU's memory once shaders reach them (`crate::arena`); tables name
//! them where no copy holds the page, and what shaders write there stays on
//! the card.
//! shortcut: a page two copies share without either holding all of it is
//! the larger one's.
//! shortcut: a table names copies in at most [`CAPACITY`] 256 MiB regions;
//! past that, the smallest copies read as missing (logged once).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::descriptor::host::HostHeap;
use crate::host::HostList;
use crate::shadow::Shadow;

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
const LEVEL1: usize = 4096;
/// Level-2 blocks a table has room for besides the null one: copies in 512
/// distinct 256 MiB regions (the world's ~4,100 copies use a few dozen).
const CAPACITY: usize = 512;
/// Sentinel in [`Layout::owner`] for a page with no host copy.
const NO_OWNER: u32 = u32::MAX;
/// Total dwords in a page table buffer (level-1 + `CAPACITY + 1` blocks of
/// `BLOCK` 2-dword entries).
const TABLE_DWORDS: usize = LEVEL1 + (CAPACITY + 1) * BLOCK * 2;
/// Bits in a list's dirty-page bitmap (one per level-2 entry across all blocks).
const BITMAP_BITS: usize = (CAPACITY + 1) * BLOCK;
/// 32-bit words in a dirty-page bitmap.
const BITMAP_WORDS: usize = BITMAP_BITS / 32;
/// Byte size of a dirty-page bitmap buffer (256-byte aligned).
const BITMAP_BYTES: u64 = (BITMAP_BITS / 8).next_multiple_of(256) as u64;
/// Maximum number of dirty-page bitmaps kept in [`BITMAP_POOL`].
const BITMAP_POOL_CAP: usize = 64;
/// The key of our own view heap ([`heap`]) among the title's.
pub(crate) const OURS: usize = usize::MAX;

/// A list's use of title memory with a view heap bound.
pub(crate) struct Map {
    /// The heap's page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The list's page bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
    /// Ranges that go back byte for byte instead (root UAVs, see
    /// [`Map::precise`]).
    precise: Arc<Mutex<Vec<(u64, u64)>>>,
    /// The BVH builder ran in the list with the arena on
    /// ([`Map::note_builder`]).
    builder: Arc<std::sync::atomic::AtomicBool>,
}

impl Map {
    /// The BVH builder runs in the list with the arena on: what shaders
    /// write through the page table into title buffers larger than
    /// [`crate::arena::BACK`] stays out of title memory.
    pub(crate) fn note_builder(&self) {
        self.builder.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Shaders write `[address, address + len)` through a root UAV, which
    /// goes back by its bytes before and after the run
    /// ([`Shadow::gpu_writes`]): the marked pages it touches are left to
    /// that. A whole page from the copy would put back what it held at
    /// the submission over what the title (or DirectStorage, through
    /// physical pages no watch sees) wrote there since.
    /// shortcut: a marked page partly in such a range loses what shaders
    /// wrote in its other part.
    pub(crate) fn precise(&self, address: u64, len: u64) {
        self.precise
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, len));
    }
}

/// What a heap's page table names: the copies, the level-1 index of each
/// block (block 0 is the null one), and the copy each entry names (or [`NO_OWNER`]).
#[derive(Default)]
struct Layout {
    copies: Vec<Arc<Shadow>>,
    /// Each copy's bit in the change tracker (`crate::watch::clean_bit`).
    bits: Vec<Option<u32>>,
    blocks: Vec<usize>,
    owner: Vec<u32>,
    /// Entries naming an arena page (`crate::arena`): what shaders write
    /// there stays on the card, and reaching it is no miss.
    arena: Vec<bool>,
}

impl Layout {
    fn in_arena(&self, entry: usize) -> bool {
        self.arena.get(entry).copied().unwrap_or(false)
    }
}

/// A 256 KiB dirty-page bitmap buffer kept zeroed while in [`BITMAP_POOL`].
struct Bitmap {
    _resource: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: `data` is the persistent mapping of `_resource`, owned exclusively by
// one `PooledBitmap` while bound or by `BITMAP_POOL` while idle; D3D12
// resources are free-threaded.
unsafe impl Send for Bitmap {}
// SAFETY: as above; `defer_missed` and `defer_after` access the mapping
// sequentially on the submitting thread after the submission's fence waits.
unsafe impl Sync for Bitmap {}

/// Reusable zeroed dirty-page bitmap buffers, capped at [`BITMAP_POOL_CAP`].
static BITMAP_POOL: Mutex<Vec<Bitmap>> = Mutex::new(Vec::new());

impl Bitmap {
    /// Pops a zeroed bitmap from [`BITMAP_POOL`] or allocates and zeroes a new one.
    fn acquire() -> Option<Self> {
        if let Some(bitmap) = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Some(bitmap);
        }
        let (resource, data) = crate::host::host()?.shared_buffer(BITMAP_BYTES)?;
        crate::host::name(&resource, || "title memory written pages".into());
        // SAFETY: `data` is a newly mapped shared buffer of at least `BITMAP_WORDS` dwords.
        unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, BITMAP_WORDS) };
        // SAFETY: `resource` is a live D3D12 buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        Some(Self {
            _resource: resource,
            data: data as usize,
            gpu,
        })
    }
}

/// RAII guard keeping a [`Bitmap`] owned by a `HostList` until `HostList::reset`
/// or `Drop`, ensuring it is zeroed before returning to [`BITMAP_POOL`].
struct PooledBitmap {
    bitmap: Option<Bitmap>,
}

impl PooledBitmap {
    fn new(bitmap: Bitmap) -> Self {
        Self {
            bitmap: Some(bitmap),
        }
    }

    fn gpu(&self) -> u64 {
        self.bitmap.as_ref().expect("bitmap present until drop").gpu
    }

    fn data(&self) -> usize {
        self.bitmap
            .as_ref()
            .expect("bitmap present until drop")
            .data
    }
}

impl Drop for PooledBitmap {
    fn drop(&mut self) {
        let Some(bitmap) = self.bitmap.take() else {
            return;
        };
        // SAFETY: `PooledBitmap` drops only when `HostList::reset` or `Drop`
        // releases its deferred closures after any GPU submission has finished;
        // `bitmap.data` is mapped for `BITMAP_WORDS` dwords.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(bitmap.data as *mut u32, BITMAP_WORDS) };
        // `defer_after` clears non-zero words after execution and un-executed
        // lists start clean; verify and clear any stray non-zero words before pooling.
        if let Some(first) = marks.iter().position(|&m| m != 0) {
            for mark in &mut marks[first..] {
                if *mark != 0 {
                    *mark = 0;
                }
            }
        }
        let overflow = {
            let mut pool = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < BITMAP_POOL_CAP {
                pool.push(bitmap);
                None
            } else {
                Some(bitmap)
            }
        };
        drop(overflow);
    }
}

/// A view heap's page table, shared by every list using title memory with
/// the heap bound, and the view pairs of the copies it names.
struct Shared {
    /// The host heap it was made for (a heap made later at a released one's
    /// address is another).
    host: usize,
    table: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// The pair of null views entries without a copy name.
    null: u32,
    /// Each copy's view pair (its first slot), by copy.
    pairs: HashMap<usize, u32>,
    /// The copies' generation it names ([`crate::shadow::generation`]).
    generation: u64,
    /// The arena chunks' view pairs, by chunk, and the arena's generation
    /// it names ([`crate::arena::generation`]).
    arena_pairs: Vec<u32>,
    arena_generation: u64,
    layout: Arc<Layout>,
}

// SAFETY: `data` is the persistent mapping of `table`, written only under
// the lock; D3D12 resources are free-threaded.
unsafe impl Send for Shared {}

/// Page tables by view heap (the title heap's address, or [`OURS`]).
static SHARED: Mutex<Vec<(usize, Shared)>> = Mutex::new(Vec::new());

fn shared() -> std::sync::MutexGuard<'static, Vec<(usize, Shared)>> {
    SHARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// View heap `key` was released: its page table goes with it (lists that
/// name it hold the table), and a heap made later at its address starts
/// with none of its state.
#[cfg(windows)]
pub(crate) fn release(key: usize) {
    shared().retain(|(k, _)| *k != key);
}

/// Whether view heap `key` has a page table.
#[cfg(test)]
pub(crate) fn has_table(key: usize) -> bool {
    shared().iter().any(|(k, _)| *k == key)
}

/// Runs `f` on the host heap behind view heap `key` while it cannot go
/// away; None if it is gone or has no host heap.
fn with_heap<R>(key: usize, f: impl FnOnce(&HostHeap) -> R) -> Option<R> {
    if key == OURS {
        return heap().map(f);
    }
    crate::descriptor::with_host(key, f)
}

/// Since the last frame line: draws and dispatches with shaders using title
/// memory (by pipeline), page tables written, pages copied back, pages and
/// 256 MiB regions missed (no copy), and the first missed addresses.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    tables: u64,
    written: u64,
    /// Pages shaders wrote that were not copied back: their list read
    /// memory with no copy, so what they computed is not the console's.
    withheld: u64,
    /// The same for ranges written through views (bytes).
    withheld_bytes: u64,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
    /// Misses at addresses with no title memory, not counted
    /// (`crate::arena::unmapped_misses_ignored`): pages, regions.
    unmapped: (u64, u64),
}

static STATS: std::sync::Mutex<Option<Stats>> = std::sync::Mutex::new(None);

fn stats<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    f(STATS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// `bytes` of a range shaders wrote through a view did not go back into
/// title memory: their list read memory with no copy.
pub(crate) fn note_withheld(bytes: u64) {
    stats(|s| s.withheld_bytes += bytes);
}

/// A draw or dispatch of `pipeline` uses title memory.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| *s.uses.entry(pipeline.to_owned()).or_default() += 1);
}

/// Frames presented, and the last one in which a list using title memory
/// read or wrote memory with no copy ([`MISS_NEVER`] if none has).
static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSED_IN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(MISS_NEVER);
const MISS_NEVER: u64 = u64::MAX;

/// Whether a list using title memory in the submission being completed
/// read or wrote memory with no copy (completions run one at a time).
static SUBMISSION_MISSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A submission's lists ran: whether one using title memory read or wrote
/// memory with no copy.
pub(crate) fn note_submission(missed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    SUBMISSION_MISSED.store(missed, Relaxed);
    if missed {
        MISSED_IN.store(FRAMES.load(Relaxed), Relaxed);
    }
}

/// Whether what lists using title memory wrote stays out of title memory
/// (in the host copies, for the GPU): some such list missed memory this
/// frame or the one before. A list that missed computed from zeros, and
/// the lists after it compute from its results in the host copies, the
/// BVH builder's write addresses among them, so withholding only the lists
/// that missed broke the title's own data (run drops1: an AABB's +inf/-inf
/// in a pointer the title followed). The frame before counts too: its
/// lists' results are read in this one, before this frame's first miss.
pub(crate) fn withholding() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    withheld_in(FRAMES.load(Relaxed), MISSED_IN.load(Relaxed))
}

fn withheld_in(frame: u64, missed: u64) -> bool {
    missed != MISS_NEVER && frame <= missed.saturating_add(1)
}

/// Whether withheld writes stay in the copies until the frame's end even
/// when a list of their submission missed (`VICTORY_BVH_RETAKE_LATE=1`,
/// off by default, local test): taken back right after the submission, the
/// ray tracing programs later in the frame never saw the builder's
/// structures (run 165348: of the headers a builder submission wrote in
/// copies, none was left when the readers ran). Nothing of it goes into
/// title memory either way; the risk is the frame's own later GPU work
/// reading results computed from zeros (run shards20).
pub(crate) fn retake_late() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_RETAKE_LATE").is_ok_and(|v| v.trim() == "1"))
}

/// Title memory whose host copies hold what shaders wrote this frame and
/// the frame withheld ([`withholding`]; ranges).
static DROPPED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Shaders wrote `[address, address + len)` of title memory in host copies
/// and the submission withheld it: the copies take title memory's bytes
/// again at their next refresh, from right after the submission if a list
/// of it missed (its results would feed the frame's later submissions:
/// garbage BVH nodes over meshlet buffers, run shards20), else from the
/// frame's end ([`retake_dropped`]).
pub(crate) fn drop_withheld(address: usize, len: usize) {
    if SUBMISSION_MISSED.load(std::sync::atomic::Ordering::Relaxed) && !retake_late() {
        retake_unkept(address, len);
    } else {
        DROPPED.lock().unwrap_or_else(|e| e.into_inner()).push((address, len));
    }
}

/// Values a held queue waits for that lie in `[from, from + len)` of `copy`,
/// whose writes are held back from title memory: the copy's value goes into
/// title memory, so that the queue runs (a fence value written by a shader
/// of a list that missed memory otherwise holds it until the 10 s safety
/// valve, twice in a row, and the title's 30 s watchdog ends the run:
/// runs 100335, 112712, 125814, 132102). Only those 4 or 8 bytes.
pub(crate) fn release_waited(copy: &Shadow, from: u64, len: u64) {
    let waited = crate::queue::waited();
    if waited.is_empty() {
        return;
    }
    let (start, size) = copy.range();
    for (at, bytes) in waited {
        let (word, n) = (at & !3, if bytes > 4 { 2 } else { 1 });
        if at < from || at >= from + len || word < start || word + 4 * n as u64 > start + size {
            continue;
        }
        let Some(v) = copy.host_dwords(word, n, true) else { continue };
        let value = u64::from(v[0]) | v.get(1).map_or(0, |&h| u64::from(h) << 32);
        let mut title = [0u8; 8];
        let readable = crate::host::read_into(at as *const u8, &mut title[..usize::from(bytes.min(8))]);
        if readable && u64::from_le_bytes(title) != value {
            crate::queue::Written::Value(value).write(at as *mut u8, usize::from(bytes));
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                tracing::info!(
                    address = format_args!("{at:#x}"),
                    title = format_args!("{:#x}", u64::from_le_bytes(title)),
                    copy = format_args!("{value:#x}"),
                    "a value a queue waits for went to title memory from a copy whose writes are held back"
                );
            }
        }
    }
}

/// At a frame's end: what the frame withheld leaves the host copies, so
/// that results computed from memory with no copy (garbage record arrays
/// and node pointers of the BVH builder) do not feed later frames whose
/// writes go back (a box16 node's f16 bounds over a title object, run
/// textures9). Each copy holding such memory takes it from title memory at
/// its next refresh, through its tracker.
fn retake_dropped() {
    let dropped = std::mem::take(&mut *DROPPED.lock().unwrap_or_else(|e| e.into_inner()));
    for (address, len) in dropped {
        retake_unkept(address, len);
    }
}

/// Whether pages of large title buffers that the BVH builder's lists wrote
/// keep the GPU's bytes in their copies until the title writes the page
/// (`VICTORY_BVH_KEEP_PAGES=1`, with the arena; off by default, local test;
/// `=2`: not what a submission wrote of which a list using title memory
/// `missed` memory itself, which stays withheld and dropped as before).
/// Such writes never go into title memory (`crate::arena::BACK`), and
/// dropped from the copies at the frame's end they were gone for good: the
/// title has a bottom level unpacked in place once and builds top levels
/// over it from then on, while title memory, which the copies took again,
/// still held the packed one (runs 193641, 203823, 164739: headers back from
/// 0x8002 to 0x8102 at the same address, no page copied back; 76 % of the
/// unpacking dispatches ran in frames that missed some page, so keeping
/// only what frames without a miss wrote would keep little).
///
/// A kept page is the GPU's alone until the title writes it, in the copy
/// the builder wrote (its keeper):
/// * no retake reaches it there ([`drop_withheld`], whole declared ranges
///   too); the other copies holding the page are retaken as ever (what was
///   withheld of them and stayed would go into title memory with their next
///   copy-back of the page);
/// * nothing of it goes into title memory by comparing the keeper with
///   title memory (a later copy-back of the page would carry the kept bytes
///   along: run 162252 with `VICTORY_BVH_ARENA_KEEP`), so later writes to
///   it stay in the copy as well;
/// * a title write ends it: the copy takes the page from title memory at
///   its next refresh, as any page the title wrote.
// shortcut: the copy takes a page the title wrote whole, over what shaders
// wrote elsewhere in it (a bottom level beside one the title just loaded is
// packed again); counted as "taken from title memory again", as are the
// pages of a copy made anew.
// shortcut: pages the builder writes through views that do not mark their
// writes are not known and stay as before (retaken with their range).
pub(crate) fn keeps_pages(missed: bool) -> bool {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let level = || *LEVEL.get_or_init(|| std::env::var("VICTORY_BVH_KEEP_PAGES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    crate::arena::enabled() && keeps_at(level(), missed)
}

fn keeps_at(level: u8, missed: bool) -> bool {
    match level {
        1 => true,
        2 => !missed,
        _ => false,
    }
}

/// The kept pages ([`keeps_pages`]; page numbers) with the copy each is
/// kept in (never made alive from here: a copy's last holder is not us) and
/// its tracker, and since the last frame line: pages that became kept, kept
/// pages their copy took from title memory again (the title wrote them) or
/// whose copy is gone, copy-backs not made, retakes not made.
#[derive(Default)]
struct Kept {
    pages: std::collections::BTreeMap<usize, (std::sync::Weak<Shadow>, crate::watch::Keeper)>,
    entered: u64,
    ended: u64,
    refused: u64,
    spared: u64,
}

static KEPT: Mutex<Option<Kept>> = Mutex::new(None);
/// Whether any page was ever kept (the paths below cost one load until then).
static KEPT_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn kept<R>(f: impl FnOnce(&mut Kept) -> R) -> R {
    f(KEPT.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// The builder's list wrote the 64 KiB page at `address` in `copy`, and the
/// write does not go into title memory: the page is kept in that copy, if
/// it is a large one of tracked memory and the title has not written the
/// page since the copy took it (then the copy takes the title's bytes at
/// its next refresh anyway; a copy of memory not tracked takes them at
/// every use). True if the page is kept now.
pub(crate) fn keep_page(copy: &Arc<Shadow>, address: usize) -> bool {
    if copy.range().1 <= crate::arena::BACK || copy.title_changed(address) != Some(false) {
        return false;
    }
    let Some(keeper) = copy.keeper() else {
        return false;
    };
    kept(|k| {
        let before = k.pages.insert(address / crate::watch::PAGE, (Arc::downgrade(copy), keeper));
        if !before.is_some_and(|(c, _)| std::ptr::eq(c.as_ptr(), Arc::as_ptr(copy))) {
            k.entered += 1;
        }
    });
    KEPT_ANY.store(true, std::sync::atomic::Ordering::Release);
    true
}

/// Whether the 64 KiB page at `address` is kept.
#[cfg(test)]
pub(crate) fn kept_page(address: usize) -> bool {
    KEPT_ANY.load(std::sync::atomic::Ordering::Acquire) && kept(|k| k.pages.contains_key(&(address / crate::watch::PAGE)))
}

/// Whether the page at `address` is kept in `copy`: a copy-back of the page
/// out of it is not made then (counted). Small copies hold no kept writes.
pub(crate) fn kept_from_title(copy: &Shadow, address: usize) -> bool {
    if copy.range().1 <= crate::arena::BACK || !KEPT_ANY.load(std::sync::atomic::Ordering::Acquire) {
        return false;
    }
    kept(|k| {
        let held = k.pages.get(&(address / crate::watch::PAGE)).is_some_and(|(c, _)| std::ptr::eq(c.as_ptr(), copy));
        k.refused += u64::from(held);
        held
    })
}

/// The large copy `copy` took `[from, to)` from title memory (the title
/// wrote there): the pages kept in it are kept no longer, nor those whose
/// copy is gone.
pub(crate) fn kept_taken(copy: &Shadow, from: usize, to: usize) {
    if !KEPT_ANY.load(std::sync::atomic::Ordering::Acquire) || from >= to {
        return;
    }
    let page = crate::watch::PAGE;
    kept(|k| {
        let gone: Vec<usize> = k
            .pages
            .range(from / page..to.div_ceil(page))
            .filter(|(_, (c, _))| std::ptr::eq(c.as_ptr(), copy) || c.strong_count() == 0)
            .map(|(&p, _)| p)
            .collect();
        for p in &gone {
            k.pages.remove(p);
        }
        k.ended += gone.len() as u64;
    });
}

/// `[address, address + len)` without the pages `kept` (page numbers, in
/// order): the ranges between them.
fn around_pages(address: usize, len: usize, kept: impl Iterator<Item = usize>) -> Vec<(usize, usize)> {
    let page = crate::watch::PAGE;
    let end = address.saturating_add(len);
    let (mut at, mut out) = (address, Vec::new());
    for p in kept {
        let (from, to) = ((p * page).min(end), (p + 1) * page);
        if at < from {
            out.push((at, from - at));
        }
        at = at.max(to);
    }
    if at < end {
        out.push((at, end - at));
    }
    out
}

/// [`crate::watch::retake`] of the range, but of its kept pages only for
/// the copies they are not kept in. A page whose copy is gone is kept no
/// longer.
fn retake_unkept(address: usize, len: usize) {
    if !KEPT_ANY.load(std::sync::atomic::Ordering::Acquire) {
        return crate::watch::retake(address, len);
    }
    let page = crate::watch::PAGE;
    let (ranges, keepers) = kept(|k| {
        let range = address / page..address.saturating_add(len).div_ceil(page);
        let gone: Vec<usize> = k.pages.range(range.clone()).filter(|(_, (c, _))| c.strong_count() == 0).map(|(&p, _)| p).collect();
        for p in &gone {
            k.pages.remove(p);
        }
        k.ended += gone.len() as u64;
        let keepers: Vec<(usize, crate::watch::Keeper)> = k.pages.range(range).map(|(&p, (_, t))| (p, t.clone())).collect();
        k.spared += keepers.len() as u64;
        (around_pages(address, len, keepers.iter().map(|k| k.0)), keepers)
    });
    for (a, n) in ranges {
        crate::watch::retake(a, n);
    }
    for (p, keeper) in keepers {
        keeper.retake_others(p);
    }
}

/// A line per frame in which the kept pages did something.
fn log_kept() {
    if !KEPT_ANY.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let (held, entered, ended, refused, spared) = kept(|k| {
        let line = (k.pages.len(), k.entered, k.ended, k.refused, k.spared);
        (k.entered, k.ended, k.refused, k.spared) = (0, 0, 0, 0);
        line
    });
    if entered + ended + refused + spared > 0 {
        tracing::info!(
            "kept pages this frame: {held} held, {entered} new, {ended} taken from title memory again, \
             {refused} copy-backs not made, {spared} retakes not made"
        );
    }
}

/// Logs what title memory did since the last call, if anything: a line per
/// presented frame while shaders use it.
pub(crate) fn log_frame() {
    FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    retake_dropped();
    log_kept();
    crate::unpacked::log_frame();
    crate::arena::log_frame();
    let s = stats(std::mem::take);
    if s.uses.is_empty() && s.written + s.withheld + s.withheld_bytes + s.missed_pages + s.missed_regions == 0 {
        return;
    }
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    let missed: Vec<String> = s.first_missed.iter().map(|a| format!("{a:#x}")).collect();
    tracing::info!(
        "title memory this frame: {} draws/dispatches ({}), {} page tables, {} pages \
         copied back, {} withheld and {:.1} MB of views kept (their list read memory with no copy), missed {} pages \
         and {} 256 MiB regions with no copy (first: {}); not counted (no title memory there) {} pages {} regions",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.tables,
        s.written,
        s.withheld,
        s.withheld_bytes as f64 / 1e6,
        s.missed_pages,
        s.missed_regions,
        missed.join(" "),
        s.unmapped.0,
        s.unmapped.1,
    );
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

/// What a page table names besides the copies (`crate::arena`): arena
/// pages, each its title address, its chunk's view pair and its byte offset
/// in the chunk; and 256 MiB regions that get a block with no page named in
/// them yet (so that shaders' marks there name pages).
#[derive(Default)]
struct Extra {
    pages: Vec<(u64, u32, u32)>,
    regions: Vec<usize>,
}

/// Writes the page table for host copies `copies` (start, bytes), copy `k`'s
/// view pair at heap index `pairs[k]`, and `null` for unowned pages directly into
/// `out` (which must hold at least [`TABLE_DWORDS`] dwords), with arena pages
/// (`extra`) where no copy holds the page, after the copies. Returns
/// `(written_dwords, blocks, owner, arena)` where `owner` holds copy indices or
/// [`NO_OWNER`] and `arena` whether each entry names an arena page. At most
/// [`CAPACITY`] blocks besides the null one.
fn write_table(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
    extra: &Extra,
    out: &mut [u32],
) -> (usize, Vec<usize>, Vec<u32>, Vec<bool>) {
    assert!(out.len() >= TABLE_DWORDS);
    let mut order: Vec<usize> = (0..copies.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
    // Build level-1 in cached stack memory so we never read from write-combined upload memory.
    let mut level1 = [0u32; LEVEL1];
    // Block 0 is the null block (no copy).
    let mut blocks = vec![usize::MAX];
    let mut owner = vec![NO_OWNER; BLOCK];
    let mut full = false;
    for &k in &order {
        let (start, len) = copies[k];
        let Some(end) = start.checked_add(len) else {
            continue;
        };
        if len == 0 || end > 1 << 40 {
            continue;
        }
        for page in start >> 16..=(end - 1) >> 16 {
            let l1 = (page >> 12) as usize;
            if level1[l1] == 0 {
                if blocks.len() > CAPACITY {
                    full = true;
                    continue;
                }
                level1[l1] = blocks.len() as u32;
                blocks.push(l1);
                owner.resize(owner.len() + BLOCK, NO_OWNER);
            }
            let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it only takes it if free.
            let whole = page << 16 >= start && (page + 1) << 16 <= end;
            if whole || owner[entry] == NO_OWNER {
                owner[entry] = k as u32;
            }
        }
    }
    if full {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(CAPACITY, "title memory copies in more 256 MiB regions than a page table has room for: the rest read as missing"));
    }
    // Arena pages, where no copy holds the page (regions first: a region
    // wanted alone gets its block as a page's region does).
    let mut arena: Vec<Option<(u32, u32)>> = Vec::new();
    let block_of = |l1: usize, level1: &mut [u32], blocks: &mut Vec<usize>, owner: &mut Vec<u32>| -> bool {
        if level1[l1] == 0 {
            if blocks.len() > CAPACITY {
                return false;
            }
            level1[l1] = blocks.len() as u32;
            blocks.push(l1);
            owner.resize(owner.len() + BLOCK, NO_OWNER);
        }
        true
    };
    let mut arena_full = false;
    for &l1 in &extra.regions {
        if l1 < LEVEL1 && !block_of(l1, &mut level1, &mut blocks, &mut owner) {
            arena_full = true;
        }
    }
    for &(at, pair, offset) in &extra.pages {
        let page = at >> 16;
        let l1 = (page >> 12) as usize;
        if l1 >= LEVEL1 || !block_of(l1, &mut level1, &mut blocks, &mut owner) {
            arena_full |= l1 < LEVEL1;
            continue;
        }
        let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
        if owner[entry] == NO_OWNER {
            arena.resize(owner.len(), None);
            // The shader's byte offset is the address's low dword minus the
            // entry's: the page's offset in its chunk.
            arena[entry] = Some((pair, (at as u32).wrapping_sub(offset)));
        }
    }
    if arena_full {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(CAPACITY, "arena pages in more 256 MiB regions than a page table has room for: the rest read as missing"));
    }
    arena.resize(owner.len(), None);
    out[..LEVEL1].copy_from_slice(&level1);
    let l2_dwords = owner.len() * 2;
    let l2 = &mut out[LEVEL1..LEVEL1 + l2_dwords];
    let (slots, _) = l2.as_chunks_mut::<2>();
    for ((slot, &o), a) in slots.iter_mut().zip(&owner).zip(&arena) {
        match (o, *a) {
            (NO_OWNER, Some(a)) => {
                slot[0] = a.0;
                slot[1] = a.1;
            }
            (NO_OWNER, None) => {
                slot[0] = null;
                slot[1] = 0;
            }
            (k, _) => {
                let k = k as usize;
                slot[0] = pairs[k];
                slot[1] = copies[k].0 as u32;
            }
        }
    }
    let backed = arena.iter().map(Option::is_some).collect();
    (LEVEL1 + l2_dwords, blocks, owner, backed)
}

/// [`write_table`] into a fresh vector, owners as options: the table's
/// dwords, the level-1 index of each block, and the copy each entry names.
#[cfg(test)]
fn table(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let (dwords, blocks, owner, _) = table_with(copies, pairs, null, &Extra::default());
    (dwords, blocks, owner)
}

/// [`table`] with arena pages where no copy holds the page, after the
/// copies: also whether each entry names an arena page.
#[cfg(test)]
fn table_with(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
    extra: &Extra,
) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>, Vec<bool>) {
    let mut dwords = vec![0u32; TABLE_DWORDS];
    let (len, blocks, owner, arena) = write_table(copies, pairs, null, extra, &mut dwords);
    dwords.truncate(len);
    let owner = owner
        .into_iter()
        .map(|o| (o != NO_OWNER).then_some(o as usize))
        .collect();
    (dwords, blocks, owner, arena)
}

impl Shared {
    /// A page table for `heap`, named nothing yet (written at the first
    /// submission that uses it).
    fn new(heap: &HostHeap) -> Option<Shared> {
        let bytes = TABLE_DWORDS as u64 * 4;
        let (table, data) = crate::host::host()?.upload_buffer(bytes)?;
        crate::host::name(&table, || "title memory page table".into());
        let null = heap.pair()?;
        heap.raw_pair_at(null, None, 0);
        // SAFETY: a live buffer.
        let gpu = unsafe { table.GetGPUVirtualAddress() };
        Some(Shared {
            host: std::ptr::from_ref(heap) as usize,
            table,
            data: data as usize,
            gpu,
            null,
            pairs: HashMap::new(),
            generation: u64::MAX,
            arena_pairs: Vec::new(),
            arena_generation: 0,
            layout: Arc::default(),
        })
    }

    /// Names every host copy there is now (`now`: their generation): view
    /// pairs made for new copies, those of copies gone made null and given
    /// back, the table rewritten. Only while no submission runs (the GPU
    /// reads the table and the views).
    fn rewrite(&mut self, heap: &HostHeap, now: u64) {
        let copies = crate::shadow::all();
        let live: HashSet<usize> = copies.iter().map(|c| Arc::as_ptr(c) as usize).collect();
        self.pairs.retain(|copy, &mut slot| {
            let keep = live.contains(copy);
            if !keep {
                heap.raw_pair_at(slot, None, 0);
                heap.free_pair(slot);
            }
            keep
        });
        let (mut named, mut slots) = (Vec::with_capacity(copies.len()), Vec::with_capacity(copies.len()));
        for c in copies {
            let key = Arc::as_ptr(&c) as usize;
            let slot = match self.pairs.get(&key) {
                Some(&slot) => slot,
                None => {
                    let Some(slot) = heap.pair() else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| tracing::warn!("no descriptor slots for a title memory copy's views: it reads as missing"));
                        continue;
                    };
                    heap.raw_pair_at(slot, Some(&c.buffer), c.range().1.next_multiple_of(4));
                    self.pairs.insert(key, slot);
                    slot
                }
            };
            named.push(c);
            slots.push(slot);
        }
        let ranges: Vec<(u64, u64)> = named.iter().map(|c| c.range()).collect();
        // Arena pages, through a view pair per chunk.
        let mut extra = Extra::default();
        if crate::arena::enabled() {
            let view = crate::arena::view();
            for chunk in &view.chunks[self.arena_pairs.len().min(view.chunks.len())..] {
                let Some(slot) = heap.pair() else {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| tracing::warn!("no descriptor slots for a BVH arena chunk's views: its pages read as missing"));
                    break;
                };
                heap.raw_pair_at(slot, Some(chunk), crate::arena::CHUNK_BYTES);
                self.arena_pairs.push(slot);
            }
            extra.pages = view
                .pages
                .iter()
                .filter_map(|&(at, s)| Some((at, *self.arena_pairs.get(s.chunk())?, s.offset() as u32)))
                .collect();
            extra.regions = view.regions;
            self.arena_generation = view.generation;
        }
        // SAFETY: `self.data` is the persistent mapping of `self.table`, with room
        // for `TABLE_DWORDS` dwords; no submission runs.
        let out = unsafe { std::slice::from_raw_parts_mut(self.data as *mut u32, TABLE_DWORDS) };
        let (_, blocks, owner, arena) = write_table(&ranges, &slots, self.null, &extra, out);
        tracing::debug!(copies = ranges.len(), blocks = blocks.len() - 1, "title memory page table");
        self.layout = Arc::new(Layout {
            bits: named.iter().map(|c| c.dirty_bit()).collect(),
            copies: named,
            blocks,
            owner,
            arena,
        });
        self.generation = now;
        stats(|s| s.tables += 1);
    }
}

/// The page table of view heap `key`, as the last submission using it
/// left it.
fn layout(key: usize) -> Arc<Layout> {
    shared()
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, s)| s.layout.clone())
        .unwrap_or_default()
}

/// At submission, for a list using title memory with view heap `key`
/// bound: the heap's page table names every host copy there is, each
/// brought up to date (`add` takes a copy's range, once per submission
/// until a step writes title memory).
pub(crate) fn prepare(key: usize, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let layout = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        let (_, s) = shared.iter_mut().find(|(k, s)| *k == key && s.host == host)?;
        let now = crate::shadow::generation();
        if s.generation != now || (crate::arena::enabled() && s.arena_generation != crate::arena::generation()) {
            s.rewrite(heap, now);
        }
        Some(s.layout.clone())
    })
    .flatten();
    // Copies the title has not written since they took it are passed by
    // without touching them (the table names every copy, ~4,700 in the world).
    for l in layout.iter() {
        for (c, bit) in l.copies.iter().zip(&l.bits) {
            if !crate::watch::clean_bit(*bit) {
                let (start, len) = c.range();
                add(c, start as usize, len as usize);
            }
        }
    }
}

/// Whether any marked page in `marks` fell outside a valid copy in `l`.
fn has_missed(marks: &[u32], l: &Layout) -> bool {
    // `VICTORY_BVH_MISS_UNMAPPED=ignore`: marks where no title memory is
    // are no miss.
    let ignored = crate::arena::unmapped_misses_ignored();
    for (w, &mark) in marks.iter().enumerate() {
        let mut bits = mark;
        while bits != 0 {
            let entry = w * 32 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let block = entry / BLOCK;
            if block == 0 {
                if !ignored || crate::arena::title_memory_in_region(entry) {
                    return true;
                }
                continue;
            }
            let Some(&l1) = l.blocks.get(block) else {
                return true;
            };
            if l.owner.get(entry).is_none_or(|&o| o == NO_OWNER)
                && !l.in_arena(entry)
                && (!ignored || crate::arena::title_memory_at((l1 << 28 | (entry % BLOCK) << 16) as u64))
            {
                return true;
            }
        }
    }
    false
}

/// `list` uses title memory with view heap `key` bound: the heap's page
/// table (made on first use) and a page bitmap of the list's, the table
/// brought up to date at each submission of the list, its written pages
/// copied back after.
pub(crate) fn bind(list: &HostList, key: usize) -> Option<Map> {
    let table = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        if let Some((_, s)) = shared.iter().find(|(k, s)| *k == key && s.host == host) {
            list.keep(&s.table);
            return Some(s.gpu);
        }
        // A released heap's table goes with it (lists that name it hold it).
        shared.retain(|(k, _)| *k != key);
        let s = Shared::new(heap)?;
        list.keep(&s.table);
        let gpu = s.gpu;
        shared.push((key, s));
        Some(gpu)
    })
    .flatten()?;
    let pooled = Arc::new(PooledBitmap::new(Bitmap::acquire()?));
    let dirty = pooled.gpu();
    list.defer_memory(key);
    let precise: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let byte_for_byte = precise.clone();
    let missed_bitmap = Arc::clone(&pooled);
    let builder: Arc<std::sync::atomic::AtomicBool> = Arc::default();
    let ran_builder = builder.clone();
    // `VICTORY_BVH_GT`: the marked pages as `defer_after` below finds them,
    // read before it clears the bitmap.
    if crate::bvh_gt::target().is_some() {
        let gt_bitmap = Arc::clone(&pooled);
        list.defer_gt(move || {
            let l = layout(key);
            // SAFETY: the bitmap is mapped and the list's run is over.
            let marks = unsafe {
                std::slice::from_raw_parts(gt_bitmap.data() as *const u32, BITMAP_WORDS)
            };
            let mut m = crate::bvh_gt::Marks::default();
            for (w, &mark) in marks.iter().enumerate() {
                let mut bits = mark;
                while bits != 0 {
                    let entry = w * 32 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let (block, page) = (entry / BLOCK, entry % BLOCK);
                    let Some(&l1) = l.blocks.get(block).filter(|_| block != 0) else {
                        m.missed_regions += 1;
                        if m.first_missed.len() < 3 {
                            m.first_missed.push((entry as u64) << 28);
                        }
                        continue;
                    };
                    let at = l1 << 28 | page << 16;
                    if l.in_arena(entry) {
                        m.arena.push(at);
                        continue;
                    }
                    match l.owner.get(entry) {
                        Some(&k) if k != NO_OWNER => m.owned.push((l.copies[k as usize].clone(), at)),
                        _ if crate::arena::unmapped_misses_ignored() && !crate::arena::title_memory_at(at as u64) => {}
                        _ => {
                            m.missed_pages += 1;
                            if m.first_missed.len() < 3 {
                                m.first_missed.push(at as u64);
                            }
                        }
                    }
                }
            }
            m
        });
    }
    list.defer_missed(move || {
        let l = layout(key);
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts(missed_bitmap.data() as *const u32, BITMAP_WORDS) };
        has_missed(marks, &l)
    });
    list.defer_after(move || {
        let l = layout(key);
        let Layout { copies, blocks, owner, .. } = &*l;
        let arena = crate::arena::enabled();
        // Only the builder's misses ask the arena for pages: other programs'
        // took its whole budget in the loading screens before the builder
        // first ran (run 144238: budget used, then its first submission read
        // zeros and the device hung).
        // (The synthetic test has no builder: its stage stands for one.)
        let asks = arena && (ran_builder.load(std::sync::atomic::Ordering::Relaxed) || cfg!(test));
        let precise = byte_for_byte.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(pooled.data() as *mut u32, BITMAP_WORDS) };
        let (mut pages, mut regions, mut missed, mut on_card) = (0, 0, Vec::new(), 0);
        let mut wanted = Vec::new();
        let mut shader_pages = Vec::new();
        // With the arena, the builder's writes to title buffers that do not
        // go back stay in the copies (`crate::arena::keep_builder_writes`):
        // taken back from title memory they were lost before the next
        // submission's stages read them (runs 152953-161111: G headers in
        // copies invalid, those in arena pages valid).
        let keep = arena && ran_builder.load(std::sync::atomic::Ordering::Relaxed) && crate::arena::keep_builder_writes();
        let mut writes = Vec::new();
        let ignored = crate::arena::unmapped_misses_ignored();
        let mut unmapped = (0, 0);
        for (w, mark) in marks.iter_mut().enumerate() {
            if *mark == 0 {
                continue;
            }
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                // Block 0's bits are 256 MiB regions with no block.
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    if ignored && block == 0 && !crate::arena::title_memory_in_region(entry) {
                        unmapped.1 += 1;
                        continue;
                    }
                    regions += 1;
                    missed.push((entry as u64) << 28);
                    if asks {
                        crate::arena::want_region(entry);
                    }
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                // Arena pages: what shaders wrote stays on the card.
                if l.in_arena(entry) {
                    on_card += 1;
                    shader_pages.push(at);
                    continue;
                }
                match owner.get(entry) {
                    Some(_) if precise.iter().any(|&(a, n)| a < at + (1 << 16) && at < a + n) => {}
                    Some(&k) if k != NO_OWNER => writes.push((k as usize, at)),
                    _ if ignored && !crate::arena::title_memory_at(at) => unmapped.0 += 1,
                    _ => {
                        pages += 1;
                        missed.push(at);
                        if asks {
                            wanted.push(at);
                        }
                    }
                }
            }
        }
        if on_card > 0 {
            crate::arena::note_writes(on_card);
            crate::arena::note_shader_pages(&shader_pages);
        }
        crate::arena::want_pages(&wanted);
        // What shaders computed from memory with no copy (read as zeros) is
        // not the console's: it does not go into title memory, where it broke
        // the title's own data (garbage pointers in title threads, run c1),
        // nor while any list using title memory misses ([`withholding`]).
        let mut withheld = if pages + regions > 0 || withholding() { writes.len() as u64 } else { 0 };
        // The builder's pages in large copies that do not go back: kept,
        // each until the title writes it ([`keeps_pages`]).
        let keeps = arena && ran_builder.load(std::sync::atomic::Ordering::Relaxed) && keeps_pages(pages + regions > 0);
        if withheld > 0 {
            for (k, at) in writes.drain(..) {
                release_waited(&copies[k], at, 1 << 16);
                if !keep && !(keeps && keep_page(&copies[k], at as usize)) {
                    drop_withheld(at as usize, 1 << 16);
                }
            }
        } else if arena && ran_builder.load(std::sync::atomic::Ordering::Relaxed) {
            // With the arena, the builder's writes go back only into small
            // title buffers (the results the title reads); the rest stays in
            // the copies for the GPU until the frame's end.
            writes.retain(|&(k, at)| {
                let small = copies[k].range().1 <= crate::arena::BACK;
                if !small {
                    withheld += 1;
                    release_waited(&copies[k], at, 1 << 16);
                    if !keep && !(keeps && keep_page(&copies[k], at as usize)) {
                        drop_withheld(at as usize, 1 << 16);
                    }
                }
                small
            });
        }
        let mut written = 0;
        for &(k, at) in &writes {
            let n = copies[k].copy_back_page(at as usize);
            tracing::debug!(
                page = format_args!("{at:#x}"),
                copy = format_args!("{:#x?}", copies[k].range()),
                bytes = n,
                "title memory page written back"
            );
            written += 1;
        }
        stats(|s| {
            s.written += written;
            s.withheld += withheld;
            s.missed_pages += pages;
            s.missed_regions += regions;
            s.unmapped.0 += unmapped.0;
            s.unmapped.1 += unmapped.1;
            let room = 4usize.saturating_sub(s.first_missed.len());
            s.first_missed.extend(missed.into_iter().take(room));
        });
    });
    Some(Map {
        table,
        dirty,
        precise,
        builder,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dropped range is retaken around its kept pages, cut to the range.
    #[test]
    fn a_range_is_retaken_around_its_kept_pages() {
        let p = crate::watch::PAGE;
        assert_eq!(around_pages(p + 16, 4 * p, [2, 3].into_iter()), [(p + 16, p - 16), (4 * p, p + 16)]);
        assert_eq!(around_pages(0, 3 * p, std::iter::empty()), [(0, 3 * p)]);
        assert!(around_pages(2 * p, p, [2].into_iter()).is_empty());
        assert!(around_pages(2 * p + 8, 16, [2].into_iter()).is_empty());
        assert_eq!(around_pages(p, 4 * p, [1, 4].into_iter()), [(2 * p, 2 * p)]);
        assert!(!keeps_at(0, false) && keeps_at(1, true) && keeps_at(2, false) && !keeps_at(2, true) && !keeps_at(3, false));
    }

    /// Writes of lists using title memory are held back from a frame with a
    /// miss through the next one, then go back again.
    #[test]
    fn a_miss_holds_back_its_frame_and_the_next() {
        assert!(!withheld_in(0, MISS_NEVER));
        assert!(!withheld_in(u64::MAX, MISS_NEVER));
        assert!(withheld_in(7, 7));
        assert!(withheld_in(8, 7), "the next frame reads this one's results");
        assert!(!withheld_in(9, 7));
    }

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }

    /// Copies in more regions than a table has room for: the largest are
    /// named, the rest read as missing; the table fits its buffer.
    #[test]
    fn a_full_table_names_the_largest_copies() {
        let copies: Vec<(u64, u64)> = (0..CAPACITY as u64 + 2).map(|k| (k << 28, 0x1_0000 + k)).collect();
        let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 10 + 2 * k).collect();
        let (dwords, blocks, _) = table(&copies, &pairs, 4);
        assert_eq!(blocks.len(), CAPACITY + 1);
        assert_eq!(dwords.len(), LEVEL1 + (CAPACITY + 1) * BLOCK * 2);
        assert_eq!(look_up(&dwords, (CAPACITY as u64 + 1) << 28).0, pairs[CAPACITY + 1]);
        assert_eq!(look_up(&dwords, 0).0, 4, "the smallest copy is left out");
    }

    #[test]
    fn table_skips_overflowing_and_out_of_range_copies() {
        let copies = [
            (u64::MAX - 0x100, 0x200),
            (1u64 << 40, 0x1_0000),
            (0x40_0000, 0x1_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        assert_eq!(blocks, vec![usize::MAX, 0]);
        assert_eq!(look_up(&dwords, 0x40_0010), (106, 0x10));
        assert_eq!(look_up(&dwords, 0).0, 100);
        assert_eq!(owner[BLOCK + 0x40], Some(2));
    }

    #[test]
    fn has_missed_checks_block_zero_out_of_range_unowned_and_owned() {
        let mut owner = vec![NO_OWNER; 2 * BLOCK];
        owner[BLOCK + 5] = 0;
        let l = Layout {
            copies: Vec::new(),
            bits: Vec::new(),
            blocks: vec![usize::MAX, 0],
            owner,
            arena: Vec::new(),
        };
        let mut marks = vec![0u32; BITMAP_WORDS];
        assert!(!has_missed(&marks, &l));

        // Owned entry in block 1.
        let owned = BLOCK + 5;
        marks[owned / 32] = 1 << (owned % 32);
        assert!(!has_missed(&marks, &l));

        // Unowned entry in the same word as an owned entry.
        let unowned = BLOCK + 6;
        marks[unowned / 32] |= 1 << (unowned % 32);
        assert!(has_missed(&marks, &l));

        // Block 0 bit (missing 256 MiB region).
        marks.fill(0);
        marks[0] = 1 << 3;
        assert!(has_missed(&marks, &l));

        // Out-of-range block.
        marks.fill(0);
        let oob = 2 * BLOCK + 1;
        marks[oob / 32] = 1 << (oob % 32);
        assert!(has_missed(&marks, &l));
    }

    fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn bitmap_pool_reuses_and_zeroes_across_resets() {
        let _guard = gpu_test_lock();
        let (Some(host), Some(_views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let a = host.new_list().unwrap();
        let b = host.new_list().unwrap();
        let map_a = bind(&a, OURS).expect("bitmap for list a");
        let map_b = bind(&b, OURS).expect("bitmap for list b");
        let va_a = map_a.dirty;
        let va_b = map_b.dirty;
        assert_ne!(va_a, va_b, "simultaneously open lists get distinct bitmaps");
        drop((map_a, map_b));
        a.close();
        a.reset();
        let map_a2 = bind(&a, OURS).expect("reused bitmap for list a");
        assert_eq!(map_a2.dirty, va_a, "reset returns bitmap to pool for reuse");
        assert_ne!(map_a2.dirty, va_b, "list b still holds its bitmap");
        drop(map_a2);
        a.close();
        b.close();
        a.reset();
        b.reset();

        // Also verify PooledBitmap zeroes dirty words when dropped without executing.
        let pooled = PooledBitmap::new(Bitmap::acquire().unwrap());
        let data = pooled.data();
        // SAFETY: exclusive ownership of this newly acquired bitmap.
        unsafe { (data as *mut u32).add(7).write(0xDEAD_BEEF) };
        drop(pooled);
        let reused = Bitmap::acquire().unwrap();
        // SAFETY: `reused.data` is mapped for `BITMAP_WORDS` dwords.
        let words = unsafe { std::slice::from_raw_parts(reused.data as *const u32, BITMAP_WORDS) };
        assert!(
            words.iter().all(|&w| w == 0),
            "pooled bitmap must be zeroed on return"
        );
        drop(PooledBitmap::new(reused));
    }

    /// Arena pages (`crate::arena`) are named where no copy holds the page,
    /// at their offset in their chunk's view; a region wanted alone gets a
    /// block; copies keep their pages.
    #[test]
    fn arena_pages_fill_what_no_copy_holds() {
        let copies = [(0xdd_0000_0000, 0x2_0000)];
        let extra = Extra {
            pages: vec![
                // In the copy's region, beside it.
                (0xdd_0002_0000, 200, 0x3_0000),
                // Under the copy: the copy keeps it.
                (0xdd_0001_0000, 200, 0x4_0000),
                // In a region of its own, in another chunk.
                (0xe5_1234_0000, 202, 0),
            ],
            regions: vec![0xe8f],
        };
        let (dwords, blocks, owner, arena) = table_with(&copies, &[102], 100, &extra);
        assert_eq!(look_up(&dwords, 0xdd_0002_0010), (200, 0x3_0010));
        assert_eq!(look_up(&dwords, 0xdd_0002_fffc), (200, 0x3_fffc));
        assert_eq!(look_up(&dwords, 0xdd_0001_0004), (102, 0x1_0004), "the copy's page");
        assert_eq!(look_up(&dwords, 0xe5_1234_8000), (202, 0x8000));
        // The pages around stay missing (null pair), as the wanted region.
        assert_eq!(look_up(&dwords, 0xe5_1235_0000).0, 100);
        assert_eq!(look_up(&dwords, 0xe8_f000_0000).0, 100);
        assert_eq!(blocks.len(), 4, "null, the copy's region, 0xe51, the wanted 0xe8f");
        assert!(blocks.contains(&0xe8f));
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK + (a >> 16 & 0xFFF) as usize;
        assert!(arena[entry(0xdd_0002_0000)] && arena[entry(0xe5_1234_0000)]);
        assert!(!arena[entry(0xdd_0001_0000)] && owner[entry(0xdd_0001_0000)] == Some(0));
        assert!(!arena[entry(0xe5_1235_0000)]);
        assert_eq!(arena.len(), owner.len());
        // Without arena pages, the table is the copies' alone.
        let (plain, _, _) = table(&copies, &[102], 100);
        let (with_none, _, _, none) = table_with(&copies, &[102], 100, &Extra::default());
        assert_eq!(plain, with_none);
        assert!(none.iter().all(|&a| !a));
    }

    /// Arena pages past the table's room read as missing; the table fits
    /// its buffer.
    #[test]
    fn arena_pages_past_the_room_read_as_missing() {
        let copies: Vec<(u64, u64)> = (0..CAPACITY as u64).map(|k| (k << 28, 0x1_0000)).collect();
        let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 10 + 2 * k).collect();
        let extra = Extra {
            pages: vec![(0xff_f000_0000, 7, 0), (0xa0_0000_0000, 7, 0x1_0000)],
            regions: vec![],
        };
        let (dwords, blocks, _, _) = table_with(&copies, &pairs, 4, &extra);
        assert_eq!(blocks.len(), CAPACITY + 1);
        assert_eq!(dwords.len(), LEVEL1 + (CAPACITY + 1) * BLOCK * 2);
        assert_eq!(look_up(&dwords, 0xff_f000_0000).0, 4);
    }

    /// Reads 4 dwords of title memory at (`lo`, `hi`) through the page
    /// table and the heap's views, as translated shaders do, into `dst`
    /// from byte `at`.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=3, b0)"
ByteAddressBuffer views[] : register(t0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; uint at; }
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint a = lo + id.x * 4;
    bool inside = hi < 256;
    uint l1 = inside ? (hi << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    dst.Store(at + id.x * 4, views[NonUniformResourceIndex(e.x)].Load(a - e.y));
}
"#;

    fn pipeline(source: &str) -> (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES, D3DCompile};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call; the errors are text.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("memory"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES,
                0,
                &mut code,
                Some(&mut errors),
            );
            let text = errors.map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned());
            assert!(made.is_ok(), "{text:?}");
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes; a compiled shader with its root
        // signature inside.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A copy's view pair, in view heap `key`'s table.
    fn pair_of(key: usize, copy: &Arc<Shadow>) -> Option<u32> {
        shared()
            .iter()
            .find(|(k, _)| *k == key)?
            .1
            .pairs
            .get(&(Arc::as_ptr(copy) as usize))
            .copied()
    }

    /// Two lists of a submission read title memory through one page table;
    /// the copies' views stay where they are for later lists and
    /// submissions, also when another copy comes; none takes a per-draw
    /// dynamic slot.
    #[test]
    fn lists_share_a_table_and_copies_keep_their_views() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let (root, pso) = pipeline(SOURCE);
        let (out, at) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // Title memory (below 2^40, as the runtime places it): 1, 2 and
        // (from the second submission on) 1 page, write-watched as its pools.
        let pool = |pages: usize| -> usize {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            (0..256u64)
                .map(|k| 0x40_0000_0000 + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, pages * PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        let memory = [pool(1), pool(2), pool(1)];
        let copy = |p: usize, pages: usize| {
            crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
            crate::shadow::find(p as u64, (pages * PAGE) as u64).unwrap()
        };
        let (a, b) = (copy(memory[0], 1), copy(memory[1], 2));
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        // Each list reads 4 dwords at its address into its 16 bytes of `out`.
        let record = |reads: [usize; 2]| -> Vec<Map> {
            let maps: Vec<Map> = lists.iter().map(|l| bind(l, OURS).expect("a page table")).collect();
            for (k, (list, &address)) in lists.iter().zip(&reads).enumerate() {
                let constants = [address as u32, (address >> 32) as u32, 16 * k as u32];
                // SAFETY: an open list, live objects of ours.
                unsafe {
                    list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                    list.list.SetComputeRootShaderResourceView(1, maps[k].table);
                    list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    list.list.SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
                    list.list.SetPipelineState(&pso);
                    list.list.Dispatch(1, 1, 1);
                }
                list.close();
            }
            maps
        };
        // The title writes after recording, before submitting.
        let fill = |p: usize, base: u32| {
            for k in 0..4 {
                // SAFETY: our pool (a guarded page faults into the handler).
                unsafe { (p as *mut u32).add(k).write_volatile(base + k as u32) };
            }
        };
        let read = || -> Vec<u32> {
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { std::slice::from_raw_parts(at.cast::<u32>(), 8).to_vec() }
        };
        let maps = record([memory[0] + 16, memory[1] + PAGE]);
        assert_eq!(maps[0].table, maps[1].table, "one table for the heap");
        assert_ne!(maps[0].dirty, maps[1].dirty, "a bitmap per list");
        fill(memory[0] + 16, 0x100);
        fill(memory[1] + PAGE, 0x200);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x100, 0x101, 0x102, 0x103, 0x200, 0x201, 0x202, 0x203]);
        let pairs = (pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap());
        assert!(pairs.0 >= 1 << 18 && pairs.1 >= 1 << 18, "past the dynamic slots");
        // A second submission, after a third copy came.
        let c = copy(memory[2], 1);
        for list in &lists {
            list.reset();
        }
        record([memory[2] + 32, memory[0] + 16]);
        fill(memory[2] + 32, 0x300);
        fill(memory[0] + 16, 0x400);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x300, 0x301, 0x302, 0x303, 0x400, 0x401, 0x402, 0x403]);
        assert_eq!((pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap()), pairs, "views kept");
        assert!(pair_of(OURS, &c).is_some());
        for (&p, copy) in memory.iter().zip([a, b, c]) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
        }
    }

    /// Writes `value` at (`lo`, `hi`) through the page table, marking its
    /// page as translated shaders do, and with `missing_hi` non-zero reads
    /// (`missing_lo`, `missing_hi`), marking it if no copy holds it.
    const WRITE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0, space=2), RootConstants(num32BitConstants=5, b0)"
RWByteAddressBuffer views[] : register(u0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0, space2);
cbuffer c : register(b0) { uint lo; uint hi; uint value; uint missing_lo; uint missing_hi; }
void access(uint a, uint h, bool write) {
    bool inside = h < 256;
    uint l1 = inside ? (h << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    bool miss = e.x == table.Load(4096 * 4);
    uint n = miss && block == 0 ? l1 : entry;
    if (write || miss) {
        dirty.InterlockedOr(n >> 5 << 2, 1u << (n & 31));
    }
    if (write && !miss) {
        views[NonUniformResourceIndex(e.x + 1)].Store(a - e.y, value);
    }
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    access(lo, hi, true);
    if (missing_hi != 0) {
        access(missing_lo, missing_hi, false);
    }
}
"#;

    /// What a submission that missed memory wrote through the page table
    /// leaves the host copies right after it, so the next submission reads
    /// title memory's bytes (run shards20: garbage BVH nodes over meshlet
    /// buffers until the frame's end); one that missed nothing leaves its
    /// writes in the copies.
    #[test]
    fn a_submission_that_missed_leaves_the_copies_at_once() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let ((write_root, writer), (read_root, reader)) = (pipeline(WRITE), pipeline(SOURCE));
        let (out, out_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // A page of title memory below 2^40, write-watched as the pools.
        let p = (0..256u64)
            .map(|k| 0x50_0000_0000 + (k << 28))
            // SAFETY: a fresh allocation where nothing is mapped, or none.
            .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
            .find(|&p| p != 0)
            .expect("memory below 2^40");
        let at = p + 16;
        // SAFETY: our pool, not guarded yet.
        unsafe { (at as *mut u32).write_volatile(0x11) };
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        // One submission of one dispatch; the list is open again after it.
        let submit = |root: &windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
                      pso: &windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
                      uav: Option<u64>,
                      constants: &[u32]| {
            let map = bind(&list, OURS).expect("a page table");
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, uav.unwrap_or(map.dirty));
                list.list.SetComputeRoot32BitConstants(3, constants.len() as u32, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
        };
        let write = |value: u32, miss: bool| {
            // Memory no copy holds, in a 256 MiB region with no block.
            let missing: u64 = if miss { 0xff_f000_0000 } else { 0 };
            let constants = [at as u32, (at >> 32) as u32, value, missing as u32, (missing >> 32) as u32];
            submit(&write_root, &writer, None, &constants);
        };
        let read = || -> u32 {
            submit(&read_root, &reader, Some(out_gpu), &[at as u32, (at >> 32) as u32, 0]);
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { out_cpu.cast::<u32>().read_volatile() }
        };
        // SAFETY: our pool.
        let title = || unsafe { (at as *const u32).read_volatile() };
        write(0x33, true);
        // Another test's submission may have refreshed the copy already.
        let held = copy.read_host(at as u64, 1)[0];
        assert!(!copy.clean() || held == 0x11, "the written page counts as changed at once");
        assert_eq!(title(), 0x11, "nothing went back");
        assert_eq!(read(), 0x11, "the next submission reads title memory's bytes");
        // Unless another test ends a frame meanwhile (its retake is right).
        let frame = FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        write(0x44, false);
        let kept = read();
        if FRAMES.load(std::sync::atomic::Ordering::Relaxed) == frame {
            assert_eq!(kept, 0x44, "a submission that missed nothing keeps its writes");
        }
        crate::shadow::unregister(p, p as u64);
        drop(copy);
    }

    /// Mode 0: 64 threads each copy a dword from (`q_lo`, `q_hi`) to
    /// (`p_lo`, `p_hi`) through the page table, as a builder stage reads its
    /// inputs and writes a structure; with `missing_hi` non-zero thread 0
    /// also reads (`missing_lo`, `missing_hi`). Mode 1: the 64 dwords at
    /// (`p_lo`, `p_hi`) through the page table into `result`, as a ray
    /// tracing program reads a structure. Misses read 0 and are marked.
    const BUILD: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0, space=2), UAV(u1, space=2), RootConstants(num32BitConstants=7, b0)"
RWByteAddressBuffer views[] : register(u0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0, space2);
RWByteAddressBuffer result : register(u1, space2);
cbuffer c : register(b0) { uint p_lo; uint p_hi; uint q_lo; uint q_hi; uint mode; uint missing_lo; uint missing_hi; }
uint access(uint a, uint h, bool write, uint value) {
    bool inside = h < 256;
    uint l1 = inside ? (h << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    bool miss = e.x == table.Load(4096 * 4);
    uint n = miss && block == 0 ? l1 : entry;
    if (write || miss) {
        dirty.InterlockedOr(n >> 5 << 2, 1u << (n & 31));
    }
    if (miss) {
        return 0;
    }
    if (write) {
        views[NonUniformResourceIndex(e.x + 1)].Store(a - e.y, value);
        return value;
    }
    return views[NonUniformResourceIndex(e.x + 1)].Load(a - e.y);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint o = id.x * 4;
    if (mode == 0) {
        access(p_lo + o, p_hi, true, access(q_lo + o, q_hi, false, 0));
        if (id.x == 0 && missing_hi != 0) {
            access(missing_lo, missing_hi, false, 0);
        }
    } else {
        result.Store(o, access(p_lo + o, p_hi, false, 0));
    }
}
"#;

    /// bvh-test-plan (1), synthetic: only the way from a builder stage's
    /// writes to the readers. A stage reads a small structure
    /// (`crate::bvh_gt::mini_bvh`) from page Q and writes it into page P
    /// (title memory with a copy, holding older bytes); readers read P
    /// through the page table (i) in the same list, (ii) in the next
    /// submission, (iii) after the frame's end, and the CPU reads P after
    /// `Host::execute`. Three cases:
    /// * V1: Q has a copy, nothing missed: everyone sees the structure.
    /// * V3: Q has a copy, the submission also reads memory with no copy
    ///   (case B of the plan): (i) sees the right structure, (ii), (iii)
    ///   and the CPU the older bytes (`drop_withheld`, `watch::retake`).
    /// * V2: Q is private memory with no copy (as the builder's pools):
    ///   the stage reads zeros (case A1), (i) sees the zeros, the others
    ///   the older bytes.
    ///
    /// Asserts what the current code does, after printing what each reader
    /// saw. Fails without a GPU (no silent pass).
    ///
    /// Run alone, in an interactive session, with no other GPU work:
    /// `cargo test --release -p victory-gpu --features diagnostics a_bvh_written_from_pages -- --ignored --test-threads=1 --nocapture`
    #[test]
    #[ignore = "needs a D3D12 GPU; run alone with --ignored --test-threads=1 --nocapture"]
    fn a_bvh_written_from_pages_with_and_without_copies_reaches_its_readers() {
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const OLD: u32 = 0x1111_1111;
        let host = crate::host::host().expect("no D3D12 device: this test must not pass without a GPU");
        let views = heap().expect("no view heap for title memory");
        let (root, pso) = pipeline(BUILD);
        let (result, result_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let result_gpu = unsafe { result.GetGPUVirtualAddress() };
        // A page below 2^40 from `from` on, write-watched as the title's
        // memory, or plain private memory (`watched` false).
        let page = |from: u64, watched: bool| -> usize {
            let watch = if watched { crate::watch::tests::MEM_WRITE_WATCH } else { 0 };
            (0..256u64)
                .map(|k| from + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | watch, PAGE_READWRITE) } as usize)
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        let (p, q, bare) = (page(0x60_0000_0000, true), page(0x68_0000_0000, true), page(0x70_0000_0000, false));
        let bvh = crate::bvh_gt::mini_bvh();
        let (base, input, input_bare) = (p + 0x100, q + 0x100, bare + 0x100);
        let write = |at: usize, words: &[u32]| {
            for (k, &w) in words.iter().enumerate() {
                // SAFETY: our pages (a guarded page faults into the handler).
                unsafe { (at as *mut u32).add(k).write_volatile(w) };
            }
        };
        // SAFETY: our page.
        let title = || -> Vec<u32> { (0..64).map(|k| unsafe { (base as *const u32).add(k).read_volatile() }).collect() };
        write(base, &[OLD; 64]);
        write(input, &bvh);
        write(input_bare, &bvh);
        crate::shadow::register(p, p as u64, PAGE as u64);
        crate::shadow::register(q, q as u64, PAGE as u64);
        let copies = (crate::shadow::find(p as u64, PAGE as u64).unwrap(), crate::shadow::find(q as u64, PAGE as u64).unwrap());
        let list = host.new_list().unwrap();
        // One submission: the stage reading `from` (and `missing`) if any,
        // a UAV barrier, then the reader if `read`; the list open again
        // after it. What the reader saw.
        let submit = |from: Option<(usize, u64)>, read: bool| -> Vec<u32> {
            let map = bind(&list, OURS).expect("a page table");
            let constants = |mode: u32, q: usize, missing: u64| {
                [base as u32, (base >> 32) as u32, q as u32, (q >> 32) as u32, mode, missing as u32, (missing >> 32) as u32]
            };
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, map.dirty);
                list.list.SetComputeRootUnorderedAccessView(3, result_gpu);
                list.list.SetPipelineState(&pso);
            }
            if let Some((q, missing)) = from {
                let c = constants(0, q, missing);
                // SAFETY: as above.
                unsafe {
                    list.list.SetComputeRoot32BitConstants(4, c.len() as u32, c.as_ptr().cast(), 0);
                    list.list.Dispatch(1, 1, 1);
                }
                crate::list::global_uav_barrier(&list);
            }
            if read {
                let c = constants(1, 0, 0);
                // SAFETY: as above.
                unsafe {
                    list.list.SetComputeRoot32BitConstants(4, c.len() as u32, c.as_ptr().cast(), 0);
                    list.list.Dispatch(1, 1, 1);
                }
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the mapped output, the GPU done with it.
            (0..64).map(|k| unsafe { result_cpu.cast::<u32>().add(k).read_volatile() }).collect()
        };
        let what = |w: &[u32]| -> String {
            let read = |at: u64| w.get(at.checked_sub(0x1000)? as usize / 4).copied();
            if w == bvh {
                format!("the structure (valid {})", crate::bvh_gt::check(&read, 0x1000).valid)
            } else if w.iter().all(|&x| x == OLD) {
                "the older bytes".into()
            } else if w.iter().all(|&x| x == 0) {
                format!("zeros (valid {})", crate::bvh_gt::check(&read, 0x1000).valid)
            } else {
                format!("something else: {:x?}", &w[..16])
            }
        };
        // Missed memory, in a 256 MiB region with no block.
        let elsewhere: u64 = 0xff_f000_0000;
        let mut seen = Vec::new();
        for (name, from, missing) in [
            ("V1 Q with a copy", input, 0),
            ("V3 Q with a copy, a miss elsewhere", input, elsewhere),
            ("V2 Q with no copy", input_bare, 0),
        ] {
            // No miss of earlier submissions withholds this one.
            log_frame();
            log_frame();
            write(base, &[OLD; 64]);
            let i = submit(Some((from, missing)), true);
            let cpu = title();
            let ii = submit(None, true);
            log_frame();
            let iii = submit(None, true);
            println!("{name}: (i) {}; CPU {}; (ii) {}; (iii) {}", what(&i), what(&cpu), what(&ii), what(&iii));
            seen.push((i, cpu, ii, iii));
        }
        let old = vec![OLD; 64];
        let (v1, v3, v2) = (&seen[0], &seen[1], &seen[2]);
        assert!(v1.0 == bvh && v1.1 == bvh && v1.2 == bvh && v1.3 == bvh, "V1: every reader and the CPU see the structure");
        assert!(v3.0 == bvh, "V3: a reader in the same list sees the right structure");
        assert!(v3.1 == old && v3.2 == old && v3.3 == old, "V3: later readers and the CPU see the older bytes (case B)");
        assert!(v2.0.iter().all(|&x| x == 0), "V2: the stage read zeros (case A1)");
        assert!(v2.1 == old && v2.2 == old && v2.3 == old, "V2: later readers and the CPU see the older bytes");
        for at in [p, q] {
            crate::shadow::unregister(at, at as u64);
        }
        drop(copies);
    }

    /// V2 of the test above with the arena (`crate::arena`): Q and P are
    /// private memory with no copy, as the builder's pools. The first
    /// submissions miss (the regions get blocks, then the pages arena pages,
    /// Q filled from title memory, P too); from then on the stage reads the
    /// structure from Q's arena page and writes it into P's, and readers
    /// see it (i) in the same list, (ii) in the next submission, (iii) after
    /// the frame's end, while title memory at P keeps its older bytes
    /// (nothing goes back). G, P's arena page fetched as `crate::bvh_gt`
    /// does, passes the validator; the readers' lists miss nothing.
    ///
    /// `cargo test --release -p victory-gpu --features diagnostics with_the_arena -- --ignored --test-threads=1 --nocapture`
    #[test]
    #[ignore = "needs a D3D12 GPU; run alone with --ignored --test-threads=1 --nocapture"]
    fn with_the_arena_a_bvh_built_from_pages_with_no_copy_reaches_its_readers() {
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const OLD: u32 = 0x1111_1111;
        crate::arena::tests::FORCED.with(|f| f.set(Some(true)));
        let host = crate::host::host().expect("no D3D12 device: this test must not pass without a GPU");
        let views = heap().expect("no view heap for title memory");
        let (root, pso) = pipeline(BUILD);
        let (result, result_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let result_gpu = unsafe { result.GetGPUVirtualAddress() };
        // Write-watched private memory below 2^40 (as the pools), no copy.
        let page = |from: u64| -> usize {
            (0..256u64)
                .map(|k| from + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        let (p, q) = (page(0x78_0000_0000), page(0x7c_0000_0000));
        let bvh = crate::bvh_gt::mini_bvh();
        let (base, input) = (p + 0x100, q + 0x100);
        let write = |at: usize, words: &[u32]| {
            for (k, &w) in words.iter().enumerate() {
                // SAFETY: our pages.
                unsafe { (at as *mut u32).add(k).write_volatile(w) };
            }
        };
        // SAFETY: our page.
        let title = || -> Vec<u32> { (0..64).map(|k| unsafe { (base as *const u32).add(k).read_volatile() }).collect() };
        write(base, &[OLD; 64]);
        write(input, &bvh);
        let list = host.new_list().unwrap();
        // One submission: the stage if `build`, a UAV barrier, the reader;
        // what the reader saw and whether the list missed memory.
        let submit = |build: bool| -> (Vec<u32>, bool) {
            let map = bind(&list, OURS).expect("a page table");
            let constants = |mode: u32| [base as u32, (base >> 32) as u32, input as u32, (input >> 32) as u32, mode, 0, 0];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, map.dirty);
                list.list.SetComputeRootUnorderedAccessView(3, result_gpu);
                list.list.SetPipelineState(&pso);
            }
            for mode in [0, 1].into_iter().filter(|&m| build || m == 1) {
                let c = constants(mode);
                // SAFETY: as above.
                unsafe {
                    list.list.SetComputeRoot32BitConstants(4, c.len() as u32, c.as_ptr().cast(), 0);
                    list.list.Dispatch(1, 1, 1);
                }
                crate::list::global_uav_barrier(&list);
            }
            list.close();
            let misses = || stats(|s| s.missed_pages + s.missed_regions);
            let before = misses();
            host.execute(&[&list]);
            let missed = misses() != before;
            list.reset();
            // SAFETY: the mapped output, the GPU done with it.
            ((0..64).map(|k| unsafe { result_cpu.cast::<u32>().add(k).read_volatile() }).collect(), missed)
        };
        log_frame();
        log_frame();
        let mut rounds = 0;
        let mut i = Vec::new();
        for round in 1..=6 {
            let missed;
            (i, missed) = submit(true);
            println!(
                "arena round {round}: (i) {}, list missed {missed}",
                if i == bvh { "the structure" } else if i.iter().all(|&x| x == 0) { "zeros" } else { "something else" }
            );
            if i == bvh {
                rounds = round;
                break;
            }
        }
        let cpu = title();
        let (ii, ii_missed) = submit(false);
        log_frame();
        let (iii, iii_missed) = submit(false);
        let g = crate::arena::fetch(&[p as u64]);
        let g_words: Vec<u32> = g
            .first()
            .map(|(_, b)| b[0x100..0x200].as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect())
            .unwrap_or_default();
        let read = |at: u64| g_words.get(at.checked_sub(0x1000)? as usize / 4).copied();
        let g_valid = crate::bvh_gt::check(&read, 0x1000).valid;
        let view = crate::arena::view();
        println!(
            "with the arena: structure after {rounds} submissions; CPU {}; (ii) {}; (iii) {}; G valid {g_valid}; arena {} pages, {} regions",
            if cpu == vec![OLD; 64] { "the older bytes" } else { "something else" },
            ii == bvh,
            iii == bvh,
            view.pages.len(),
            view.regions.len()
        );
        crate::arena::tests::FORCED.with(|f| f.set(None));
        assert!(rounds > 0, "the stage never read the structure through the arena");
        assert!(rounds <= 3, "regions, then pages, then backed: at most 3 submissions");
        assert!(view.pages.iter().any(|&(a, _)| a == p as u64) && view.pages.iter().any(|&(a, _)| a == q as u64));
        assert_eq!(i, bvh, "V2 with the arena: (i) the structure");
        assert_eq!(ii, bvh, "(ii) the next submission reads it from the arena");
        assert_eq!(iii, bvh, "(iii) so does the next frame");
        assert_eq!(cpu, vec![OLD; 64], "nothing goes back into title memory");
        assert!(g_valid, "G from the arena passes the validator");
        assert!(!ii_missed && !iii_missed, "readers of arena pages miss nothing");
    }
}
