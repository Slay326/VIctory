//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The copies of big title buffers are in the GPU's own memory ([`cards`]):
//! shaders read and write those at the GPU's speed, and the pages they
//! wrote come down before they are compared (`crate::host::card`).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
    /// With [`cards`], for a big title buffer: `buffer` is in the GPU's own
    /// memory, and this is the rest of the copy.
    card: Option<Card>,
}

/// What a copy in the GPU's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in system memory: what `data` maps. Title memory is
    /// copied into it as into any copy, and what changed in it goes to the
    /// GPU's side before the next submission; what shaders wrote on the
    /// GPU's side comes down into it before it is compared.
    staging: ID3D12Resource,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the GPU's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
}

/// Whether the host copies of big title buffers are in the GPU's own memory
/// (`VICTORY_CARD_BUFFERS=0` keeps every copy in system memory; read when a
/// copy is made).
///
/// Every host copy was a buffer in system memory that the GPU reads and
/// writes over the bus, so that our code can fill and compare it. The
/// title's compute shaders (skinning, hair, its acceleration structure
/// builds) read and write almost all their data in a few big title buffers,
/// and their loads waited for the bus one after the other: on an RTX 3070
/// those dispatches were about 120 ms of a world frame's GPU time.
///
/// With the switch a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the GPU's memory. The CPU's side
/// (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the GPU's side
/// before each submission. After a submission, the pages its shaders
/// marked as written ([`copy_back_marks`]) and the ranges declared as
/// written without marks are copied from the GPU's side to the CPU's
/// before they are compared, so the compares see what the GPU wrote as
/// they do for a copy in system memory.
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the GPU's memory, and the
/// most bytes of such copies if set, both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (1) and `VICTORY_CARD_BUFFERS_MB` (unset:
/// [`CARD_MOST`], and at most a quarter of the host GPU's memory budget).
/// The world of Grand Theft Auto VI has eleven title buffers of 16 MiB or
/// more, 583 MB, which hold what its skinning reads and writes; its hair
/// simulation works in smaller ones.
fn card_limits() -> (u64, Option<u64>) {
    static LIMITS: std::sync::OnceLock<(u64, Option<u64>)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|n| n << 20);
        (
            mb("VICTORY_CARD_BUFFERS_MIN_MB").unwrap_or(1 << 20).max(1 << 20),
            mb("VICTORY_CARD_BUFFERS_MB"),
        )
    })
}

/// The most bytes of copies in the GPU's memory unless
/// `VICTORY_CARD_BUFFERS_MB` says otherwise (the world of Grand Theft Auto
/// VI fills it on an 8 GB card).
const CARD_MOST: u64 = 1 << 30;

/// The most bytes of copies in the GPU's memory with the BVH arena on
/// (`crate::arena`) unless `VICTORY_CARD_BUFFERS_MB` says otherwise: the
/// arena takes the room (copies past it stay in system memory, slower for
/// the GPU, the same bytes).
const CARD_MOST_ARENA: u64 = 512 << 20;

/// The most bytes of copies in the GPU's memory on `host` ([`card_limits`]).
fn card_most(host: &crate::host::Host) -> u64 {
    let most = if crate::arena::enabled() { CARD_MOST_ARENA } else { CARD_MOST };
    card_limits().1.unwrap_or_else(|| most.min(host.card_budget().map_or(u64::MAX, |b| b / 4)))
}

/// The cap on copies in the GPU's memory now, and what it would be without
/// the BVH arena (`crate::arena`'s log line).
pub(crate) fn card_cap() -> (u64, u64) {
    let Some(host) = host() else { return (0, 0) };
    let without = card_limits().1.unwrap_or_else(|| CARD_MOST.min(host.card_budget().map_or(u64::MAX, |b| b / 4)));
    (card_most(host), without)
}

/// Copies refused a GPU side by the cap so far, and their bytes.
static CARD_REFUSED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) fn card_refused() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CARD_REFUSED[0].load(Relaxed), CARD_REFUSED[1].load(Relaxed))
}

/// The smallest title buffer whose copy goes to the GPU's memory
/// ([`card_limits`]; a test sets its own on its thread).
fn card_least() -> u64 {
    #[cfg(test)]
    if let Some(least) = tests::LEAST.with(std::cell::Cell::get) {
        return least;
    }
    card_limits().0
}

/// Bytes alive of copies' GPU sides.
static CARD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The copies in the GPU's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// A buffer of `size` bytes in the GPU's memory for a title buffer's copy,
/// if the copy is to be there: [`cards`] is on, the size is at least
/// `least` and within [`card_most`], and the host has what it takes.
fn card_for(size: u64, least: u64) -> Option<ID3D12Resource> {
    use std::sync::atomic::Ordering::Relaxed;
    if !cards() || size < least {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if CARD_BYTES.fetch_add(size, Relaxed) + size > card_most(host) {
        CARD_BYTES.fetch_sub(size, Relaxed);
        CARD_REFUSED[0].fetch_add(1, Relaxed);
        CARD_REFUSED[1].fetch_add(size, Relaxed);
        return None;
    }
    let buffer = host.card_buffer(size);
    if buffer.is_none() {
        CARD_BYTES.fetch_sub(size, Relaxed);
    }
    buffer
}

/// Copies, each with ranges of it (offsets).
pub(crate) type Ranges = Vec<(Arc<Shadow>, Vec<(usize, usize)>)>;

/// The copies in the GPU's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the GPU's
/// side (`crate::host::card`).
pub(crate) fn card_dirty() -> Ranges {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| std::mem::take(&mut *c.dirty.lock().unwrap_or_else(|e| e.into_inner())))
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// How many copies are in the GPU's memory, and their bytes (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn card_copies() -> (usize, u64) {
    let live = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count();
    (live, CARD_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}

/// Ranges in order, those that overlap or touch as one.
pub(crate) fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Snapshot segments (offset, bytes; sorted, disjoint, possibly touching)
/// and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults),
/// largest first.
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// (the smallest that fits).
fn pooled(n: usize) -> Vec<u8> {
    if n < POOLED_LEAST {
        return Vec::with_capacity(n);
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fits = pool.partition_point(|v| v.capacity() >= n);
    let mut v = match fits {
        0 => Vec::with_capacity(n),
        i => pool.remove(i - 1),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    if v.capacity() < POOLED_LEAST {
        return;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let at = pool.partition_point(|p| p.capacity() >= v.capacity());
    pool.insert(at, v);
    let mut total = 0;
    let keep = pool
        .iter()
        .take_while(|v| {
            total += v.capacity();
            total <= POOL_BYTES
        })
        .count();
    pool.truncate(keep);
}

/// Whether the snapshot kept every byte of `[from, to)` (offsets in the
/// copy), in one segment or in touching ones.
fn covered(segments: &[(usize, Vec<u8>)], from: usize, to: usize) -> bool {
    let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
    let mut next = from;
    while next < to {
        let Some((at, bytes)) = segments.get(i).filter(|(at, _)| *at <= next) else {
            return false;
        };
        next = at + bytes.len();
        i += 1;
    }
    true
}

/// The snapshot's bytes `[from, to)` (offsets in the copy): a slice of the
/// segment holding them, or put together in `scratch` from touching
/// segments. None where a byte of it was not kept ([`covered`]).
fn kept<'a>(segments: &'a [(usize, Vec<u8>)], from: usize, to: usize, scratch: &'a mut Vec<u8>) -> Option<&'a [u8]> {
    let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
    let (at, bytes) = segments.get(i)?;
    if *at > from {
        return None;
    }
    if to <= at + bytes.len() {
        return Some(&bytes[from - at..to - at]);
    }
    scratch.clear();
    let mut next = from;
    while next < to {
        let (at, bytes) = segments.get(i).filter(|(at, _)| *at <= next)?;
        let end = (at + bytes.len()).min(to);
        scratch.extend_from_slice(&bytes[next - at..end - at]);
        next = end;
        i += 1;
    }
    Some(scratch)
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;

/// Snapshot buffers smaller than this come from the allocator and do not
/// go to the pool (small segments would take its big buffers).
const POOLED_LEAST: usize = 64 << 10;

/// The runs `[start, end)` of bytes where the copy differs from title
/// memory (`host`, `title`: the same bytes of each) and `gpu(i)` says the
/// GPU wrote byte `i`: what a copy-back puts into title memory.
///
/// Equal stretches (most of a page) are skipped a block at a time, then a
/// line at a time inside a block that differs: one comparison per 8 bytes
/// cost ~30 us per 64 KiB page, ~70 ms per world frame of marked pages.
fn differing(host: &[u8], title: &[u8], gpu: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    const BLOCK: usize = 1 << 10;
    const LINE: usize = 32;
    let n = host.len().min(title.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut block = 0;
    while block < n {
        let block_end = (block + BLOCK).min(n);
        if host[block..block_end] != title[block..block_end] {
            let mut line = block;
            while line < block_end {
                let line_end = (line + LINE).min(block_end);
                if host[line..line_end] != title[line..line_end] {
                    for i in line..line_end {
                        if host[i] != title[i] && gpu(i) {
                            match runs.last_mut() {
                                Some((_, end)) if *end == i => *end = i + 1,
                                _ => runs.push((i, i + 1)),
                            }
                        }
                    }
                }
                line = line_end;
            }
        }
        block = block_end;
    }
    runs
}

/// [`differing`] of a copy-back with no bytes saved (every differing byte
/// goes back), by whole 32-byte lines: a line with a differing byte is a run.
/// The line's other bytes are title memory's as read, as between the runs
/// [`write_runs`] joins (gaps under [`SPAN_GAP`]). Byte by byte, the
/// world's written pages (many lines differ) cost ~33 us each, ~50 ms per
/// frame (run d16 20261125-225812).
fn differing_rows(host: &[u8], title: &[u8]) -> Vec<(usize, usize)> {
    const BLOCK: usize = 1 << 10;
    const LINE: usize = 32;
    let n = host.len().min(title.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut block = 0;
    while block < n {
        let block_end = (block + BLOCK).min(n);
        if host[block..block_end] != title[block..block_end] {
            let mut line = block;
            while line < block_end {
                let line_end = (line + LINE).min(block_end);
                if host[line..line_end] != title[line..line_end] {
                    match runs.last_mut() {
                        Some((_, end)) if *end == line => *end = line_end,
                        _ => runs.push((line, line_end)),
                    }
                }
                line = line_end;
            }
        }
        block = block_end;
    }
    runs
}

/// Runs of a copy-back closer than this go into title memory as one span,
/// with title memory's bytes between them as the copy-back read them: a
/// kernel copy per run cost over 1 ms per page in the world, where shaders
/// change many scattered words of a page.
const SPAN_GAP: usize = 256;

/// Writes the runs `[start, end)` of `merged` (title memory as read, with
/// the shaders' bytes in the runs) into title memory at `at`, and nothing
/// far from them: the title's threads write the same pages while a
/// copy-back runs, and a byte they wrote after the copy-back read the page
/// must stay. (Putting back the whole page undid those writes.) True if a
/// run was written.
fn write_runs(at: usize, merged: &[u8], runs: &[(usize, usize)]) -> bool {
    write_runs_by(at, merged, runs, None) != Put::Nothing
}

/// What a copy-back's write did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Put {
    Nothing,
    /// Written at the title's address, its guard opened and the page
    /// marked changed (the caller takes it).
    Title,
    /// Written through our own view of the section (`crate::watch`'s
    /// `write_beside`): the title's guard stayed, the page is taken unless
    /// the title wrote it meanwhile.
    Beside,
}

/// [`write_runs`]; with a tracker (a marked page's copy-back), through our
/// own view of the memory's section where there is one.
fn write_runs_by(at: usize, merged: &[u8], runs: &[(usize, usize)], tracked: Option<&crate::watch::Tracked>) -> Put {
    if runs.is_empty() {
        return Put::Nothing;
    }
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for &(start, end) in runs {
        match spans.last_mut() {
            Some((_, last)) if start - *last < SPAN_GAP => *last = end,
            _ => spans.push((start, end)),
        }
    }
    // True if every span landed.
    let put = |delta: usize| {
        let mut all = true;
        for &(start, end) in &spans {
            all &= crate::host::write_mapped(
                (at + start).wrapping_add(delta) as *mut u8,
                merged[start..].as_ptr(),
                end - start,
            );
        }
        all
    };
    if let Some(t) = tracked {
        if t.write_beside(at, merged.len(), put).is_some() {
            return Put::Beside;
        }
        #[cfg(feature = "diagnostics")]
        if crate::watch::own_view_on() {
            crate::copystat::add(crate::copystat::C::BesideNone, 1);
        }
    }
    let _writing = crate::watch::writing(at, merged.len());
    let _ = put(0);
    Put::Title
}

/// Outside `runs`, the copy takes title memory's bytes (`title`, read at the
/// copy-back). True if that changed the copy (then a copy in the GPU's
/// memory sends them up).
fn take_others(host: &mut [u8], title: &[u8], runs: &[(usize, usize)]) -> bool {
    let mut changed = false;
    let mut take = |host: &mut [u8], title: &[u8]| {
        if host != title {
            host.copy_from_slice(title);
            changed = true;
        }
    };
    let mut at = 0;
    for &(start, end) in runs {
        take(&mut host[at..start], &title[at..start]);
        at = end;
    }
    take(&mut host[at..], &title[at..]);
    changed
}

thread_local! {
    /// Title memory as a copy-back reads it, a page at most: a fresh zeroed
    /// buffer cost a page's memset per page, thousands per world frame.
    static TITLE: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0; PAGE]);
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` as its CPU side
    /// holds them (of a copy in the GPU's memory, what went up or came down
    /// last), or, with `fetch`, as shaders left them (fetched from the GPU's
    /// side first: only while no submission is on the GPU).
    pub(crate) fn host_dwords(&self, address: u64, n: usize, fetch: bool) -> Option<Vec<u32>> {
        let at = address.checked_sub(self.start)? as usize;
        if at + 4 * n > self.len as usize {
            return None;
        }
        if fetch {
            crate::host::card_fetch(&[(self, at, at + 4 * n)]);
        }
        // SAFETY: inside the mapping (checked).
        Some(unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() })
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        crate::host::card_fetch(&[(self, at, at + 4 * n)]);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
        self.card_changed(at, at + bytes.len());
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// The copy's bytes of the 64 KiB page at `page` (cut to the copy) as
    /// shaders left them, and the title address they start at
    /// (`crate::bvh_gt`, after a submission's fence). Of a copy in the
    /// GPU's memory the page is fetched from there and the CPU side is left
    /// as it was.
    pub(crate) fn gpu_bytes(&self, page: usize) -> Option<(usize, Vec<u8>)> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return None;
        }
        let at = from - start;
        // SAFETY: inside the mapping; only the submitting thread writes it,
        // and the GPU is done with the submission.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), to - from) };
        if self.card.is_none() {
            return Some((from, host.to_vec()));
        }
        let before = host.to_vec();
        crate::host::card_fetch(&[(self, at, at + (to - from))]);
        // SAFETY: as above, after the fetch.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), to - from) };
        let fetched = host.to_vec();
        host.copy_from_slice(&before);
        Some((from, fetched))
    }

    /// Whether the title wrote the page at `page` since the copy took it
    /// (or it is to be taken again); None for memory not tracked.
    pub(crate) fn title_changed(&self, page: usize) -> Option<bool> {
        self.tracked.as_ref().map(|t| t.changed(page / PAGE))
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        // A title write from here on saves the page's bytes from before it:
        // armed before the check, and before a copy in the GPU's memory
        // fetches the page (a round trip to the GPU, while the title runs).
        let mut saved = None;
        if let Some(t) = &self.tracked {
            t.arm(from, to);
            saved = t.saved(page / PAGE);
            if saved.is_none() && t.changed(page / PAGE) {
                t.unarm(page / PAGE);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        page = format_args!("{page:#x}"),
                        "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                    )
                });
                return 0;
            }
        }
        // A copy in the GPU's memory: the page comes down first.
        if self.card.is_some() {
            crate::host::card_fetch(&[(self, from - start, to - start)]);
        }
        if self.put_back(page, from, to, saved, false) == Put::Nothing {
            return 0;
        }
        // Our write is not the title's: the copy takes the page as it is.
        if let Some(t) = &self.tracked {
            self.take_and_read(t, page, from, to);
        }
        to - from
    }

    /// After a copy-back: the page at `page` is taken, then the copy's bytes
    /// `[from, to)` (title addresses) are read from title memory again. The
    /// title writes the page while it goes back, so bytes read before the
    /// guard is back may be old: a title write that lands before the guard is
    /// in this read, and one after it faults and marks the page changed.
    /// (Taking the page with the bytes read earlier lost such writes from the
    /// copy, and a later copy-back of the page put the copy's older bytes
    /// over them.) Of a copy in the GPU's memory, bytes that came from the
    /// title this way go up to the GPU's side.
    fn take_and_read(&self, t: &crate::watch::Tracked, page: usize, from: usize, to: usize) {
        t.taken(&[page / PAGE]);
        let at = from - self.start as usize;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), to - from) };
        if self.card.is_none() {
            crate::host::read_into(from as *const u8, host);
            return;
        }
        // The reused read buffer, as in `put_back` (done with it here).
        let changed = TITLE.with(|title| {
            let title = &mut title.borrow_mut()[..to - from];
            let differs = crate::host::read_into(from as *const u8, title) && *host != *title;
            if differs {
                host.copy_from_slice(title);
            }
            differs
        });
        if changed {
            self.card_changed(at, at + (to - from));
        }
    }

    /// The copy-back of `[from, to)` in the page at `page`: the bytes the
    /// shaders changed go into title memory, and no others (the title's
    /// threads write the page meanwhile: a whole page put back undid their
    /// writes since it was read), and the copy takes the title's others.
    /// What the shaders changed differs from the copy's bytes before the
    /// submission: `saved`, or what the page held when a title write landed
    /// while it was read (saved by the guard while the page is armed), else
    /// title memory. True if bytes were written.
    ///
    /// With `beside`, through our own view of the memory where there is one
    /// ([`Put::Beside`]).
    fn put_back(&self, page: usize, from: usize, to: usize, saved: Option<Box<[u8]>>, beside: bool) -> Put {
        TITLE.with(|title| self.put_back_with(page, from, to, saved, beside, &mut title.borrow_mut()))
    }

    fn put_back_with(
        &self,
        page: usize,
        from: usize,
        to: usize,
        saved: Option<Box<[u8]>>,
        beside: bool,
        title: &mut [u8],
    ) -> Put {
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host =
            unsafe { std::slice::from_raw_parts_mut(self.data.add(from - self.start as usize), n) };
        let title = &mut title[..n];
        let read = crate::host::read_into(from as *const u8, title);
        #[cfg(test)]
        tests::after_read();
        // Our own write below must not save the page for this copy again.
        let saved = match &self.tracked {
            Some(t) => {
                t.unarm(page / PAGE);
                saved.or_else(|| t.saved(page / PAGE))
            }
            None => saved,
        };
        if !read {
            return Put::Nothing;
        }
        let runs = match &saved {
            None => differing_rows(host, title),
            Some(before) => {
                let before = &before[from - page..to - page];
                differing(host, title, |i| host[i] != before[i])
            }
        };
        // The copy takes the title's other bytes, and holds what goes back.
        // Without bytes saved, the runs hold every byte that differs (by
        // lines: `differing_rows`): the others are the title's already (copying them cost a 64 KiB write
        // per page).
        if saved.is_some() && take_others(host, title, &runs) {
            // Not on the GPU's side of a copy there.
            let at = from - self.start as usize;
            self.card_changed(at, at + n);
        }
        write_runs_by(from, host, &runs, if beside { self.tracked.as_ref() } else { None })
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        Shadow::with_card(buffer, data, gpu, start, len, slice, None)
    }

    /// [`Shadow::new_at`] of a copy whose `buffer` is in the GPU's memory
    /// where `card` is given (`data` maps its CPU side then).
    fn with_card(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        slice: Option<Slice>,
        card: Option<Card>,
    ) -> Arc<Shadow> {
        let on_card = card.is_some();
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            base: slice.map_or(0, |s| s.offset),
            slice,
            card,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        if on_card {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        shadow
    }

    /// A copy of the title buffer at `start` (`len` bytes, `size` with the
    /// rounding): in the GPU's memory if it is to be there ([`card_for`];
    /// title buffers of `least` bytes or more), else in system memory.
    fn made(start: u64, len: u64, size: u64, least: u64) -> Option<Arc<Shadow>> {
        let (buffer, data) = host()?.shared_buffer(size)?;
        let (buffer, card) = match card_for(size, least) {
            Some(card) => (card, Some(Card { staging: buffer, dirty: Mutex::default() })),
            None => (buffer, None),
        };
        let on_card = card.is_some();
        crate::host::name(&buffer, || {
            format!("title buffer {start:#x}+{len:#x}{}", if on_card { " (GPU memory)" } else { "" })
        });
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some(Shadow::with_card(buffer, data, gpu, start, len, None, card))
    }

    /// Whether the copy is in the GPU's memory ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU's side of a copy in the GPU's memory.
    pub(crate) fn card_staging(&self) -> Option<&ID3D12Resource> {
        self.card.as_ref().map(|c| &c.staging)
    }

    /// Bytes `[from, to)` (offsets) changed on the CPU's side of a copy in
    /// the GPU's memory: they go up before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        if let Some(card) = &self.card
            && from < to
        {
            card.dirty.lock().unwrap_or_else(|e| e.into_inner()).push((from, to));
        }
    }

    /// `[address, address + len)` of title memory as offsets of the copy,
    /// cut to it.
    pub(crate) fn offsets(&self, address: usize, len: usize) -> (usize, usize) {
        self.inside(address, len)
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // Only the parts no segment holds yet are taken from the copy, each
        // as a segment of its own; what segments hold stays as it was taken
        // (merging them into one copied the kept bytes again at each grown
        // range: the world declares a 64 MiB copy piece by piece, 215 MB of
        // snapshots per frame for it, run 20261126-025033).
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
        let mut gaps = Vec::new();
        let mut at = from;
        for (taken, bytes) in &segments[first..] {
            if *taken >= to {
                break;
            }
            if at < *taken {
                gaps.push((at, *taken));
            }
            at = at.max(taken + bytes.len());
        }
        if at < to {
            gaps.push((at, to));
        }
        if gaps.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        let mut index = first;
        for (a, b) in gaps {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::SnapshotBytes, (b - a) as u64);
                crate::copystat::snapshot(self.start, self.len, (b - a) as u64);
            }
            let mut bytes = pooled(b - a);
            bytes.extend_from_slice(host(a, b));
            index += segments[index..].partition_point(|(taken, _)| *taken < a);
            segments.insert(index, (a, bytes));
        }
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let mut scratch = Vec::new();
        if back
            && from < to
            && covered(segments, from, to)
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                let Some(old) = kept(segments, at, at + n, &mut scratch) else {
                    at += n;
                    continue;
                };
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // The GPU's bytes go into title memory, and no others:
                    // the title writes the piece meanwhile.
                    let runs = differing(host, title, |i| host[i] != old[i]);
                    // The copy takes the title's bytes in turn (not on the
                    // GPU's side of a copy there).
                    if take_others(host, title, &runs) {
                        self.card_changed(at, at + n);
                    }
                    if write_runs(start + at, host, &runs) {
                        written += n;
                    }
                    synced.push((start + at, n));
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                for (address, n) in synced {
                    if (first..last).contains(&(address / PAGE)) {
                        self.take_and_read(t, address, address, address + n);
                    }
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        let saved = t.saved(page / PAGE);
        if saved.is_none() && t.changed(page / PAGE) {
            t.unarm(page / PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        // A title write while the page is read saves its bytes from before
        // (armed before a copy in the GPU's memory fetched the page:
        // `copy_back_marks`).
        t.arm(from, to);
        let put = self.put_back(page, from, to, saved, true);
        // Equal to title memory now: taken, unless it holds the page in part.
        // A page neither we nor the title wrote meanwhile (a title write
        // marks it changed) is still taken and guarded, and the copy has
        // title memory's bytes: nothing to read again. (A third of the
        // world's marked pages hold no changed byte.) So is a page written
        // through our own view that the title did not write meanwhile: its
        // guard stayed, and the copy holds what went back and, around it,
        // title memory as read (unchanged since, or the guard had faulted).
        let retaken = t.changed(page / PAGE);
        #[cfg(feature = "diagnostics")]
        if put == Put::Beside && retaken {
            crate::copystat::add(crate::copystat::C::BesideChanged, 1);
        }
        if from == page && to == page + PAGE && (put == Put::Title || retaken) {
            self.take_and_read(t, page, from, to);
        }
        if put != Put::Nothing { to - from } else { 0 }
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        let mut sync_before = |a: usize, b: usize| {
            let Some(before) = &mut before else { return };
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            sync_before(a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            self.card_changed(a - start, b - start);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += self.copy(a, b);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        self.card_changed(from - start, to - start);
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.card.is_some() {
            // SAFETY: a live buffer.
            let size = unsafe { self.buffer.GetDesc() }.Width;
            CARD_BYTES.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        }
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// With copies in the GPU's memory ([`cards`]): the bitmap shaders
    /// read and write is there too (every mark waited for the bus), `gpu`
    /// is its address, and `data` is the CPU's bitmap, which takes its bits
    /// after each submission ([`marks_fetched`]). `VICTORY_CARD_MARKS=0`
    /// keeps the bitmaps in system memory.
    card: Option<CardMarks>,
}

/// A bitmap of written pages and read slots in the GPU's memory.
struct CardMarks {
    /// What shaders read and write.
    marks: ID3D12Resource,
    /// Zeros of its size, to clear its written pages from.
    zeros: ID3D12Resource,
    /// Where it is fetched to after a submission: system memory, mapped.
    fetched: ID3D12Resource,
    fetched_data: usize,
}

/// What a submission does for one bitmap in the GPU's memory
/// ([`card_marks`], `crate::host::card`).
pub(crate) struct CardMarksWork {
    /// What shaders read and write, zeros of its size, where it is fetched
    /// to, and the CPU's bitmap (system memory).
    pub(crate) marks: ID3D12Resource,
    pub(crate) zeros: ID3D12Resource,
    pub(crate) fetched: ID3D12Resource,
    pub(crate) cpu: ID3D12Resource,
    /// The bytes copies' written-page bits lie in: cleared before the
    /// submission, fetched after it.
    pub(crate) written: u64,
    /// The read marks (offset, bytes): fetched after the submission.
    pub(crate) reads: (u64, u64),
    /// The CPU cleared read marks since they last went up: its read marks
    /// go to the GPU's side before the submission.
    pub(crate) reads_up: bool,
    memory: bool,
}

/// Whether the CPU cleared read marks that the bitmaps in the GPU's memory
/// still hold ([`take_reads`], [`reads_since`]).
static READS_CLEARED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// With the bitmaps in the GPU's memory, before a submission: what it does
/// for each (none: nothing to clear or fetch).
pub(crate) fn card_marks() -> Vec<CardMarksWork> {
    let written = u64::from(marks().end.div_ceil(32)) * 4;
    let reads = (u64::from(victory_shader::emit::VIEW_READS / 8), u64::from(victory_shader::emit::HEAP_VIEWS / 8));
    let work: Vec<CardMarksWork> = [false, true]
        .into_iter()
        .filter_map(|memory| {
            let b = bitmap(memory)?;
            let card = b.card.as_ref()?;
            Some(CardMarksWork {
                marks: card.marks.clone(),
                zeros: card.zeros.clone(),
                fetched: card.fetched.clone(),
                cpu: b.buffer.clone(),
                written,
                reads,
                reads_up: false,
                memory,
            })
        })
        .collect();
    let up = !work.is_empty() && READS_CLEARED.swap(false, std::sync::atomic::Ordering::Relaxed);
    work.into_iter().map(|w| CardMarksWork { reads_up: up, ..w }).collect()
}

/// After a submission, with the bitmaps in the GPU's memory: they were
/// fetched ([`card_marks`]); their bits go into the CPU's bitmaps, which
/// [`copy_back_marks`], [`take_reads`] and [`reads_since`] read.
pub(crate) fn marks_fetched(work: &[CardMarksWork]) {
    for w in work {
        let Some((b, card)) = bitmap(w.memory).and_then(|b| Some((b, b.card.as_ref()?))) else {
            continue;
        };
        for (at, bytes) in [(0, w.written), w.reads] {
            let (first, words) = ((at / 4) as usize, (bytes / 4) as usize);
            // SAFETY: both mappings hold the whole bitmap, which the ranges
            // are within; the GPU is done with them (one submission runs at
            // a time).
            let (bits, fetched) = unsafe {
                (
                    std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words),
                    std::slice::from_raw_parts((card.fetched_data as *const u32).add(first), words),
                )
            };
            for (bit, &f) in bits.iter_mut().zip(fetched) {
                *bit |= f;
            }
        }
    }
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::HEAP_VIEWS;
            let bytes = u64::from(bits / 8);
            let host = host()?;
            let (buffer, data) = host.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let on_card = cards() && std::env::var_os("VICTORY_CARD_MARKS").is_none_or(|v| v != "0") && host.card_ready();
            let card = on_card
                .then(|| {
                    let (fetched, fetched_data) = host.shared_buffer(bytes)?;
                    // SAFETY: our mapped buffer of `bytes` bytes.
                    unsafe { std::ptr::write_bytes(fetched_data, 0, bytes as usize) };
                    let (marks, zeros) = (host.card_buffer(bytes)?, host.card_buffer(bytes)?);
                    crate::host::name(&marks, || "views' written pages and read slots (GPU memory)".into());
                    Some(CardMarks { marks, zeros, fetched, fetched_data: fetched_data as usize })
                })
                .flatten();
            // SAFETY: a live buffer.
            let gpu = unsafe { card.as_ref().map_or(&buffer, |c| &c.marks).GetGPUVirtualAddress() };
            Some(Bitmap {
                buffer,
                data: data as usize,
                gpu,
                card,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let mut maps: Vec<&mut [u32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    if !forgotten.is_empty() {
        READS_CLEARED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    if !slots.is_empty() {
        READS_CLEARED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; the bitmaps are cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        // Marks of offsets past the bitmap land in its last bit, no copy's.
        bits[words - 1] = 0;
        let used = (marks().end as usize).div_ceil(32).min(words);
        for (w, word) in bits[..used]
            .iter_mut()
            .enumerate()
            .filter(|(_, word)| **word != 0)
        {
            let mut x = std::mem::take(word);
            let into = if memory && memory_withheld { &mut held } else { &mut marked };
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    // Of copies in the GPU's memory, the marked pages come down to the
    // CPU's sides first (one fetch for all of them). They are armed before:
    // the fetch is a round trip to the GPU while the title runs, and a title
    // write meanwhile must save the page's bytes from before it
    // (`Shadow::copy_back_marked`).
    let fetch: Vec<(&Shadow, usize, usize)> = pages
        .iter()
        .filter(|(copy, _)| copy.carded())
        .map(|(copy, page)| {
            if let Some(t) = &copy.tracked {
                t.arm(*page, *page + PAGE);
            }
            let (from, to) = copy.inside(*page, PAGE);
            (&**copy, from, to)
        })
        .collect();
    crate::host::card_fetch(&fetch);
    drop(fetch);
    let pages: Vec<(Arc<Shadow>, usize)> = pages
        .into_iter()
        .filter(|(copy, page)| {
            !withheld
                .iter()
                .any(|(c, a, n)| Arc::ptr_eq(c, copy) && *a < page + PAGE && *page < a + n)
        })
        .collect();
    let back = |(copy, page): &(Arc<Shadow>, usize)| {
        let written = copy.copy_back_marked(*page);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    };
    let threads = back_threads();
    if threads < 2 || pages.len() < PARALLEL_LEAST {
        pages.iter().for_each(back);
        return;
    }
    // Pages on their own go back on several threads at once (each is read,
    // compared and written apart from the others: ~25 us a page, ~70 ms of
    // a world frame on one thread, run 20261125-231403). A page two of the
    // marked copies hold, or one whose memory another marked page shows too
    // (`crate::watch::title_memory_aliased`), goes back after them, in the
    // order it had: a copy-back there is seen by the other copy's.
    let alone = pages_alone(&pages.iter().map(|(_, page)| page / PAGE).collect::<Vec<_>>());
    let (apart, together): (Vec<_>, Vec<_>) = pages.iter().zip(&alone).partition(|&(_, &alone)| alone);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        loop {
            // A few pages at a time: neighbours stay on one thread.
            let i = next.fetch_add(PARALLEL_TAKE, std::sync::atomic::Ordering::Relaxed);
            if i >= apart.len() {
                break;
            }
            apart[i..(i + PARALLEL_TAKE).min(apart.len())].iter().for_each(|(page, _)| back(page));
        }
    };
    if apart.len() >= PARALLEL_LEAST {
        std::thread::scope(|scope| {
            for _ in 1..threads {
                scope.spawn(work);
            }
            work();
        });
    } else {
        work();
    }
    together.iter().for_each(|(page, _)| back(page));
}

/// Fewer marked pages than this go back on the submitting thread alone.
const PARALLEL_LEAST: usize = 32;

/// The pages a copy-back thread takes at a time.
const PARALLEL_TAKE: usize = 8;

/// The threads marked pages go back on (`VICTORY_BACK_THREADS`, 1 to 8;
/// 1, the default, is the submitting thread alone).
fn back_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        let threads = std::env::var("VICTORY_BACK_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        if threads > 1 {
            tracing::info!(threads, "marked pages go back on several threads (VICTORY_BACK_THREADS)");
        }
        threads
    })
}

/// For each of `pages` (page numbers): whether no other of them is the same
/// page or shows the same memory at another address.
fn pages_alone(pages: &[usize]) -> Vec<bool> {
    let mut count: std::collections::HashMap<usize, u32> = std::collections::HashMap::with_capacity(pages.len());
    for &page in pages {
        *count.entry(page).or_default() += 1;
    }
    let elsewhere = crate::watch::shown_elsewhere(pages);
    pages
        .iter()
        .zip(elsewhere)
        .map(|(page, elsewhere)| count[page] == 1 && !elsewhere.iter().any(|other| count.contains_key(other)))
        .collect()
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let shadow = Shadow::made(key.0, len, len.next_multiple_of(1 << 16), card_least())?;
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(shadow);
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice));
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Invariants of the copies (CPU and GPU writes, interleaved submissions),
/// tested without game data.
#[cfg(test)]
#[path = "shadow_invariants.rs"]
mod invariants;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Marked pages go back on several threads only where each is on its
    /// own: not a page two marked copies hold, nor two showing one memory.
    #[test]
    fn pages_two_copies_hold_or_shown_elsewhere_go_back_in_order() {
        // Page numbers no test's memory has (addresses past user space).
        let base = 0x7fff_0000_0000usize;
        let (a, b, c, d, e) = (base, base + 1, base + 2, base + 3, base + 4);
        crate::watch::title_memory_aliased(d * PAGE, &[e * PAGE]);
        assert_eq!(pages_alone(&[a, b, a, c, d]), [false, true, false, true, true]);
        assert_eq!(pages_alone(&[e, c]), [true, true]);
        assert_eq!(pages_alone(&[d, c, e]), [false, true, false]);
        crate::watch::title_memory_remapped(d * PAGE, PAGE);
        assert_eq!(pages_alone(&[d, e]), [true, true]);
    }

    thread_local! {
        /// [`card_least`] for the test on this thread.
        pub(super) static LEAST: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
        /// Title writes (address, 8 bytes) [`during_fetch`] makes, last first.
        static DURING_FETCH: std::cell::RefCell<Vec<(usize, u64)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A title thread's write landing while a copy in the GPU's memory is
    /// fetched to its CPU side (tests): the next of [`DURING_FETCH`]'s.
    pub(crate) fn during_fetch() {
        if let Some((at, value)) = DURING_FETCH.with(|w| w.borrow_mut().pop()) {
            // SAFETY: 8 bytes of the test's title memory (guarded or not: a
            // guard's fault opens it).
            unsafe { (at as *mut u64).write_volatile(value) };
        }
    }

    /// A BVH builder's node (fp16 boxes; an empty one is +inf..-inf,
    /// `0xfc007c007c007c00`) next to a pointer the title keeps in the same
    /// page of a copy in the GPU's memory. The slot held an empty box from
    /// an earlier build; the title writes its pointer there while the
    /// copy-back fetches the page from the GPU (a round trip). The pointer
    /// stays and the node goes back, for a page shaders wrote at computed
    /// addresses and a page they marked alike, and the GPU's side gets the
    /// title's pointer. (Arming the page after the fetch, or putting back
    /// the whole page, wrote the empty box over the pointer: the title then
    /// read it as an address and faulted with rcx = 0xfc007c007c007c00.)
    #[test]
    fn a_title_pointer_next_to_builder_nodes_survives_a_card_copy_back() {
        const EMPTY: u64 = 0xfc00_7c00_7c00_7c00;
        const NODE: u64 = 0x3c00_3c00_bc00_bc00;
        const POINTER: u64 = 0x0000_0234_5678_9ab0;
        let (Some(host), Some(b)) = (host(), bitmap(false)) else {
            return; // no host GPU here
        };
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        // SAFETY: (all title reads and writes below) bytes of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const u64).read_volatile() };
        for page in [0, PAGE] {
            // SAFETY: as above.
            unsafe { ((p + page + 0x100) as *mut u64).write_volatile(EMPTY) };
        }
        register(p, p as u64, len as u64);
        LEAST.with(|l| l.set(Some(0)));
        let copy = find(p as u64, len as u64);
        LEAST.with(|l| l.set(None));
        let copy = copy.expect("a copy");
        if !copy.carded() {
            assert!(!cards(), "the copy is in GPU memory unless that is switched off");
            unregister(p, p as u64);
            return;
        }
        // The builder writes a node into the GPU's side of pages 0 and 1.
        let (upload, data) = host.upload_buffer(256).unwrap();
        // SAFETY: the buffer's mapping.
        unsafe { data.cast::<u64>().write_unaligned(NODE) };
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        for page in [0, PAGE] {
            // SAFETY: an open list; the bytes lie in both buffers.
            unsafe { list.list.CopyBufferRegion(&copy.buffer, (page + 0x200) as u64, &upload, 0, 8) };
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!((title(0x200), title(PAGE + 0x200)), (0, 0), "nothing declared: nothing back yet");
        // Page 1: written at addresses shaders computed.
        DURING_FETCH.with(|w| *w.borrow_mut() = vec![(p + PAGE + 0x100, POINTER)]);
        assert_eq!(copy.copy_back_page(p + PAGE), PAGE);
        assert_eq!(title(PAGE + 0x100), POINTER, "the title's pointer stays, page 1: {:#x}", title(PAGE + 0x100));
        assert_eq!(title(PAGE + 0x200), NODE, "the node goes back, page 1");
        // Page 0: marked through a view no list declared.
        DURING_FETCH.with(|w| *w.borrow_mut() = vec![(p + 0x100, POINTER)]);
        let bit = copy.marks.expect("tracked memory gets mark bits") as usize;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(b.data as *mut u32).add(bit / 32) |= 1 << (bit % 32) };
        copy_back_marks(&[], false);
        assert_eq!(title(0x100), POINTER, "the title's pointer stays, page 0: {:#x}", title(0x100));
        assert_eq!(title(0x200), NODE, "the node goes back, page 0");
        // The title's bytes the copy took go up with the next submission.
        list.reset();
        list.close();
        host.execute(&[&list]);
        let gpu_side = |at: usize| {
            let w = copy.read_host((p + at) as u64, 2);
            u64::from(w[0]) | u64::from(w[1]) << 32
        };
        assert_eq!((gpu_side(0x100), gpu_side(PAGE + 0x100)), (POINTER, POINTER), "the GPU's side has the pointers");
        assert_eq!((gpu_side(0x200), gpu_side(PAGE + 0x200)), (NODE, NODE));
        drop(list);
        unregister(p, p as u64);
    }

    /// Copies in the GPU's memory take at most 1 GiB and a quarter of the
    /// host GPU's memory budget, unless `VICTORY_CARD_BUFFERS_MB` is set.
    #[test]
    fn copies_in_gpu_memory_keep_within_the_budget() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if card_limits().1.is_some() {
            return; // set by hand
        }
        let budget = host.card_budget();
        assert!(card_most(host) <= CARD_MOST);
        if let Some(budget) = budget {
            assert!(budget > 0 && card_most(host) <= budget / 4);
        }
    }

    /// A copy in the GPU's memory: title memory reaches the GPU's side;
    /// what shaders write in a range declared without marks goes back into
    /// title memory (those bytes only, and nothing written outside it);
    /// what they write in a page they mark goes back; what the title writes
    /// later goes up.
    #[test]
    fn writes_to_a_copy_in_gpu_memory_go_back() {
        let (Some(host), Some(b)) = (host(), bitmap(false)) else {
            return; // no host GPU here
        };
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        LEAST.with(|l| l.set(Some(0)));
        let copy = find(p as u64, len as u64);
        LEAST.with(|l| l.set(None));
        let copy = copy.expect("a copy");
        if !copy.carded() {
            assert!(!cards(), "the copy is in GPU memory unless that is switched off");
            unregister(p, p as u64);
            return;
        }
        let pattern = |at: usize| (at * 7 + 1) as u8;
        // SAFETY: (all title reads and writes below) bytes of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        for at in 0..len {
            // SAFETY: as above.
            unsafe { ((p + at) as *mut u8).write_volatile(pattern(at)) };
        }
        // What the shaders write: bytes of an upload buffer.
        let (upload, data) = host.upload_buffer(256).unwrap();
        // SAFETY: the buffer's mapping.
        unsafe { data.write_bytes(0xaa, 256) };
        let list = host.new_list().unwrap();
        let shader_writes = |at: usize, n: u64| {
            // SAFETY: an open list; the bytes lie in both buffers.
            unsafe { list.list.CopyBufferRegion(&copy.buffer, at as u64, &upload, 0, n) };
        };
        // The GPU's side, fetched to the CPU's and read there.
        let gpu_side = |at: usize| {
            crate::host::card_fetch(&[(&*copy, at & !3, (at & !3) + 4)]);
            copy.read_host((p + (at & !3)) as u64, 1)[0].to_le_bytes()[at % 4]
        };
        // A range declared without marks: the snapshot's compare.
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, (p + PAGE + 100) as u64, 8192, false);
        shader_writes(PAGE + 200, 3);
        shader_writes(PAGE + 4096 + 50, 2);
        // Outside the declared range: stays on the GPU's side.
        shader_writes(2 * PAGE + 10, 1);
        list.close();
        host.execute(&[&list]);
        for at in [PAGE + 200, PAGE + 201, PAGE + 202, PAGE + 4096 + 50, PAGE + 4096 + 51] {
            assert_eq!(title(at), 0xaa, "the shader's write at {at:#x} goes back");
        }
        for at in [0, PAGE + 99, PAGE + 199, PAGE + 203, PAGE + 4096 + 52, 2 * PAGE + 10, len - 1] {
            assert_eq!(title(at), pattern(at), "nothing else does ({at:#x})");
        }
        assert_eq!((gpu_side(5), gpu_side(PAGE + 201)), (pattern(5), 0xaa), "title bytes went up");
        assert_eq!(gpu_side(2 * PAGE + 10), 0xaa);
        // A page the shaders mark.
        list.reset();
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, PAGE as u64, true);
        shader_writes(300, 4);
        // The mark, set as the submission is prepared (under its lock: the
        // bitmap is every submission's).
        let bit = copy.marks.expect("tracked memory gets mark bits") as usize;
        let bits = b.data;
        // SAFETY: a dword of the mapped bitmap.
        list.defer_host(move || unsafe { *(bits as *mut u32).add(bit / 32) |= 1 << (bit % 32) });
        list.close();
        host.execute(&[&list]);
        assert_eq!((title(299), title(300), title(303), title(304)), (pattern(299), 0xaa, 0xaa, pattern(304)));
        // The title writes: the byte goes up with the next submission.
        // SAFETY: a byte of our pool.
        unsafe { ((p + 7) as *mut u8).write_volatile(0x33) };
        list.reset();
        copy.refresh(&list, p as u64, PAGE as u64);
        list.close();
        host.execute(&[&list]);
        assert_eq!((gpu_side(6), gpu_side(7), gpu_side(8)), (pattern(6), 0x33, pattern(8)));
        drop(list);
        unregister(p, p as u64);
    }


    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "only the parts not kept yet");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// A range grown over parts kept before takes only the bytes not kept
    /// yet (no byte twice, however often ranges grow), keeps the earlier
    /// parts as they were taken, and goes back across the touching
    /// segments as one range would.
    #[test]
    fn growing_ranges_keep_each_byte_once_and_go_back_across_segments() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        let segments = |s: &Shadow| s.before.lock().unwrap().0.len();
        copy.writing(p + 100, 100, false);
        copy.writing(p + 300, 100, false);
        copy.writing(p, PAGE + 50, false);
        copy.writing(p + 50, 2 * PAGE - 50, false);
        assert_eq!(bytes(&copy), 2 * PAGE, "each byte once");
        assert_eq!(segments(&copy), 6, "the parts taken, apart");
        // The GPU writes in several segments of each page.
        copy.write_host((p + 10) as u64, &[1]);
        copy.write_host((p + 150) as u64, &[2]);
        copy.write_host((p + 250) as u64, &[3]);
        copy.write_host((p + PAGE - 1) as u64, &[4]);
        copy.write_host((p + PAGE + 100) as u64, &[5]);
        assert_eq!(copy.copy_back(p, PAGE + 50), PAGE, "the first page written");
        assert_eq!(copy.copy_back(p + 100, 100), 0, "compared once");
        assert_eq!(copy.copy_back(p + 300, 100), 0);
        // The second page's piece starts in a part of the first range.
        assert_eq!(copy.copy_back(p + 50, 2 * PAGE - 50), PAGE, "the second page written");
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!(
            [title(p + 10), title(p + 150), title(p + 250), title(p + PAGE - 1), title(p + PAGE + 100)],
            [1, 2, 3, 4, 5]
        );
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// A marked page whose bytes the shaders left as they were (as for a
    /// third of the world's marked pages) goes back as nothing: title memory
    /// is not written, and the page stays taken (not changed, so the next
    /// refresh does not copy it again).
    #[test]
    fn a_marked_page_the_shaders_left_alone_stays_taken() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let (Some(copy), Some(b)) = (find(p as u64, PAGE as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        // The shaders write the byte the copy already has.
        copy.write_host((p + 10) as u64, &[0]);
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);
        let t = copy.tracked.as_ref().expect("tracked memory");
        assert!(!t.changed(p / PAGE), "still taken");
        assert_eq!(copy.copy_back(p, PAGE), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    thread_local! {
        /// Title writes [`after_read`] makes, last first.
        static AFTER_READ: std::cell::RefCell<Vec<(usize, u8)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A title thread's write landing just after a copy-back read the page
    /// (tests): the next of [`AFTER_READ`]'s.
    pub(super) fn after_read() {
        if let Some((at, value)) = AFTER_READ.with(|w| w.borrow_mut().pop()) {
            // SAFETY: a byte of the test's title memory (guarded or not: a
            // guard's fault opens it).
            unsafe { (at as *mut u8).write_volatile(value) };
        }
    }

    /// The title writes a page while it goes back, just after the copy-back
    /// read it. Only the bytes the shaders changed go into title memory, for
    /// a page marked through views and a page shaders wrote at computed
    /// addresses alike: the title's byte stays (a whole page put back undid
    /// it), and the copy has it too.
    #[test]
    fn a_copy_back_puts_back_only_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        // The shaders write a byte in each page.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        let title = |at: usize| (p + at) as *mut u8;
        AFTER_READ.with(|w| *w.borrow_mut() = vec![(p + PAGE + 40, 0x88), (p + 40, 0x77)]);
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        // SAFETY: (reads below) bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        // Page 0 was marked through a view.
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);
        assert_eq!(
            (read(10), read(40)),
            (0xaa, 0x77),
            "the shader's byte and the title's, page 0"
        );
        assert_eq!(host(40), 0x77, "the copy reads the title's byte");
        // Page 1 was written at addresses shaders computed.
        assert_eq!(copy.copy_back_page(p + PAGE), PAGE);
        assert_eq!(
            (read(PAGE + 10), read(PAGE + 40)),
            (0xbb, 0x88),
            "both bytes, page 1"
        );
        assert_eq!(host(PAGE + 40), 0x88, "the copy reads the title's byte");
        unregister(p, p as u64);
    }

    /// The block-wise comparison finds the same runs as a byte-wise one:
    /// across block and line edges, at the ends, with the GPU's bytes told
    /// apart, and for lengths that are not whole blocks.
    #[test]
    fn differing_matches_a_byte_wise_comparison() {
        let naive = |host: &[u8], title: &[u8], gpu: &dyn Fn(usize) -> bool| {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for i in 0..host.len() {
                if host[i] != title[i] && gpu(i) {
                    match runs.last_mut() {
                        Some((_, end)) if *end == i => *end = i + 1,
                        _ => runs.push((i, i + 1)),
                    }
                }
            }
            runs
        };
        for len in [0, 1, 31, 33, 1023, 1025, PAGE - 7, PAGE] {
            let title = vec![0x5au8; len];
            let mut host = title.clone();
            // Single bytes, runs over line and block edges, the last byte.
            let mut at = 3;
            while at < len {
                host[at] ^= 0xff;
                at += 997;
            }
            for i in (1020..1060).filter(|&i| i < len) {
                host[i] = 0;
            }
            if let Some(last) = host.last_mut() {
                *last = 1;
            }
            let gpu = |i: usize| !i.is_multiple_of(5);
            assert_eq!(
                differing(&host, &title, gpu),
                naive(&host, &title, &gpu),
                "length {len}"
            );
            assert_eq!(
                differing(&host, &title, |_| true),
                naive(&host, &title, &|_| true),
                "length {len}"
            );
            // By lines: in order, apart, on line edges, each line holding a
            // differing byte, and every differing byte inside one.
            let rows = differing_rows(&host, &title);
            assert!(rows.windows(2).all(|w| w[0].1 < w[1].0), "length {len}: apart, in order");
            for &(a, b) in &rows {
                assert!(a % 32 == 0 && (b % 32 == 0 || b == len) && a < b, "length {len}: {a}..{b} on line edges");
                for line in (a..b).step_by(32) {
                    let end = (line + 32).min(b);
                    assert!(host[line..end] != title[line..end], "length {len}: line {line} differs");
                }
            }
            for (a, b) in naive(&host, &title, &|_| true) {
                assert!(
                    rows.iter().any(|&(x, y)| x <= a && b <= y),
                    "length {len}: {a}..{b} inside a line run"
                );
            }
        }
    }

    /// A marked page's copy-back into view memory goes through our own view
    /// of the section: the lines go back, the title's other bytes stay (one
    /// it writes after the copy-back read the page too), the guard stays
    /// without being opened, and the writer's tracker needs no take; into
    /// private memory it goes as before.
    #[test]
    fn marked_copy_backs_into_view_memory_go_beside_the_guard() {
        let (_, a, _) = crate::watch::tests::reported_section(1);
        let t = crate::watch::track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        let read_title = vec![0u8; PAGE];
        let mut host = read_title.clone();
        host[100] = 0xaa;
        host[5000] = 0xab;
        let runs = differing_rows(&host, &read_title);
        // SAFETY: the title's view, guarded: the handler opens it.
        unsafe { ((a + 2000) as *mut u8).write_volatile(0x13) };
        // The page is changed now (the title wrote it): taken again first,
        // as a copy-back finds it after a refresh.
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(write_runs_by(a, &host, &runs, Some(&t)), Put::Beside);
        // SAFETY: bytes of the view.
        let read = |at: usize| unsafe { ((a + at) as *const u8).read_volatile() };
        assert_eq!((read(100), read(5000), read(2000), read(101)), (0xaa, 0xab, 0x13, 0));
        assert!(t.guarded(a / PAGE) && t.clean(), "guarded all along, nothing to take");
        // Private memory: as before.
        let p = crate::watch::tests::pool(1);
        let v = crate::watch::track(p, PAGE, true).expect("pool memory is tracked");
        assert_eq!(v.take(p, p + PAGE), [(p, p + PAGE)]);
        assert_eq!(write_runs_by(p, &host, &runs, Some(&v)), Put::Title);
        // SAFETY: bytes of the pool.
        assert_eq!(unsafe { ((p + 5000) as *const u8).read_volatile() }, 0xab);
        assert!(v.changed(p / PAGE) && !v.guarded(p / PAGE), "opened and changed: the caller takes it");
        drop(v);
        crate::watch::tests::free(p);
    }

    /// A copy-back with no bytes saved puts back whole 32-byte lines: title
    /// bytes outside the lines the shaders changed stay, also those the title
    /// writes while the page goes back. With bytes saved (the title wrote the
    /// page after the copy took it) it stays byte by byte: a title write in
    /// the shader's very line, before and while the page goes back, stays.
    #[test]
    fn copy_backs_by_lines_leave_the_titles_other_bytes() {
        // By lines, into plain memory: the title writes bytes after the
        // copy-back read them, outside the changed lines (next line, a gap
        // over SPAN_GAP away, the last byte) and they stay.
        let read_title = vec![0x5au8; PAGE];
        let mut host = read_title.clone();
        host[100] = 0xaa;
        host[5000] = 0xab;
        let runs = differing_rows(&host, &read_title);
        assert_eq!(runs, [(96, 128), (4992, 5024)]);
        let mut memory = read_title.clone();
        memory[128] = 0x11;
        memory[95] = 0x12;
        memory[2000] = 0x13;
        memory[PAGE - 1] = 0x14;
        assert!(write_runs(memory.as_mut_ptr() as usize, &host, &runs));
        let mut want = read_title.clone();
        (want[100], want[5000], want[128], want[95], want[2000], want[PAGE - 1]) = (0xaa, 0xab, 0x11, 0x12, 0x13, 0x14);
        assert!(memory == want, "only the two lines written");

        // Through the copy: page 0 goes back by lines (nothing saved), page
        // 1 byte by byte (saved).
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        // SAFETY: as above.
        unsafe { title(130).write_volatile(0x77) };
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        // The shaders write a byte in line 96..128 of each page.
        copy.write_host((p + 100) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 100) as u64, &[0xbb]);
        // Page 1: the title writes the shader's line before the copy-back
        // (saved), and again while the copy-backs run.
        // SAFETY: as above.
        unsafe { title(PAGE + 105).write_volatile(0x55) };
        AFTER_READ.with(|w| *w.borrow_mut() = vec![(p + PAGE + 110, 0x66)]);
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        assert_eq!((read(100), read(130)), (0xaa, 0x77), "page 0: the shader's line, the title's byte after it");
        assert_eq!(
            (read(PAGE + 100), read(PAGE + 105), read(PAGE + 110)),
            (0xbb, 0x55, 0x66),
            "page 1 (saved): the shader's byte and both title writes in its line"
        );
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p);
        copy.copy_back_marked(p + PAGE);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// A marked page of view memory goes back through our own view of the
    /// section: the shader's byte lands, the title's guard stays without
    /// being opened, and the copy needs no take (it holds what title memory
    /// does). A page the title wrote before the copy-back (open already)
    /// goes as before: both bytes stay, and it is taken.
    #[test]
    fn marked_pages_of_view_memory_go_back_beside_the_guard() {
        let (section, a, _) = crate::watch::tests::reported_section(2);
        let len = 2 * PAGE;
        register(a, a as u64, len as u64);
        let Some(copy) = find(a as u64, len as u64) else {
            unregister(a, a as u64);
            crate::watch::title_memory_section_gone(section);
            return; // no host GPU here
        };
        copy.update(a, len);
        copy.writing(a, len, true);
        copy.write_host((a + 10) as u64, &[0xaa]);
        copy.write_host((a + PAGE + 10) as u64, &[0xbb]);
        // SAFETY: the title's view, guarded: the handler opens it.
        unsafe { ((a + PAGE + 40) as *mut u8).write_volatile(0x11) };
        assert_eq!(copy.copy_back_marked(a), PAGE);
        assert_eq!(copy.copy_back_marked(a + PAGE), PAGE);
        // SAFETY: bytes of the title's view.
        let read = |at: usize| unsafe { ((a + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10), read(PAGE + 40)), (0xaa, 0xbb, 0x11));
        let t = copy.tracked.as_ref().expect("views are tracked");
        assert!(t.guarded(a / PAGE) && !t.changed(a / PAGE), "page 0: guarded all along, nothing to take");
        assert!(t.guarded(a / PAGE + 1) && !t.changed(a / PAGE + 1), "page 1: as before, taken");
        copy.copy_back(a, len);
        unregister(a, a as u64);
        crate::watch::title_memory_section_gone(section);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
}
