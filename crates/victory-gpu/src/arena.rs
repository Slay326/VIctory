//! Pages in the GPU's memory for title memory no host copy holds (local
//! test branch, `VICTORY_BVH_ARENA=1`; off by default).
//!
//! The acceleration-structure builder reads and writes pools of title
//! memory that are in no title buffer (the world's ~3 GB at
//! 0xdd0000000-0xe8fffffff): the page table (`crate::memory`) names no copy
//! there, so its stages read zeros and their stores are dropped (run
//! 20261126-100335: 70,746 pages and 89 regions missed per submission, 1 of
//! 14 headers valid). With the switch such pages get a page of their own in
//! an arena in the GPU's memory:
//! * a directory over the title's address space in 64 KiB pages
//!   ([`Directory`]) names each page's slot in the arena, chunks of
//!   [`CHUNK_PAGES`] pages for a budget (`VICTORY_BVH_ARENA_MB`, 2048), all
//!   made at the first submission before the title's textures fill the card
//!   and kept resident with high priority ([`reserve`]); copies of title
//!   buffers get less of the card meanwhile (`crate::shadow`, 512 MiB);
//! * pages shaders reached with no copy (their bits in the lists' bitmaps)
//!   are asked for after the submission and given before the next one
//!   ([`provide`], at most `VICTORY_BVH_ARENA_STEP` (4096) new pages a
//!   submission); a 256 MiB region with no block first gets a block, so that
//!   the next marks there name pages;
//! * a new page takes title memory's bytes where memory is committed, else
//!   it stays zero; free memory gets no page (an address computed from
//!   zeros);
//! * the page tables of every view heap name arena pages where no copy holds
//!   the page, so the builder's later stages and the ray tracing programs
//!   read what earlier stages wrote, in any later submission or frame;
//! * nothing written there goes back into title memory: the structures stay
//!   on the card; of title buffers, only those of at most [`BACK`] bytes take
//!   the builder's writes back (larger ones keep them in their copies until
//!   the frame's end, as withheld writes);
//! * the title's own later writes to such pages (write-watched memory, the
//!   pools) go up 4 KiB at a time before each submission
//!   (`VICTORY_BVH_ARENA_REFRESH`, on).
//!
//! The arena principle follows the RX 7900 XTX fork (avatarmanz/VIctory-7900XTX,
//! GPL-3.0; its idea only).
//!
//! shortcut: a 4 KiB piece the title wrote goes up whole, over what shaders
//! wrote in it since.
//! shortcut: memory that is not write-watched (section views, placeholders)
//! keeps the bytes it had when its page was made.
//! shortcut: arena pages are never given back while the process runs.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// A page of the directory and the arena: 64 KiB, as the page table's.
pub(crate) const PAGE: u64 = 1 << 16;
/// Pages per chunk of the arena (64 MiB, one buffer and one view pair per
/// view heap).
pub(crate) const CHUNK_PAGES: u32 = 1024;
pub(crate) const CHUNK_BYTES: u64 = CHUNK_PAGES as u64 * PAGE;
/// The largest title buffer the builder's writes go back to with the arena
/// on (the small result buffers whose sizes the title reads); what it writes
/// to larger ones stays on the card (`crate::memory`, `crate::host`).
pub(crate) const BACK: u64 = 256 << 10;
/// Title pages below 2^40.
const PAGES: u64 = 1 << 24;
/// Bytes of the upload buffer new pages and the title's writes go up
/// through.
const STAGE: usize = 16 << 20;

/// Whether the arena is on (`VICTORY_BVH_ARENA=1`; a test sets its own on
/// its thread).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_ARENA").is_ok_and(|v| v.trim() == "1"))
}

fn number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// The most pages the arena holds (`VICTORY_BVH_ARENA_MB`, 2048).
fn most_pages() -> u32 {
    (number("VICTORY_BVH_ARENA_MB", 2048) << 20).div_ceil(PAGE).min(u64::from(u32::MAX)) as u32
}

/// The most bytes of one builder pool given whole ([`want_around`];
/// `VICTORY_BVH_ARENA_AROUND_MB`, 4096).
fn around_most() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_AROUND_MB", 4096) << 20)
}

/// The most new pages a submission gives (`VICTORY_BVH_ARENA_STEP`, 4096).
fn step_pages() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_STEP", 4096).max(1) as usize)
}

/// Whether reserved memory with nothing committed gets (zero) pages
/// (`VICTORY_BVH_ARENA_RESERVED`, on). On PC 2 the builder's pools are such
/// memory (run 132102: 134,297 missed pages from 0x730000000 on, its root
/// descriptors there too); committed pages are given first.
fn reserved_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_BVH_ARENA_RESERVED").is_ok_and(|v| v.trim() == "0"))
}

/// Bytes around a builder address in reserved memory given up front (a
/// 4 MiB window; the rest as shaders reach it).
const RESERVED_AROUND: u64 = 4 << 20;

fn refresh_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_BVH_ARENA_REFRESH").is_ok_and(|v| v.trim() == "0"))
}

/// A page's place in the arena: slot `n` is page `n % CHUNK_PAGES` of
/// chunk `n / CHUNK_PAGES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot(pub(crate) u32);

impl Slot {
    pub(crate) fn chunk(self) -> usize {
        (self.0 / CHUNK_PAGES) as usize
    }

    /// Byte offset in its chunk.
    pub(crate) fn offset(self) -> u64 {
        u64::from(self.0 % CHUNK_PAGES) * PAGE
    }
}

/// Why a page got no slot.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The arena holds its most pages.
    Budget,
    /// At or above 2^40, where the page table names nothing.
    Outside,
}

/// The directory over the title's address space: the arena slot of each
/// 64 KiB page that has one (by page number, address >> 16), slots given in
/// order and never taken back, at most `most`; and the 256 MiB regions
/// (level-1 indices) whose page-table block is wanted with no page in it.
pub(crate) struct Directory {
    pages: BTreeMap<u64, Slot>,
    most: u32,
    regions: BTreeSet<usize>,
}

impl Directory {
    pub(crate) fn new(most: u32) -> Directory {
        Directory {
            pages: BTreeMap::new(),
            most,
            regions: BTreeSet::new(),
        }
    }

    pub(crate) fn get(&self, page: u64) -> Option<Slot> {
        self.pages.get(&page).copied()
    }

    /// Slots given.
    pub(crate) fn len(&self) -> u32 {
        self.pages.len() as u32
    }

    /// The slot the next new page gets.
    pub(crate) fn next(&self) -> Slot {
        Slot(self.len())
    }

    /// Chunks the slots given fill.
    #[cfg(test)]
    pub(crate) fn chunks(&self) -> usize {
        self.len().div_ceil(CHUNK_PAGES) as usize
    }

    /// Page `page`'s slot, and whether it is new.
    pub(crate) fn add(&mut self, page: u64) -> Result<(Slot, bool), Refused> {
        if page >= PAGES {
            return Err(Refused::Outside);
        }
        if let Some(slot) = self.get(page) {
            return Ok((slot, false));
        }
        if self.len() >= self.most {
            return Err(Refused::Budget);
        }
        let slot = self.next();
        self.pages.insert(page, slot);
        Ok((slot, true))
    }

    /// Wants a block for region `l1`; whether it is new.
    pub(crate) fn add_region(&mut self, l1: usize) -> bool {
        l1 < 4096 && self.regions.insert(l1)
    }

    pub(crate) fn has_region(&self, l1: usize) -> bool {
        self.regions.contains(&l1)
    }

    /// Every page with a slot: its title address and slot.
    pub(crate) fn pages(&self) -> impl Iterator<Item = (u64, Slot)> + '_ {
        self.pages.iter().map(|(&p, &s)| (p * PAGE, s))
    }

    pub(crate) fn regions(&self) -> impl Iterator<Item = usize> + '_ {
        self.regions.iter().copied()
    }
}

/// Since the last frame line.
#[derive(Default)]
struct Stats {
    filled: u64,
    zero: u64,
    refused: u64,
    nothing: u64,
    regions: u64,
    writes: u64,
    refreshed: u64,
    fill_us: u64,
    refresh_us: u64,
    asked: u64,
    /// Pages at committed addresses of the host's memory (not given).
    host: u64,
    /// Builder pools given whole before their first run, and their pages.
    around_runs: u64,
    around: u64,
}

struct Arena {
    dir: Directory,
    chunks: Vec<ID3D12Resource>,
    /// Pages (numbers) and regions asked for since the last [`provide`].
    wanted: BTreeSet<u64>,
    /// Root descriptor addresses of the builder whose runs are wanted whole
    /// ([`want_around`]), and the runs already given (start).
    around: BTreeSet<u64>,
    around_done: HashSet<u64>,
    /// Pages such addresses lay in that hold no title memory, or that a copy
    /// holds (not looked at again).
    not_title: HashSet<u64>,
    wanted_regions: BTreeSet<usize>,
    /// Pages with no memory at their address when asked (not asked again).
    nothing: HashSet<u64>,
    /// Pages with slots whose title memory is write-watched, by page number:
    /// their allocation's base (runs of them are asked together).
    watched: BTreeMap<u64, usize>,
    generation: u64,
    /// Why the arena stopped growing, if it did (logged once).
    stopped: Option<String>,
    /// Whether its chunks were made up front ([`reserve`]).
    reserved: bool,
    stage: Option<(ID3D12Resource, usize)>,
    stats: Stats,
}

// SAFETY: D3D12 resources are free-threaded; `stage`'s pointer is the
// persistent mapping of its buffer, used only under the lock.
unsafe impl Send for Arena {}

static ARENA: Mutex<Option<Arena>> = Mutex::new(None);

fn arena() -> std::sync::MutexGuard<'static, Option<Arena>> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

fn new_arena() -> Arena {
    Arena {
        dir: Directory::new(most_pages()),
        chunks: Vec::new(),
        wanted: BTreeSet::new(),
        around: BTreeSet::new(),
        around_done: HashSet::new(),
        not_title: HashSet::new(),
        wanted_regions: BTreeSet::new(),
        nothing: HashSet::new(),
        watched: BTreeMap::new(),
        generation: 0,
        stopped: None,
        reserved: false,
        stage: None,
        stats: Stats::default(),
    }
}

fn with<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    let mut a = arena();
    f(a.get_or_insert_with(new_arena))
}

/// Whether the arena's chunks are made at the first submission, before the
/// title's textures fill the card (`VICTORY_BVH_ARENA_RESERVE`, on); off,
/// they are made as pages are asked for, while the process uses less than
/// 95 % of the card's budget.
fn reserve_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_BVH_ARENA_RESERVE").is_ok_and(|v| v.trim() == "0"))
}

/// Gives `resource` high residency priority: when the process uses more
/// than the card's budget (the RTX 3070 does in the world, its textures
/// alone ~5-7 GB), the video memory manager moves others to system memory
/// before it. Only where it lives changes, not what it holds.
fn keep_resident(host: &crate::host::Host, resource: &ID3D12Resource) -> bool {
    use windows::Win32::Graphics::Direct3D12::{D3D12_RESIDENCY_PRIORITY_HIGH, ID3D12Device1, ID3D12Pageable};
    use windows::core::Interface;
    let (Ok(device), Ok(pageable)) = (host.device.cast::<ID3D12Device1>(), resource.cast::<ID3D12Pageable>()) else {
        return false;
    };
    // SAFETY: one live object and its priority.
    unsafe { device.SetResidencyPriority(1, &Some(pageable), &D3D12_RESIDENCY_PRIORITY_HIGH) }.is_ok()
}

/// Makes the arena's chunks for its whole budget now (the first submission,
/// while the card has room), each kept resident ahead of other memory. A
/// chunk that cannot be made ends it there and the budget shrinks to what
/// was made; logged once, with who yields to whom.
fn reserve(host: &crate::host::Host, a: &mut Arena) {
    a.reserved = true;
    let before = host.card_usage();
    let wanted = a.dir.most.div_ceil(CHUNK_PAGES) as usize;
    let mut resident = 0;
    while a.chunks.len() < wanted {
        let Some(chunk) = host.card_buffer(CHUNK_BYTES) else {
            let why = format!("chunk {} of {wanted} could not be made", a.chunks.len());
            tracing::warn!(target: "victory_gpu::arena", "BVH arena reserved less: {why}");
            break;
        };
        crate::host::name(&chunk, || format!("BVH arena chunk {}", a.chunks.len()));
        resident += usize::from(keep_resident(host, &chunk));
        a.chunks.push(chunk);
    }
    a.dir.most = a.dir.most.min(a.chunks.len() as u32 * CHUNK_PAGES);
    let mb = |b: u64| b >> 20;
    let card = |u: Option<(u64, u64)>| u.map_or("unknown".to_owned(), |(budget, usage)| format!("{} of {} MB", mb(usage), mb(budget)));
    let (copies, copies_was) = crate::shadow::card_cap();
    tracing::info!(
        target: "victory_gpu::arena",
        "BVH arena reserved {} chunks = {} MB up front ({} of them with high residency priority: textures and other memory go to system memory before them \
         when the card is full); card in use before {}, after {}; copies of title buffers in GPU memory capped at {} MB (without the arena {} MB)",
        a.chunks.len(),
        (a.chunks.len() as u64 * CHUNK_BYTES) >> 20,
        resident,
        card(before),
        card(host.card_usage()),
        mb(copies),
        mb(copies_was),
    );
}

/// A list using title memory reached `addresses` with no copy: their pages
/// are wanted before the next submission.
pub(crate) fn want_pages(addresses: &[u64]) {
    if addresses.is_empty() {
        return;
    }
    with(|a| {
        a.stats.asked += addresses.len() as u64;
        a.wanted.extend(addresses.iter().map(|&at| at / PAGE));
    });
}

/// The BVH builder is recorded with a root descriptor at `address`, which
/// no copy holds: the committed run of title memory around it (its pool)
/// gets arena pages before the submission runs, so its first run reads the
/// pool's bytes instead of zeros (with zeros its bottom-up stage loops on
/// parent links 0 for seconds: runs 125619/125814, 24-25 s submissions).
pub(crate) fn want_around(address: u64) {
    if !enabled() {
        return;
    }
    with(|a| {
        let page = address / PAGE;
        if a.dir.get(page).is_none() && !a.not_title.contains(&page) {
            a.around.insert(address);
        }
    });
}

/// Addresses a builder dispatch's root constants may hold: dword pairs (low,
/// high below 2^40) and single dwords in 256-byte units at or above 4 GiB
/// (where the title's GPU memory lies). Those in title memory no copy holds
/// get their committed run as [`want_around`] does; the rest are passed by.
pub(crate) fn want_constants(values: &[u32]) {
    if !enabled() {
        return;
    }
    let candidates = constant_addresses(values);
    if candidates.is_empty() {
        return;
    }
    with(|a| {
        for address in candidates {
            let page = address / PAGE;
            if a.dir.get(page).is_none() && !a.not_title.contains(&page) {
                a.around.insert(address);
            }
        }
    });
}

/// The addresses [`want_constants`] looks at.
fn constant_addresses(values: &[u32]) -> Vec<u64> {
    let pairs = values.windows(2).filter(|w| (1..256).contains(&w[1])).map(|w| u64::from(w[0]) | u64::from(w[1]) << 32);
    let units = values.iter().map(|&v| u64::from(v) << 8).filter(|&a| a >= 1 << 32);
    pairs.chain(units).collect()
}

/// A list reached memory in 256 MiB region `l1`, which has no block.
pub(crate) fn want_region(l1: usize) {
    with(|a| {
        a.wanted_regions.insert(l1);
    });
}

/// Shaders wrote `n` arena pages (they stay on the card).
pub(crate) fn note_writes(n: u64) {
    with(|a| a.stats.writes += n);
}

/// Changes whenever the arena names more pages or regions (0: none).
pub(crate) fn generation() -> u64 {
    arena().as_ref().map_or(0, |a| a.generation)
}

/// What page tables name of the arena.
#[derive(Default)]
pub(crate) struct View {
    pub(crate) generation: u64,
    pub(crate) chunks: Vec<ID3D12Resource>,
    /// Title address of each page with a slot, and the slot.
    pub(crate) pages: Vec<(u64, Slot)>,
    pub(crate) regions: Vec<usize>,
}

pub(crate) fn view() -> View {
    arena().as_ref().map_or_else(View::default, |a| View {
        generation: a.generation,
        chunks: a.chunks.clone(),
        pages: a.dir.pages().collect(),
        regions: a.dir.regions().collect(),
    })
}

/// What memory is at a title address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Memory {
    Free,
    Reserved,
    /// Committed, but the host's (not the title's).
    Host,
    /// Committed; the allocation's base and whether the page is
    /// write-watched.
    Committed(usize, bool),
}

fn memory_at(address: u64) -> Memory {
    use windows_sys::Win32::System::Memory::{GetWriteWatch, MEM_COMMIT, MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data, all zeros is valid.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: plain data filled in by the query.
    let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    if n == 0 || info.State == MEM_FREE {
        return Memory::Free;
    }
    if info.State != MEM_COMMIT {
        return Memory::Reserved;
    }
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    // Its 4 KiB page alone: a range past the allocation fails.
    let watched = unsafe { GetWriteWatch(0, (address & !0xfff) as *const _, 4096, (&raw mut found).cast(), &mut count, &mut granularity) == 0 };
    // The title's memory: its private allocations are write-watched, its
    // physical pages mapped views; the host's own (heaps, images) is neither.
    if !watched && info.Type != windows_sys::Win32::System::Memory::MEM_MAPPED {
        return Memory::Host;
    }
    Memory::Committed(info.AllocationBase as usize, watched)
}

/// The committed run of title memory around `address`: start and bytes; in
/// reserved memory, the [`RESERVED_AROUND`] window holding it (inside its
/// allocation).
fn committed_around(address: u64) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let query = |at: u64| -> Option<MEMORY_BASIC_INFORMATION> {
        // SAFETY: plain data, all zeros is valid.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: plain data filled in by the query.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        (n != 0).then_some(info)
    };
    let info = query(address)?;
    if info.State == MEM_FREE {
        return None;
    }
    let alloc = info.AllocationBase as u64;
    let region_end = info.BaseAddress as u64 + info.RegionSize as u64;
    if info.State != MEM_COMMIT {
        if !reserved_on() {
            return None;
        }
        let window = address & !(RESERVED_AROUND - 1);
        let (from, to) = (window.max(alloc), (window + RESERVED_AROUND).min(region_end));
        return Some((from, to - from));
    }
    if !matches!(memory_at(address), Memory::Committed(..)) {
        return None;
    }
    // The committed regions of its allocation that touch it: from the
    // allocation's start, the run holding `address`, then on while committed.
    let (mut at, mut start) = (alloc, alloc);
    while at <= address {
        let r = query(at)?;
        let end = r.BaseAddress as u64 + r.RegionSize as u64;
        if r.State != MEM_COMMIT {
            start = end;
        }
        at = end.max(at + 4096);
    }
    let mut end = at;
    while let Some(r) = query(end).filter(|r| r.AllocationBase as u64 == alloc && r.State == MEM_COMMIT) {
        end = r.BaseAddress as u64 + r.RegionSize as u64;
    }
    Some((start, end - start))
}

/// Whether any memory (reserved or committed) lies in 256 MiB region `l1`.
fn region_has_memory(l1: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let (start, end) = ((l1 as u64) << 28, ((l1 as u64) + 1) << 28);
    let mut at = start.max(PAGE);
    while at < end {
        // SAFETY: plain data, all zeros is valid.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: plain data filled in by the query.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if info.State != MEM_FREE {
            return true;
        }
        at = (info.BaseAddress as u64 + info.RegionSize as u64).max(at + PAGE);
    }
    false
}

/// Reads title memory `[from, from + dst.len())` into `dst`, 4 KiB at a
/// time; pieces that cannot be read stay zero.
fn read_title(from: u64, dst: &mut [u8]) {
    for (k, piece) in dst.chunks_mut(4096).enumerate() {
        if !crate::host::read_into((from + 4096 * k as u64) as *const u8, piece) {
            piece.fill(0);
        }
    }
}

/// Before a submission (under the submission lock, no submission on the
/// GPU): gives the pages and regions asked for since the last one, filling
/// new pages from title memory, and sends up what the title wrote to arena
/// pages since.
pub(crate) fn provide() {
    if !enabled() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    let mut guard = arena();
    let a = guard.get_or_insert_with(new_arena);
    if reserve_on() && !a.reserved {
        reserve(host, a);
    }
    for l1 in std::mem::take(&mut a.wanted_regions) {
        if !a.dir.has_region(l1) && region_has_memory(l1) && a.dir.add_region(l1) {
            a.stats.regions += 1;
            a.generation += 1;
        }
    }
    let started = std::time::Instant::now();
    let mut wanted = std::mem::take(&mut a.wanted);
    let mut fills: Vec<(Slot, u64)> = Vec::new();
    // The builder's pools first, whole and in this submission.
    let mut first: Vec<(u64, Memory)> = Vec::new();
    let around_started = std::time::Instant::now();
    let around_runs = a.stats.around_runs;
    for address in std::mem::take(&mut a.around) {
        if a.dir.get(address / PAGE).is_some() {
            continue;
        }
        let in_copy = crate::shadow::find(address, 1).is_some();
        let Some((start, len)) = committed_around(address).filter(|_| !in_copy) else {
            a.not_title.insert(address / PAGE);
            continue;
        };
        if !a.around_done.insert(start) {
            continue;
        }
        let len = len.min(around_most());
        a.stats.around_runs += 1;
        for page in start / PAGE..(start + len).div_ceil(PAGE) {
            if a.dir.get(page).is_none() {
                wanted.remove(&page);
                first.push((page, memory_at(page * PAGE)));
            }
        }
    }
    // Committed memory first, then reserved (zero pages): addresses computed
    // from zeros land in reservations too, and must not take the budget
    // before the pools' pages do.
    let (mut committed, mut reserved, mut looked) = (Vec::new(), Vec::new(), 0);
    for page in wanted {
        if a.dir.get(page).is_some() || a.nothing.contains(&page) {
            continue;
        }
        if a.stopped.is_some() {
            a.stats.refused += 1;
            continue;
        }
        looked += 1;
        match memory_at(page * PAGE) {
            Memory::Free => {
                a.nothing.insert(page);
                a.stats.nothing += 1;
            }
            Memory::Host => {
                a.nothing.insert(page);
                a.stats.host += 1;
            }
            Memory::Reserved if !reserved_on() => {
                a.nothing.insert(page);
                a.stats.nothing += 1;
            }
            m @ Memory::Reserved => reserved.push((page, m)),
            m @ Memory::Committed(..) => committed.push((page, m)),
        }
    }
    let _ = looked;
    let around_pages = first.len();
    let mut given = 0;
    for (k, (page, memory)) in first.into_iter().chain(committed).chain(reserved).enumerate() {
        if k >= around_pages && given >= step_pages() {
            a.wanted.insert(page);
            continue;
        }
        if a.stopped.is_some() {
            a.stats.refused += 1;
            continue;
        }
        let next = a.dir.next();
        if next.chunk() >= a.chunks.len() && !a.reserved {
            if let Some(why) = chunk_refused(host, a.chunks.len()) {
                tracing::warn!(target: "victory_gpu::arena", pages = a.dir.len(), "BVH arena stops growing: {why}");
                a.stopped = Some(why);
                a.stats.refused += 1;
                continue;
            }
            let Some(chunk) = host.card_buffer(CHUNK_BYTES) else {
                let why = "a 64 MiB buffer in GPU memory could not be made".to_owned();
                tracing::warn!(target: "victory_gpu::arena", pages = a.dir.len(), "BVH arena stops growing: {why}");
                a.stopped = Some(why);
                a.stats.refused += 1;
                continue;
            };
            crate::host::name(&chunk, || format!("BVH arena chunk {}", a.chunks.len()));
            a.chunks.push(chunk);
        }
        match a.dir.add(page) {
            Ok((slot, _)) => {
                if k < around_pages {
                    a.stats.around += 1;
                } else {
                    given += 1;
                }
                a.generation += 1;
                if let Memory::Committed(base, watched) = memory {
                    if watched {
                        a.watched.insert(page, base);
                    }
                    fills.push((slot, page * PAGE));
                    a.stats.filled += 1;
                } else {
                    a.stats.zero += 1;
                }
            }
            Err(Refused::Budget) => {
                let why = format!("its budget of {} pages (VICTORY_BVH_ARENA_MB) is used", a.dir.len());
                tracing::warn!(target: "victory_gpu::arena", "BVH arena stops growing: {why}");
                a.stopped = Some(why);
                a.stats.refused += 1;
            }
            Err(Refused::Outside) => {
                a.nothing.insert(page);
                a.stats.nothing += 1;
            }
        }
    }
    // Pages that took title memory go up: write tracking is reset first, so
    // a write while they are read goes up at the next refresh.
    let pieces: Vec<(Slot, u64, u64)> = fills
        .iter()
        .map(|&(slot, at)| {
            if a.watched.contains_key(&(at / PAGE)) {
                reset_watch(at, PAGE as usize);
            }
            (slot, at, PAGE)
        })
        .collect();
    if !pieces.is_empty() {
        upload(a, &pieces);
        a.stats.fill_us += started.elapsed().as_micros() as u64;
    }
    // Logged at once: a submission that hangs never reaches a frame line.
    if a.stats.around_runs > around_runs {
        tracing::info!(
            target: "victory_gpu::arena",
            "BVH arena: builder memory given before its run: {} runs of title memory, {} pages in this submission ({:.1} ms); arena {} of {} pages{}",
            a.stats.around_runs - around_runs,
            around_pages,
            around_started.elapsed().as_secs_f64() * 1e3,
            a.dir.len(),
            a.dir.most,
            a.stopped.as_deref().map(|w| format!(" (stopped: {w})")).unwrap_or_default(),
        );
    }
    if refresh_on() && !a.watched.is_empty() {
        let started = std::time::Instant::now();
        refresh(a);
        a.stats.refresh_us += started.elapsed().as_micros() as u64;
    }
}

/// Why chunk `k` is not to be made, if it is not: the card's own budget
/// would be nearly used (95 %).
fn chunk_refused(host: &crate::host::Host, k: usize) -> Option<String> {
    let (budget, usage) = host.card_usage()?;
    (usage + CHUNK_BYTES > budget / 20 * 19).then(|| {
        format!(
            "chunk {k}: the process uses {} of {} MB of GPU memory",
            usage >> 20,
            budget >> 20
        )
    })
}

fn reset_watch(at: u64, len: usize) {
    use windows_sys::Win32::System::Memory::ResetWriteWatch;
    // SAFETY: write tracking of the title's memory; nothing else changes.
    unsafe { ResetWriteWatch(at as *const _, len) };
}

/// Sends up title memory into the arena: pieces (slot, title address of a
/// byte in the slot's page, bytes), through the stage buffer.
fn upload(a: &mut Arena, pieces: &[(Slot, u64, u64)]) {
    let Some(host) = crate::host::host() else { return };
    if a.stage.is_none() {
        let Some((buffer, data)) = host.upload_buffer(STAGE as u64) else {
            tracing::warn!(target: "victory_gpu::arena", "no upload buffer for the BVH arena: its pages stay zero");
            return;
        };
        crate::host::name(&buffer, || "BVH arena stage".into());
        a.stage = Some((buffer, data as usize));
    }
    let Some((stage, data)) = a.stage.clone() else { return };
    let mut k = 0;
    while k < pieces.len() {
        let mut used = 0usize;
        let mut batch = Vec::new();
        while let Some(&(slot, at, len)) = pieces.get(k) {
            if used + len as usize > STAGE {
                break;
            }
            // SAFETY: inside the mapped stage buffer, which the GPU is done
            // with (each batch is waited for).
            let dst = unsafe { std::slice::from_raw_parts_mut((data + used) as *mut u8, len as usize) };
            read_title(at, dst);
            batch.push((slot, at % PAGE, used as u64, len));
            used += len as usize;
            k += 1;
        }
        let chunks = &a.chunks;
        let ran = crate::host::run_now(|list| {
            for &(slot, inside, from, len) in &batch {
                // SAFETY: live buffers; the ranges lie in them.
                unsafe { list.CopyBufferRegion(&chunks[slot.chunk()], slot.offset() + inside, &stage, from, len) };
            }
        });
        if !ran {
            tracing::warn!(target: "victory_gpu::arena", "BVH arena pages could not be sent up");
            return;
        }
    }
}

/// What the title wrote to write-watched arena pages since they were last
/// sent up goes up, 4 KiB at a time.
fn refresh(a: &mut Arena) {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    /// `GetWriteWatch`'s flag that resets what it reports.
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    // Runs of pages in one allocation.
    let mut runs: Vec<(u64, u64, usize)> = Vec::new();
    for (&page, &base) in &a.watched {
        match runs.last_mut() {
            Some((first, n, b)) if *b == base && *first + *n == page => *n += 1,
            _ => runs.push((page, 1, base)),
        }
    }
    let mut pieces = Vec::new();
    let mut found = vec![0usize; 4096];
    for (first, n, _) in runs {
        // A few rounds while the answer fills `found`.
        for _ in 0..16 {
            let (mut count, mut granularity) = (found.len(), 0u32);
            // SAFETY: room for `count` addresses; the run is in one allocation.
            let failed = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    (first * PAGE) as *const _,
                    (n * PAGE) as usize,
                    found.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                ) != 0
            };
            if failed {
                break;
            }
            for &at in &found[..count] {
                let at = at as u64;
                if let Some(slot) = a.dir.get(at / PAGE) {
                    let g = u64::from(granularity.max(4096));
                    pieces.push((slot, at & !(g - 1), g));
                }
            }
            if count < found.len() {
                break;
            }
        }
    }
    if !pieces.is_empty() {
        a.stats.refreshed += pieces.len() as u64;
        upload(a, &pieces);
    }
}

/// The arena's bytes of the 64 KiB pages at `pages` (title addresses),
/// those with a slot (`crate::bvh_gt`, after a submission's fence): each
/// page's address and bytes.
pub(crate) fn fetch(pages: &[u64]) -> Vec<(u64, Vec<u8>)> {
    let Some(host) = crate::host::host() else { return Vec::new() };
    let guard = arena();
    let Some(a) = guard.as_ref() else { return Vec::new() };
    let slots: Vec<(u64, Slot)> = pages.iter().filter_map(|&p| Some((p & !(PAGE - 1), a.dir.get(p / PAGE)?))).collect();
    if slots.is_empty() {
        return Vec::new();
    }
    let Some((readback, data)) = host.readback_buffer(slots.len() as u64 * PAGE) else {
        return Vec::new();
    };
    let ran = crate::host::run_now(|list| {
        for (k, &(_, slot)) in slots.iter().enumerate() {
            // SAFETY: live buffers; the ranges lie in them.
            unsafe { list.CopyBufferRegion(&readback, k as u64 * PAGE, &a.chunks[slot.chunk()], slot.offset(), PAGE) };
        }
    });
    if !ran {
        return Vec::new();
    }
    slots
        .iter()
        .enumerate()
        .map(|(k, &(at, _))| {
            // SAFETY: the mapped readback buffer, the GPU done with it.
            let bytes = unsafe { std::slice::from_raw_parts((data as *const u8).add(k * PAGE as usize), PAGE as usize) };
            (at, bytes.to_vec())
        })
        .collect()
}

/// Logs what the arena did since the last call, if it is on and did
/// anything: a line per presented frame.
pub(crate) fn log_frame() {
    if !enabled() {
        return;
    }
    let mut guard = arena();
    let Some(a) = guard.as_mut() else { return };
    let s = std::mem::take(&mut a.stats);
    if s.asked + s.filled + s.zero + s.refused + s.nothing + s.regions + s.writes + s.refreshed + s.around_runs == 0 {
        return;
    }
    let (kept_out, kept_out_bytes) = crate::shadow::card_refused();
    tracing::info!(
        target: "victory_gpu::arena",
        "BVH arena this frame: builder pools given whole {} ({} pages); pages asked {} -> new {} ({} from title memory, {} zero, {:.1} ms), refused {}{}, \
         no memory there {}, host memory {}, \
         regions given a block {}; arena pages written by shaders {} (stay on the card); title writes sent up {} pieces ({:.1} ms); \
         total {} pages = {} MB in {} chunks of {} MB budget, {} regions, {} waiting; title buffer copies kept in system memory by the cap {} ({} MB)",
        s.around_runs,
        s.around,
        s.asked,
        s.filled + s.zero,
        s.filled,
        s.zero,
        s.fill_us as f64 / 1e3,
        s.refused,
        a.stopped.as_deref().map(|w| format!(" (stopped: {w})")).unwrap_or_default(),
        s.nothing,
        s.host,
        s.regions,
        s.writes,
        s.refreshed,
        s.refresh_us as f64 / 1e3,
        a.dir.len(),
        (u64::from(a.dir.len()) * PAGE) >> 20,
        a.chunks.len(),
        (u64::from(a.dir.most) * PAGE) >> 20,
        a.dir.regions.len(),
        a.wanted.len(),
        kept_out,
        kept_out_bytes >> 20,
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// The arena on or off for this thread's test, whatever the
        /// environment says.
        pub(crate) static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    #[test]
    fn slots_fill_chunks_in_order_and_keep_their_pages() {
        let mut d = Directory::new(3 * CHUNK_PAGES);
        assert_eq!(d.add(0xdd000), Ok((Slot(0), true)));
        assert_eq!(d.add(0xdd001), Ok((Slot(1), true)));
        assert_eq!(d.add(0xdd000), Ok((Slot(0), false)), "a page keeps its slot");
        assert_eq!(d.get(0xdd001), Some(Slot(1)));
        assert_eq!(d.get(0xdd002), None);
        assert_eq!(d.chunks(), 1);
        for p in 2..=CHUNK_PAGES as u64 {
            d.add(0xe0000 + p).unwrap();
        }
        assert_eq!(d.len(), CHUNK_PAGES + 1);
        assert_eq!(d.chunks(), 2);
        let last = d.get(0xe0000 + u64::from(CHUNK_PAGES)).unwrap();
        assert_eq!((last.chunk(), last.offset()), (1, 0), "slot 1024 starts chunk 1");
        assert_eq!((Slot(1023).chunk(), Slot(1023).offset()), (0, 1023 * PAGE));
        let pages: Vec<(u64, Slot)> = d.pages().take(2).collect();
        assert_eq!(pages, [(0xdd000 * PAGE, Slot(0)), (0xdd001 * PAGE, Slot(1))]);
    }

    #[test]
    fn the_budget_refuses_new_pages_but_not_old_ones() {
        let mut d = Directory::new(2);
        d.add(1).unwrap();
        d.add(2).unwrap();
        assert_eq!(d.add(3), Err(Refused::Budget));
        assert_eq!(d.add(1), Ok((Slot(0), false)), "a page with a slot still has it");
        assert_eq!(d.len(), 2);
        assert_eq!(d.add(PAGES), Err(Refused::Outside), "2^40 and above");
        assert_eq!(Directory::new(0).add(5), Err(Refused::Budget));
    }

    #[test]
    fn root_constants_name_addresses_as_pairs_and_in_256_byte_units() {
        // A pool address as a pair, a count, and an address in 256-byte
        // units (0xd_7227_a400 >> 8).
        let values = [0x7227_a400, 0xd, 5768, 0x0d72_27a4];
        let found = constant_addresses(&values);
        assert!(found.contains(&0xd_7227_a400), "{found:x?}");
        assert!(found.contains(&(0x0d72_27a4u64 << 8)), "{found:x?}");
        assert!(!found.iter().any(|&a| a == 5768 << 8 || a < 1 << 32), "small values are no addresses: {found:x?}");
        assert!(constant_addresses(&[1, 0x300, 0x400]).is_empty());
    }

    #[test]
    fn regions_are_wanted_once() {
        let mut d = Directory::new(1);
        assert!(d.add_region(0xdd));
        assert!(!d.add_region(0xdd));
        assert!(!d.add_region(4096), "past 2^40");
        assert!(d.has_region(0xdd));
        assert_eq!(d.regions().collect::<Vec<_>>(), [0xdd]);
    }

    #[test]
    fn the_budget_comes_from_megabytes() {
        // The default: 2048 MB in 64 KiB pages (unless the environment
        // sets it).
        if std::env::var_os("VICTORY_BVH_ARENA_MB").is_none() {
            assert_eq!(most_pages(), 32768);
        }
    }

    #[test]
    fn free_reserved_and_committed_memory_are_told_apart() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: fresh reservations of ours.
        let (reserved, committed) = unsafe {
            (
                VirtualAlloc(std::ptr::null(), 2 * PAGE as usize, MEM_RESERVE, PAGE_READWRITE) as u64,
                VirtualAlloc(std::ptr::null(), PAGE as usize, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) as u64,
            )
        };
        assert_eq!(memory_at(reserved), Memory::Reserved);
        assert_eq!(memory_at(committed), Memory::Committed(committed as usize, true));
        assert!(region_has_memory((reserved >> 28) as usize));
        // The host's own memory (not write-watched, not a view) is no
        // title memory.
        let heap = Box::new([0u8; 1 << 17]);
        assert_eq!(memory_at(heap.as_ptr() as u64), Memory::Host);
        assert_eq!(committed_around(heap.as_ptr() as u64), None);
        // A pool: its committed run.
        assert_eq!(committed_around(committed + 16), Some((committed, PAGE)));
        // Reserved memory: a window around the address, inside its allocation.
        assert_eq!(committed_around(reserved + PAGE + 8), Some((reserved, 2 * PAGE)));
        // SAFETY: ours.
        unsafe {
            VirtualFree(reserved as *mut _, 0, MEM_RELEASE);
            VirtualFree(committed as *mut _, 0, MEM_RELEASE);
        }
        assert_eq!(memory_at(reserved), Memory::Free);
    }
}
