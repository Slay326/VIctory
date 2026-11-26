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
/// (`VICTORY_BVH_ARENA_RESERVED=1`; off: placeholders hold nothing to read,
/// and on PC 2 more than 100,000 of their pages a submission were asked for,
/// addresses computed from zeros, until the card's memory ran out, runs
/// 135658/140104).
fn reserved_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_ARENA_RESERVED").is_ok_and(|v| v.trim() == "1"))
}

/// The most pages shaders' misses get a frame (`VICTORY_BVH_ARENA_FAULTS`,
/// 512); pages prepared before the builder runs do not count.
fn faults_per_frame() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_FAULTS", 512))
}

/// Frames in which the builder was recorded, whether it was in this one,
/// and dispatches held back while learning ([`learning_skips`]).
static BUILDER_FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BUILDER_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LEARN_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a builder dispatch of program `name` is held back because the
/// arena is still learning the builder's memory: in its first
/// `VICTORY_BVH_ARENA_LEARN` (3) frames, the stages that loop on zeros
/// (`VICTORY_BVH_ARENA_LEARN_SKIP`, default 5552dac6, 3b90165a, ffd93c8a,
/// f28a33a8: in flight at every device removal on PC 2) do not run; the
/// stages before them do, and the pages they reach become arena pages.
/// Their first runs read zeros from placeholders that only the GPU uses
/// (130,000 missed pages; runs 145223-150019: 4.9 s or a hang at 7-8 s).
pub(crate) fn learning_skips(name: &str) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if !enabled() {
        return false;
    }
    BUILDER_SEEN.store(true, Relaxed);
    static LEARN: OnceLock<u64> = OnceLock::new();
    static SKIP: OnceLock<Vec<String>> = OnceLock::new();
    let learn = *LEARN.get_or_init(|| number("VICTORY_BVH_ARENA_LEARN", 8));
    if BUILDER_FRAMES.load(Relaxed) >= learn {
        return false;
    }
    let skip = SKIP.get_or_init(|| {
        std::env::var("VICTORY_BVH_ARENA_LEARN_SKIP")
            .unwrap_or_else(|_| "5552dac6,3b90165a,ffd93c8a,f28a33a8".into())
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let held = skip.iter().any(|h| name.contains(h.as_str()));
    if held {
        LEARN_SKIPPED.fetch_add(1, Relaxed);
    }
    held
}

/// Bytes of committed title memory given from a builder root argument's
/// address on (1 MiB).
const AROUND_WINDOW: u64 = 1 << 20;

/// Pages prepared after each address that looks like a structure's in the
/// builder's input buffers (256 KiB).
const PREPARED_PAGES: u64 = 4;

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
    /// Slots of pages taken back ([`Directory::remove`]), given again first,
    /// and the slots ever given.
    free: Vec<Slot>,
    top: u32,
}

impl Directory {
    pub(crate) fn new(most: u32) -> Directory {
        Directory {
            pages: BTreeMap::new(),
            most,
            regions: BTreeSet::new(),
            free: Vec::new(),
            top: 0,
        }
    }

    /// Takes page `page`'s slot back (to be given again); its slot.
    pub(crate) fn remove(&mut self, page: u64) -> Option<Slot> {
        let slot = self.pages.remove(&page)?;
        self.free.push(slot);
        Some(slot)
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
        self.free.last().copied().unwrap_or(Slot(self.top))
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
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                self.top += 1;
                Slot(self.top - 1)
            }
        };
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
    /// Slots taken back for pages missed in many frames.
    evicted: u64,
    /// Pages of unwatched title memory sent up again.
    unwatched_sent: u64,
    /// Unwatched pages builder dispatches read: named (before the cap),
    /// read and compared, sent up because their bytes changed, and the
    /// time taken ([`refresh_builder_reads`]).
    reads_named: u64,
    reads_checked: u64,
    reads_sent: u64,
    reads_us: u64,
    nothing: u64,
    regions: u64,
    writes: u64,
    refreshed: u64,
    fill_us: u64,
    refresh_us: u64,
    asked: u64,
    /// Pages at committed addresses of the host's memory (not given).
    host: u64,
    /// Title writes not sent up because shaders wrote their arena page.
    kept: u64,
    /// Missed addresses below 256 MiB (dropped), and missed pages not yet
    /// missed in enough frames.
    low: u64,
    once: u64,
    /// Builder pools given whole before their first run, and their pages.
    around_runs: u64,
    around: u64,
    /// Addresses builder dispatches' root arguments named.
    candidates: u64,
    /// The builder's input buffers scanned (bytes) and structure addresses
    /// found in them (pages).
    scanned: u64,
    found: u64,
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
    /// Why the last chunk was not made (memory), for the frame line.
    no_room: Option<String>,
    /// Whether its chunks were made up front ([`reserve`]).
    reserved: bool,
    /// Whether its budget was cut to the card's ([`provide`], once).
    capped: bool,
    /// Pages misses got this frame ([`faults_per_frame`]).
    faults: u64,
    /// Arena pages shaders wrote ([`note_shader_pages`]).
    shader_pages: HashSet<u64>,
    /// Pages given as zeros (reserved memory), and where the search for a
    /// slot to take back goes on ([`victim`]).
    zero_pages: HashSet<u64>,
    evict_cursor: u64,
    /// Pages filled from title memory that is not write-watched (mapped
    /// views): their later title writes never go up ([`refresh_unwatched`]),
    /// and where the last refresh of them ended.
    unwatched: BTreeSet<u64>,
    unwatched_cursor: u64,
    /// Addresses builder dispatches' root arguments named since the last
    /// [`provide`], pages with a slot or not ([`refresh_builder_reads`]).
    builder_reads: BTreeSet<u64>,
    /// Structure pages each input list pointed at when last scanned.
    input_pointers: BTreeMap<(u64, u64), Vec<u64>>,
    /// Unwatched pages: the hash of the bytes last sent up, and the frame
    /// they were last compared in.
    sent: std::collections::HashMap<u64, (u64, u64)>,
    /// Missed pages without a slot: in how many frames, and the last one
    /// ([`want_pages`]; old entries pruned in [`log_frame`]).
    misses: std::collections::HashMap<u64, (u32, u64)>,
    /// A builder dispatch was recorded since the last [`provide`].
    builder_now: bool,
    /// Input scans so far, and when each input buffer was last scanned.
    scans: u64,
    scanned_at: BTreeMap<(u64, u64), u64>,
    /// Title buffers (copies) the builder's root views point into: its
    /// input lists, scanned for structure addresses before it runs.
    inputs: BTreeSet<(u64, u64)>,
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
        no_room: None,
        reserved: false,
        capped: false,
        faults: 0,
        shader_pages: HashSet::new(),
        zero_pages: HashSet::new(),
        evict_cursor: 0,
        unwatched: BTreeSet::new(),
        unwatched_cursor: 0,
        builder_reads: BTreeSet::new(),
        input_pointers: BTreeMap::new(),
        sent: std::collections::HashMap::new(),
        misses: std::collections::HashMap::new(),
        builder_now: false,
        scans: 0,
        scanned_at: BTreeMap::new(),
        inputs: BTreeSet::new(),
        stage: None,
        stats: Stats::default(),
    }
}

fn with<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    let mut a = arena();
    f(a.get_or_insert_with(new_arena))
}

/// Whether the arena's chunks are made at the first submission
/// (`VICTORY_BVH_ARENA_RESERVE=1`); by default they are made as pages are
/// asked for, while the card's own and shared memory together have room
/// ([`chunk_refused`]; up-front chunks ran PC 2 out of memory, run 135658).
fn reserve_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_ARENA_RESERVE").is_ok_and(|v| v.trim() == "1"))
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
    let frame = FRAME.load(std::sync::atomic::Ordering::Relaxed);
    let need = miss_frames();
    with(|a| {
        a.stats.asked += addresses.len() as u64;
        for &at in addresses {
            // Region 0 (below 256 MiB): addresses made from zeros.
            if at < LOW {
                a.stats.low += 1;
                continue;
            }
            let page = at / PAGE;
            if a.dir.get(page).is_some() {
                continue;
            }
            // Only pages missed in `need` different frames: addresses
            // computed from zeros change from frame to frame, the builder's
            // memory does not (run 150500: 130,000-142,000 missed pages a
            // submission against 32,768 of budget).
            let e = a.misses.entry(page).or_insert((0, u64::MAX));
            if e.1 != frame {
                *e = (e.0 + 1, frame);
            }
            if e.0 >= need {
                a.wanted.insert(page);
            } else {
                a.stats.once += 1;
            }
        }
    });
}

/// Frames presented while the arena is on (for [`want_pages`]).
static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Addresses below this are taken for addresses made from zeros (region 0).
const LOW: u64 = 1 << 28;

/// The frames a page must be missed in before it gets a page
/// (`VICTORY_BVH_ARENA_MISS_FRAMES`, 2; 1: at once).
fn miss_frames() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    // (The synthetic test ends no frames between its submissions.)
    if cfg!(test) {
        return 1;
    }
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_MISS_FRAMES", 2).max(1) as u32)
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
        a.stats.candidates += 1;
        a.builder_now = true;
        if reads_most() > 0 {
            a.builder_reads.insert(address);
        }
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
        a.builder_now = true;
        a.stats.candidates += candidates.len() as u64;
        for address in candidates {
            if reads_most() > 0 {
                a.builder_reads.insert(address);
            }
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

/// Whether the builder's writes to title buffers that do not go back stay
/// in their copies instead of taking title memory's bytes again
/// (`VICTORY_BVH_ARENA_KEEP=1`; off by default: run 162252 crashed the title\r\n/// after 217 s reading 0xffffffffffffffff, kept builder bytes went back with\r\n/// later copy-backs of the same pages, and misses rose to 54,517).
pub(crate) fn keep_builder_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    enabled() && *ON.get_or_init(|| std::env::var("VICTORY_BVH_ARENA_KEEP").is_ok_and(|v| v.trim() == "1"))
}

/// Arena pages shaders wrote (by title address): the title's writes there
/// no longer go up ([`refresh`]), they would land over the builder's.
pub(crate) fn note_shader_pages(pages: &[u64]) {
    with(|a| a.shader_pages.extend(pages.iter().map(|&p| p / PAGE)));
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
    // Regions are cut where protections differ (the change tracker guards
    // pages one by one), so the walk is bounded: past `WALK` regions the run
    // is taken from the address's own region on (run 135239: an unbounded
    // walk per candidate held a submission ~30 s).
    const WALK: usize = 256;
    let (mut at, mut start, mut steps) = (alloc.max(address.saturating_sub(around_most())), alloc, 0);
    if at > alloc {
        start = at;
    }
    while at <= address && steps < WALK {
        let r = query(at)?;
        let end = r.BaseAddress as u64 + r.RegionSize as u64;
        if r.State != MEM_COMMIT {
            start = end;
        }
        at = end.max(at + 4096);
        steps += 1;
    }
    if at <= address {
        (start, at) = (info.BaseAddress as u64, region_end);
    }
    let mut end = at;
    steps = 0;
    while let Some(r) = query(end).filter(|r| r.AllocationBase as u64 == alloc && r.State == MEM_COMMIT && steps < WALK) {
        end = r.BaseAddress as u64 + r.RegionSize as u64;
        steps += 1;
    }
    Some((start, end - start))
}

/// Whether misses at addresses holding no title memory are not counted as
/// misses (`VICTORY_BVH_MISS_UNMAPPED=ignore`, off by default): no page can
/// be given there (free memory, or the host's own), so such a miss never
/// ends and keeps every later list's writes withheld (runs 152953-162826:
/// 5-14 such pages a builder submission, e.g. 0x7b5d50000 in each). The
/// console's GPU reads zeros there and drops stores, as the lists did.
pub(crate) fn unmapped_misses_ignored() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_MISS_UNMAPPED").is_ok_and(|v| v.trim() == "ignore"))
}

/// Whether title memory can be at page address `address` (reserved or the
/// title's committed memory); a 256 MiB region `region` with none at all.
pub(crate) fn title_memory_at(address: u64) -> bool {
    !matches!(memory_at(address), Memory::Free | Memory::Host)
}

pub(crate) fn title_memory_in_region(l1: usize) -> bool {
    region_has_memory(l1)
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
    if !a.capped {
        // At most an eighth of the card's own memory budget, below
        // `VICTORY_BVH_ARENA_MB` (the world fills both the card and the
        // shared memory on a 12 GB card, run 140104).
        a.capped = true;
        if let Some((local, ..)) = host.memory_usage() {
            let share = number("VICTORY_BVH_ARENA_SHARE", 8).max(1);
            a.dir.most = a.dir.most.min((local / share / PAGE) as u32);
        }
        tracing::info!(target: "victory_gpu::arena", pages = a.dir.most, "BVH arena budget: {} MB", (u64::from(a.dir.most) * PAGE) >> 20);
    }
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
        // A copy holds it (asked first: the walk below is not free): a title
        // buffer the builder reads, scanned below.
        let copy = crate::shadow::holding(address);
        if let Some(range) = copy {
            a.inputs.insert(range);
        }
        let in_copy = copy.is_some();
        let Some((start, len)) = (!in_copy).then(|| committed_around(address)).flatten() else {
            a.not_title.insert(address / PAGE);
            continue;
        };
        // From the address's page on, at most `AROUND_WINDOW` bytes of the
        // run: whole runs took the budget (run 144749: 6 runs wanted 33,344
        // pages before the builder's first submission, then 71 runs 257,243).
        let from = (address & !(PAGE - 1)).max(start);
        let (start, len) = (from, (start + len - from).min(AROUND_WINDOW).min(around_most()));
        if !a.around_done.insert(start) {
            continue;
        }
        a.stats.around_runs += 1;
        for page in start / PAGE..(start + len).div_ceil(PAGE) {
            if a.dir.get(page).is_none() {
                wanted.remove(&page);
                first.push((page, memory_at(page * PAGE)));
            }
        }
    }
    // Before the builder runs: structures its input buffers point at
    // (256-byte aligned addresses below 2^40 in qwords), a few pages each,
    // where committed title memory no copy holds lies. Its first run read
    // zeros there and its stages looped for seconds (runs 131349-133206).
    let builder = std::mem::take(&mut a.builder_now);
    if builder {
        let found = scan_inputs(a);
        for page in found {
            for p in page..page + PREPARED_PAGES {
                if a.dir.get(p).is_some() || a.nothing.contains(&p) || a.not_title.contains(&p) {
                    break;
                }
                if crate::shadow::holding(p * PAGE).is_some() {
                    a.not_title.insert(p);
                    break;
                }
                match memory_at(p * PAGE) {
                    m @ Memory::Committed(..) => {
                        wanted.remove(&p);
                        first.push((p, m));
                    }
                    _ => {
                        a.nothing.insert(p);
                        break;
                    }
                }
            }
        }
    }
    // Committed memory first, then reserved (zero pages): addresses computed
    // from zeros land in reservations too, and must not take the budget
    // before the pools' pages do.
    let (mut committed, mut reserved, mut looked) = (Vec::new(), Vec::new(), 0);
    let mut evict_for: Vec<(u64, Memory)> = Vec::new();
    for page in wanted {
        if a.dir.get(page).is_some() || a.nothing.contains(&page) {
            continue;
        }
        if a.stopped.is_some() {
            // A page missed again and again takes the slot of one no shader
            // wrote ([`evict_for_persistent`]).
            if evict_on()
                && a.misses.get(&page).is_some_and(|e| e.0 >= persist_frames())
                && let m @ Memory::Committed(..) = memory_at(page * PAGE)
            {
                evict_for.push((page, m));
                continue;
            }
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
    let (mut given, mut no_room_now) = (0, false);
    for (k, (page, memory)) in first.into_iter().chain(committed).chain(reserved).enumerate() {
        if k >= around_pages && (given >= step_pages() || a.faults >= faults_per_frame()) {
            a.stats.refused += 1;
            continue;
        }
        if a.stopped.is_some() {
            a.stats.refused += 1;
            continue;
        }
        let next = a.dir.next();
        if next.chunk() >= a.chunks.len() && !a.reserved {
            if no_room_now {
                a.stats.refused += 1;
                continue;
            }
            if let Some(why) = chunk_refused(host, a.chunks.len()) {
                no_room_now = true;
                // Not for good: memory may free up (logged once per frame line).
                a.stats.refused += 1;
                a.no_room = Some(why);
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
                    a.faults += 1;
                }
                a.generation += 1;
                if let Memory::Committed(base, watched) = memory {
                    if watched {
                        a.watched.insert(page, base);
                    } else {
                        a.unwatched.insert(page);
                    }
                    fills.push((slot, page * PAGE));
                    a.stats.filled += 1;
                } else {
                    a.stats.zero += 1;
                    a.zero_pages.insert(page);
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
    evict_for_persistent(a, evict_for, &mut fills);
    refresh_unwatched(a, &mut fills);
    let reads = std::mem::take(&mut a.builder_reads);
    if builder {
        refresh_builder_reads(a, &reads, &mut fills);
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
    if a.stats.around_runs > around_runs || around_pages > 0 {
        tracing::info!(
            target: "victory_gpu::arena",
            "BVH arena: builder memory prepared before its run: {} runs of title memory, {} pages wanted in this submission ({:.1} ms); arena {} of {} pages{}",
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

/// Whether a full arena gives pages missed in many frames the slots of pages
/// no shader wrote (`VICTORY_BVH_ARENA_EVICT=1`, off by default, local
/// test): the budget filled with pages of addresses computed from zeros in
/// the loading screens, and the builder's inputs missed for the rest of the
/// run (runs 165348-174702: 3-15 pages a builder submission, the same ones
/// frame after frame, "refused 148-12,636" a frame, only 12-20 % of the
/// bottom levels the readers met valid).
fn evict_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BVH_ARENA_EVICT").is_ok_and(|v| v.trim() == "1"))
}

/// Frames a page must be missed in to take another's slot
/// (`VICTORY_BVH_ARENA_PERSIST`, 4).
fn persist_frames() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_PERSIST", 4).max(1) as u32)
}

/// The most slots taken back per submission (`VICTORY_BVH_ARENA_EVICT_STEP`,
/// 512).
fn evict_step() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_EVICT_STEP", 512) as usize)
}

/// A page to take the slot of: no shader wrote it (its bytes are title
/// memory's, or zeros), not missed itself; zero pages (reserved memory,
/// addresses computed from zeros) first, then the others in address order
/// from where the last search ended.
fn victim(a: &mut Arena) -> Option<u64> {
    let free = |a: &Arena, p: u64| !a.shader_pages.contains(&p) && a.dir.get(p).is_some();
    if let Some(&p) = a.zero_pages.iter().find(|&&p| free(a, p)) {
        a.zero_pages.remove(&p);
        return Some(p);
    }
    let from = a.evict_cursor;
    let found = a
        .dir
        .pages
        .range(from..)
        .chain(a.dir.pages.range(..from))
        .map(|(&p, _)| p)
        .find(|&p| !a.shader_pages.contains(&p) && !a.misses.get(&p).is_some_and(|e| e.0 >= persist_frames()));
    if let Some(p) = found {
        a.evict_cursor = p + 1;
    }
    found
}

/// Gives the pages in `wanted` (missed in [`persist_frames`] frames,
/// committed title memory) the slots of victims ([`victim`]), at most
/// [`evict_step`]; their fills join `fills`.
fn evict_for_persistent(a: &mut Arena, wanted: Vec<(u64, Memory)>, fills: &mut Vec<(Slot, u64)>) {
    for (k, (page, memory)) in wanted.into_iter().enumerate() {
        if k >= evict_step() {
            a.stats.refused += 1;
            continue;
        }
        let Some(old) = victim(a) else {
            a.stats.refused += 1;
            break;
        };
        a.dir.remove(old);
        a.watched.remove(&old);
        a.unwatched.remove(&old);
        a.sent.remove(&old);
        let Ok((slot, _)) = a.dir.add(page) else { continue };
        a.generation += 1;
        a.stats.evicted += 1;
        if let Memory::Committed(base, watched) = memory {
            if watched {
                a.watched.insert(page, base);
            } else {
                a.unwatched.insert(page);
            }
            fills.push((slot, page * PAGE));
            a.stats.filled += 1;
        }
    }
}

/// Pages of unwatched title memory sent up again whole per submission
/// (`VICTORY_BVH_ARENA_REFRESH_UNWATCHED`, 0 = off; local test): such a
/// page kept the bytes it had when it was made, and the builder's bottom
/// levels took in points at the origin with complete inputs (run 184001:
/// 58-70 % of the child boxes outside their parent's were outside only by
/// the origin; zero pages of reserved memory were not the cause, run
/// 184825).
fn refresh_unwatched(a: &mut Arena, fills: &mut Vec<(Slot, u64)>) {
    static N: OnceLock<usize> = OnceLock::new();
    let n = *N.get_or_init(|| number("VICTORY_BVH_ARENA_REFRESH_UNWATCHED", 0) as usize);
    if n == 0 || a.unwatched.is_empty() {
        return;
    }
    let from = a.unwatched_cursor;
    let pages: Vec<u64> = a.unwatched.range(from..).chain(a.unwatched.range(..from)).copied().take(n).collect();
    for p in pages {
        a.unwatched_cursor = p + 1;
        if a.shader_pages.contains(&p) {
            continue;
        }
        if let Some(slot) = a.dir.get(p) {
            fills.push((slot, p * PAGE));
            a.stats.unwatched_sent += 1;
        }
    }
}

/// Most unwatched pages a builder submission reads that are compared with
/// title memory and sent up again if changed
/// (`VICTORY_BVH_ARENA_REFRESH_READS=<pages>`, 0 = off; local test).
fn reads_most() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_BVH_ARENA_REFRESH_READS", 0) as usize)
}

/// Pages a builder dispatch reads that hold unwatched title memory, each
/// once and at most `most`: from each root address's page the
/// [`AROUND_WINDOW`] after it (`windows`), from each structure page its
/// input lists point at the [`PREPARED_PAGES`] after it (`pointed`); only
/// those with a slot (`has_slot`), filled from unwatched memory and not
/// written by shaders (those keep the builder's bytes).
fn unwatched_reads(
    windows: &[u64],
    pointed: &[u64],
    has_slot: impl Fn(u64) -> bool,
    unwatched: &BTreeSet<u64>,
    shader_pages: &HashSet<u64>,
    most: usize,
) -> Vec<u64> {
    let spans = windows
        .iter()
        .map(|&p| (p, AROUND_WINDOW / PAGE))
        .chain(pointed.iter().map(|&p| (p, PREPARED_PAGES)));
    let mut seen = HashSet::new();
    let mut pages = Vec::new();
    for (first, n) in spans {
        for p in first..first + n {
            if pages.len() >= most {
                return pages;
            }
            if unwatched.contains(&p) && !shader_pages.contains(&p) && has_slot(p) && seen.insert(p) {
                pages.push(p);
            }
        }
    }
    pages
}

/// A 64-bit hash of a page's bytes (to tell whether they changed).
fn page_hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for q in bytes.as_chunks::<8>().0 {
        h = (h ^ u64::from_le_bytes(*q)).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    h
}

/// Whether `page`'s bytes differ from those last sent up (or none were
/// recorded); records them as sent, compared in `frame`.
fn changed_since_sent(sent: &mut std::collections::HashMap<u64, (u64, u64)>, page: u64, bytes: &[u8], frame: u64) -> bool {
    let h = page_hash(bytes);
    match sent.insert(page, (h, frame)) {
        Some((old, _)) => old != h,
        None => true,
    }
}

/// Before a builder submission (`VICTORY_BVH_ARENA_REFRESH_READS`): the
/// unwatched pages its dispatches read (root addresses and what its input
/// lists point at, [`unwatched_reads`]) are read from title memory, and
/// those whose bytes changed since they were last sent go up again. Such
/// pages kept the bytes they had when they got their slot: with complete
/// inputs the builder's bottom levels took in points at the origin (runs
/// 184001-185839: 46-70 % of the child boxes outside their parent's were
/// outside only by the origin), and sending every unwatched page again in
/// turn cost too much (run 185839). Each page is compared once a frame.
fn refresh_builder_reads(a: &mut Arena, reads: &BTreeSet<u64>, fills: &mut Vec<(Slot, u64)>) {
    let most = reads_most();
    if most == 0 || a.unwatched.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let (mut windows, mut pointed) = (Vec::new(), Vec::new());
    let mut lists = BTreeSet::new();
    for &address in reads {
        match crate::shadow::holding(address) {
            Some(range) => {
                lists.insert(range);
            }
            None => windows.push(address / PAGE),
        }
    }
    for range in &lists {
        if let Some(pages) = a.input_pointers.get(range) {
            pointed.extend_from_slice(pages);
        }
    }
    windows.dedup();
    let filling: HashSet<u64> = fills.iter().map(|&(_, at)| at / PAGE).collect();
    let dir = &a.dir;
    let pages = unwatched_reads(&windows, &pointed, |p| dir.get(p).is_some(), &a.unwatched, &a.shader_pages, usize::MAX);
    a.stats.reads_named += pages.len() as u64;
    let frame = FRAME.load(std::sync::atomic::Ordering::Relaxed);
    let mut bytes = vec![0u8; PAGE as usize];
    let mut checked = 0;
    for p in pages {
        if checked >= most {
            break;
        }
        if filling.contains(&p) || a.sent.get(&p).is_some_and(|e| e.1 == frame) {
            continue;
        }
        let Some(slot) = a.dir.get(p) else { continue };
        checked += 1;
        read_title(p * PAGE, &mut bytes);
        if changed_since_sent(&mut a.sent, p, &bytes, frame) {
            fills.push((slot, p * PAGE));
            a.stats.reads_sent += 1;
        }
    }
    a.stats.reads_checked += checked as u64;
    a.stats.reads_us += started.elapsed().as_micros() as u64;
}

/// Why chunk `k` is not to be made now, if it is not: the card's own and
/// the shared memory together would be 90 % used (both ran full, run
/// 140104: a list could not close).
fn chunk_refused(host: &crate::host::Host, k: usize) -> Option<String> {
    let (lb, lu, sb, su) = host.memory_usage()?;
    (lu + su + CHUNK_BYTES > (lb + sb) / 10 * 9).then(|| {
        format!(
            "chunk {k}: card {} of {} MB, shared {} of {} MB in use",
            lu >> 20,
            lb >> 20,
            su >> 20,
            sb >> 20
        )
    })
}

/// Pages that 256-byte aligned qwords (high dword 1-255) in the builder's
/// input buffers point at, with no slot and not known to hold nothing; the
/// bytes scanned count. At most 65,536 pages.
fn scan_inputs(a: &mut Arena) -> BTreeSet<u64> {
    let mut found = BTreeSet::new();
    let mut buffer = vec![0u8; 4 << 20];
    // Input lists, not every big buffer a root view lands in (run 145457
    // scanned 3.6 GB a frame), each again only every 32nd scan.
    a.scans += 1;
    let scans = a.scans;
    let due: Vec<(u64, u64)> = a
        .inputs
        .iter()
        .copied()
        .filter(|&(_, len)| len <= 128 << 20)
        .filter(|range| a.scanned_at.get(range).is_none_or(|&at| scans - at >= 32))
        .collect();
    for &range in &due {
        a.scanned_at.insert(range, scans);
    }
    let keep = reads_most() > 0;
    for &(start, len) in &due {
        let mut pointed = BTreeSet::new();
        let mut at = start;
        while at < start + len && found.len() < 1 << 16 {
            let n = (start + len - at).min(buffer.len() as u64) as usize;
            read_title(at, &mut buffer[..n]);
            a.stats.scanned += n as u64;
            for q in buffer[..n].as_chunks::<8>().0 {
                let v = u64::from_le_bytes(*q);
                if (1..256).contains(&(v >> 32)) && v & 0xff == 0 {
                    let page = v / PAGE;
                    if keep && pointed.len() < 1 << 14 {
                        pointed.insert(page);
                    }
                    if a.dir.get(page).is_none() && !a.nothing.contains(&page) && !a.not_title.contains(&page) {
                        found.insert(page);
                    }
                }
            }
            at += n as u64;
        }
        if keep {
            a.input_pointers.insert((start, len), pointed.into_iter().collect());
        }
    }
    a.stats.found += found.len() as u64;
    found
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
                // Pages shaders wrote keep what they wrote (the title's
                // pieces went up over the builder's: ~42,000 a frame).
                if a.shader_pages.contains(&(at / PAGE)) {
                    a.stats.kept += 1;
                    continue;
                }
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
    {
        use std::sync::atomic::Ordering::Relaxed;
        let frame = FRAME.fetch_add(1, Relaxed) + 1;
        if frame.is_multiple_of(4) {
            // Pages not missed in the last 16 frames are forgotten.
            if let Some(a) = arena().as_mut() {
                a.misses.retain(|_, e| frame.saturating_sub(e.1) < 16);
            }
        }
        if BUILDER_SEEN.swap(false, Relaxed) {
            let frames = BUILDER_FRAMES.fetch_add(1, Relaxed) + 1;
            let skipped = LEARN_SKIPPED.swap(0, Relaxed);
            if skipped > 0 {
                tracing::info!(target: "victory_gpu::arena", frames, skipped, "BVH arena learning: builder stages that loop on zeros held back this frame");
            }
        }
    }
    let mut guard = arena();
    let Some(a) = guard.as_mut() else { return };
    let s = std::mem::take(&mut a.stats);
    a.faults = 0;
    let no_room = a.no_room.take();
    if s.asked + s.filled + s.zero + s.refused + s.nothing + s.regions + s.writes + s.refreshed + s.around_runs + s.scanned == 0 {
        return;
    }
    let (kept_out, kept_out_bytes) = crate::shadow::card_refused();
    tracing::info!(
        target: "victory_gpu::arena",
        "BVH arena this frame: input buffers scanned {} MB -> {} structure pages; prepared before the builder ran {} pages; builder root addresses {}, \
         builder memory given up front {} runs; no room for a chunk: {}; pages asked {} -> new {} ({} from title memory, {} zero, {:.1} ms), refused {}{}, slots taken back for pages missed in many frames {}, unwatched pages sent again {} (of {}), \
         unwatched pages builders read {} -> compared {} -> changed and sent {} ({:.1} ms), no memory there {}, host memory {}, below 256 MiB {}, missed in too few frames {} (tracked {}), \
         regions given a block {}; arena pages written by shaders {} (stay on the card); title writes sent up {} pieces ({:.1} ms), not sent (page written by shaders) {}; \
         total {} pages = {} MB in {} chunks of {} MB budget, {} regions, {} waiting; title buffer copies kept in system memory by the cap {} ({} MB)",
        s.scanned >> 20,
        s.found,
        s.around,
        s.candidates,
        s.around_runs,
        no_room.as_deref().unwrap_or("no"),
        s.asked,
        s.filled + s.zero,
        s.filled,
        s.zero,
        s.fill_us as f64 / 1e3,
        s.refused,
        a.stopped.as_deref().map(|w| format!(" (stopped: {w})")).unwrap_or_default(),
        s.evicted,
        s.unwatched_sent,
        a.unwatched.len(),
        s.reads_named,
        s.reads_checked,
        s.reads_sent,
        s.reads_us as f64 / 1e3,
        s.nothing,
        s.host,
        s.low,
        s.once,
        a.misses.len(),
        s.regions,
        s.writes,
        s.refreshed,
        s.refresh_us as f64 / 1e3,
        s.kept,
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
    fn a_builders_unwatched_pages_go_up_again_when_title_memory_changed() {
        // Pages 0x100-0x10f: a root address's window; 0x200-0x203: a
        // structure its input list points at; 0x300: read by nobody.
        let mut d = Directory::new(64);
        for p in [0x100, 0x105, 0x10f, 0x110, 0x202, 0x204, 0x300, 0x101] {
            d.add(p).unwrap();
        }
        let unwatched: BTreeSet<u64> = [0x100, 0x105, 0x10f, 0x110, 0x202, 0x204, 0x300].into();
        let shader_pages: HashSet<u64> = [0x105].into();
        let has = |p| d.get(p).is_some();
        let pages = unwatched_reads(&[0x100, 0x100], &[0x200], has, &unwatched, &shader_pages, usize::MAX);
        // 0x101 is watched (its writes go up anyway), 0x105 written by
        // shaders, 0x110 past the window, 0x204 past the structure's pages,
        // 0x300 not read; 0x100 once although named twice.
        assert_eq!(pages, [0x100, 0x10f, 0x202]);
        assert_eq!(unwatched_reads(&[0x100], &[0x200], has, &unwatched, &shader_pages, 1), [0x100], "at most `most`");
        // A page goes up the first time, not again while title memory keeps
        // its bytes, and again once the title wrote to it.
        let mut sent = std::collections::HashMap::new();
        let mut bytes = vec![0u8; PAGE as usize];
        assert!(changed_since_sent(&mut sent, 0x100, &bytes, 1));
        assert!(!changed_since_sent(&mut sent, 0x100, &bytes, 2));
        bytes[0x8000] = 1;
        assert!(changed_since_sent(&mut sent, 0x100, &bytes, 3), "a write to the page");
        bytes[0x8000] = 0;
        bytes[PAGE as usize - 1] = 7;
        assert!(changed_since_sent(&mut sent, 0x100, &bytes, 4), "a write to its last byte");
        assert_eq!(sent[&0x100].1, 4, "compared in frame 4");
    }

    #[test]
    fn a_slot_taken_back_is_given_again_and_the_budget_holds() {
        let mut d = Directory::new(2);
        d.add(1).unwrap();
        d.add(2).unwrap();
        assert_eq!(d.remove(1), Some(Slot(0)));
        assert_eq!(d.get(1), None);
        assert_eq!(d.next(), Slot(0));
        assert_eq!(d.add(3), Ok((Slot(0), true)), "the freed slot");
        assert_eq!(d.add(4), Err(Refused::Budget));
        assert_eq!(d.remove(9), None);
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
        // Reserved memory: nothing by default (VICTORY_BVH_ARENA_RESERVED=1:
        // a window around the address, inside its allocation).
        if !reserved_on() {
            assert_eq!(committed_around(reserved + PAGE + 8), None);
        } else {
            assert_eq!(committed_around(reserved + PAGE + 8), Some((reserved, 2 * PAGE)));
        }
        // SAFETY: ours.
        unsafe {
            VirtualFree(reserved as *mut _, 0, MEM_RELEASE);
            VirtualFree(committed as *mut _, 0, MEM_RELEASE);
        }
        assert_eq!(memory_at(reserved), Memory::Free);
    }
}
