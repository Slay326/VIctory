//! The BVH builder capture, game side (local test branch, diagnostics
//! builds): `VICTORY_BVH_CAPTURE=<folder>` saves chosen submissions with
//! the builder's dispatches so that they run again on their own
//! (`super::harness`, format in `super::format`).
//!
//! * Every compute dispatch is recorded with what each root parameter got,
//!   by meaning ([`super::format::RootArg`]): constants (with the dwords
//!   that are heap indices or offsets of root views named), root views by
//!   title address, our own parameters (page table, page bitmap).
//! * `VICTORY_BVH_CAPTURE_AT` (`30,31,90,150`): which submissions with a
//!   builder dispatch are saved, counted from the first (two numbers in a
//!   row: two submissions in a row).
//! * Such a submission runs in two parts: the lists before the builder's
//!   first list (and that list's prelude) run and are waited for; then the
//!   pages are taken (after the refresh and the uploads into copies in the
//!   GPU's memory): every page the page table names in the 256 MiB blocks
//!   holding an arena page, a root view's address or an address in a
//!   builder's root constants, from the GPU's side, with hashes of title
//!   memory and of a carded copy's CPU side. Then the rest runs; after the
//!   fence (before anything goes back) the pages the lists' bitmaps marked
//!   are taken again (G), with the bitmaps.
//! * `VICTORY_BVH_CAPTURE_MB` (8192): the most page bytes a capture takes
//!   (the rest are left out and named in the manifest).
//!
//! * `VICTORY_BVH_CAPTURE_READER=<hash prefix>` (senior 115, the RT
//!   reader f607780a): the programs captured are that one's instead of the
//!   builder's (selection, cuts, code and root signature kept, "builder"
//!   in the file means "captured program"); their descriptor tables are
//!   recorded with their buffer views ([`RootArg::Table`]: host
//!   description, bytes before the program's first list and after the
//!   fence). `VICTORY_BVH_CAPTURE_FRAME` (0): submissions count from this
//!   presented frame on (an event such as the scene trigger), nothing is
//!   cut before it. The 256 MiB blocks of every table view's title address
//!   are taken too: f607780a reads its top level's address from t355's
//!   descriptor (the view-address table, disassembly 0x3ec-0x404 and
//!   0x650-0x688) and follows it through title memory.
//!
//! Captures hold game data: never commit them.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::format::*;
use crate::host::HostList;
use crate::pipeline::host::HostPipeline;
use crate::shadow::Shadow;

/// The folder captures go to, if capturing.
fn folder() -> Option<&'static PathBuf> {
    static FOLDER: OnceLock<Option<PathBuf>> = OnceLock::new();
    FOLDER
        .get_or_init(|| {
            std::env::var_os("VICTORY_BVH_CAPTURE")
                .map(PathBuf::from)
                .filter(|p| !p.as_os_str().is_empty())
        })
        .as_ref()
}

/// Whether captures are on.
pub(crate) fn enabled() -> bool {
    folder().is_some()
}

fn ordinals() -> &'static [u64] {
    static AT: OnceLock<Vec<u64>> = OnceLock::new();
    AT.get_or_init(|| {
        let v = std::env::var("VICTORY_BVH_CAPTURE_AT").unwrap_or_else(|_| "30,31,90,150".into());
        v.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    })
}

/// `VICTORY_BVH_CAPTURE_READER`: the program captured instead of the
/// builder (hash prefix, lower case).
pub(crate) fn reader_prefix() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| {
        std::env::var("VICTORY_BVH_CAPTURE_READER")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .filter(|v| !v.is_empty())
    })
    .as_deref()
}

/// `VICTORY_BVH_CAPTURE_FRAME`: the first presented frame counted.
fn from_frame() -> u64 {
    static F: OnceLock<u64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("VICTORY_BVH_CAPTURE_FRAME")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// The 256 MiB blocks (address >> 28) of the title addresses of table
/// views without a host view (the reader's t355 and t356: its top levels),
/// zeros left out: the replay walks title memory from there. Views with a
/// host view carry their own bytes ([`TableView::pre`]): their blocks are
/// not taken (a capture of all of them was 7.7 GB, run onCap).
pub fn view_blocks<'a>(views: impl IntoIterator<Item = &'a TableView>) -> BTreeSet<u64> {
    views
        .into_iter()
        .filter(|v| v.count == 0)
        .map(|v| v.title)
        .filter(|&a| a != 0)
        .map(|a| a >> 28)
        .collect()
}

/// Whether the program with shader hash `hash` is a captured one: the
/// reader's with `VICTORY_BVH_CAPTURE_READER`, else the builder's.
pub fn is_target(reader: Option<&str>, hash: &str, builder: bool) -> bool {
    match reader {
        Some(prefix) => hash.to_ascii_lowercase().starts_with(prefix),
        None => builder,
    }
}

/// [`is_target`] for a host pipeline.
pub(crate) fn target(p: &HostPipeline, builder: bool) -> bool {
    let hash = p.stages.first().map(|s| s.shader.hash.as_str()).unwrap_or("");
    is_target(reader_prefix(), hash, builder)
}

/// Whether binds of `p` record its tables ([`RootArg::Table`]).
pub(crate) fn wants_tables(p: &HostPipeline) -> bool {
    enabled() && reader_prefix().is_some() && target(p, false)
}

/// An indirect dispatch's arguments as a table of one
/// [`INDIRECT_ARGUMENTS`] view (parameter, index and slot `u32::MAX`),
/// their title address and bytes (stride times the most commands); none for
/// a direct dispatch.
pub fn indirect_arguments(groups: &Groups) -> Option<(RootArg, u64, u64)> {
    let Groups::Indirect { args: at, stride, most, .. } = groups else {
        return None;
    };
    let len = u64::from(*stride) * u64::from((*most).max(1));
    let view = TableView {
        kind: INDIRECT_ARGUMENTS,
        title: *at,
        address: *at,
        format: 39,
        count: (len / 4) as u32,
        raw: true,
        ..TableView::default()
    };
    let table = RootArg::Table { param: u32::MAX, index: u32::MAX, slot: u32::MAX, value: 0, views: vec![view] };
    Some((table, *at, len))
}

/// Where a table view's bytes are: the record's argument and view, the copy
/// and the host view's title range.
#[derive(Clone)]
pub(crate) struct ViewSource {
    pub(crate) arg: usize,
    pub(crate) view: usize,
    pub(crate) copy: Arc<Shadow>,
    pub(crate) address: u64,
    pub(crate) len: u64,
    pub(crate) unordered: bool,
}

fn most_bytes() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_CAPTURE_MB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(8192u64)
            << 20
    })
}

/// A dispatch as recorded.
#[derive(Clone)]
pub(crate) struct Recorded {
    pub(crate) pipeline: String,
    pub(crate) builder: bool,
    pub(crate) barrier: bool,
    pub(crate) groups: Groups,
    pub(crate) args: Vec<RootArg>,
    pub(crate) views: Vec<ViewSource>,
}

/// Host pipelines dispatches named, by name.
static PIPELINES: Mutex<BTreeMap<String, Pipeline>> = Mutex::new(BTreeMap::new());

/// With `VICTORY_BVH_CAPTURE_READER`: the captured program's shader
/// containers as the title passed them, by shader hash (written next to the
/// capture as `container-<hash>.dxbc`, for `--retranslate`).
static CONTAINERS: Mutex<BTreeMap<String, Vec<u8>>> = Mutex::new(BTreeMap::new());

/// A dispatch is about to be recorded into `list`.
pub(crate) fn record(
    list: &HostList,
    p: &HostPipeline,
    builder: bool,
    barrier: bool,
    groups: Groups,
    mut args: Vec<RootArg>,
    mut views: Vec<ViewSource>,
) {
    // With `VICTORY_BVH_CAPTURE_READER` the captured program stands where
    // the builder does.
    let builder = target(p, builder);
    // Its indirect arguments with their bytes (the page table may name
    // another copy for their page): a table of one [`INDIRECT_ARGUMENTS`]
    // view, bound nowhere.
    if builder
        && reader_prefix().is_some()
        && let Some((table, at, len)) = indirect_arguments(&groups)
    {
        if let Some(copy) = crate::shadow::find(at, len) {
            views.push(ViewSource { arg: args.len(), view: 0, copy, address: at, len, unordered: false });
        }
        args.push(table);
    }
    if builder && reader_prefix().is_some()
        && let Some(stage) = p.stages.first()
    {
        CONTAINERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(stage.shader.hash.clone())
            .or_insert_with(|| stage.shader.container().to_vec());
    }
    {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        // Only the builder's programs keep their code and root signature:
        // every compute program of a run would be hundreds of MB.
        if !pipelines
            .get(&p.name)
            .is_some_and(|known| !builder || !known.dxil.is_empty())
        {
            let stage = p.stages.first();
            let dxil = stage
                .filter(|_| builder)
                .and_then(|s| s.shader.translated())
                .map(|t| t.host.dxil.clone())
                .unwrap_or_default();
            let root = if builder {
                p.root_blob.clone()
            } else {
                Vec::new()
            };
            let hash = stage.map(|s| s.shader.hash.clone()).unwrap_or_default();
            pipelines.insert(
                p.name.clone(),
                Pipeline {
                    name: p.name.clone(),
                    hash,
                    dxil,
                    root,
                },
            );
        }
    }
    let barrier = barrier | list.take_bvh_barrier();
    list.push_bvh(Recorded {
        pipeline: p.name.clone(),
        builder,
        barrier,
        groups,
        args,
        views,
    });
}

/// Submissions with a builder dispatch so far.
static BUILDER_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

/// At a submission of `runs`: the index of the first run with a builder
/// dispatch if this submission is to be captured.
/// Whether the lists recorded now are cut around the builder
/// ([`super::cut_window`]).
pub(crate) fn cutting() -> bool {
    enabled()
        && (reader_prefix().is_none() || crate::queue::presented() + 1 >= from_frame())
        && super::cut_window(BUILDER_SUBMISSIONS.load(Ordering::Relaxed), ordinals())
}

pub(crate) fn due(runs: &[&HostList]) -> Option<usize> {
    if !enabled() {
        return None;
    }
    let first = runs
        .iter()
        .position(|l| l.bvh_records().iter().any(|r| r.builder))?;
    if reader_prefix().is_some() && crate::queue::presented() < from_frame() {
        return None;
    }
    let n = BUILDER_SUBMISSIONS.fetch_add(1, Ordering::Relaxed) + 1;
    ordinals().contains(&n).then_some(first)
}

fn hash8(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        blake3::hash(bytes).as_bytes()[..8]
            .try_into()
            .unwrap_or([0; 8]),
    )
}

/// A hash of title memory at `[at, at + len)`, 0 if it cannot be read.
fn title_hash(at: u64, len: usize) -> u64 {
    let mut bytes = vec![0u8; len];
    if crate::host::read_into(at as *const u8, &mut bytes) {
        hash8(&bytes)
    } else {
        0
    }
}

/// Where a captured page lives on the GPU now.
#[derive(Clone)]
enum Source {
    Copy(Arc<Shadow>),
    Arena,
}

/// A capture between its two parts.
pub(crate) struct Pending {
    dir: PathBuf,
    capture: Capture,
    first: usize,
    submission: u64,
    pre: std::io::BufWriter<std::fs::File>,
    pre_offset: u64,
    /// Segment of each copy (by its `Arc` pointer) and of each arena chunk.
    copy_segs: HashMap<usize, (u32, Arc<Shadow>)>,
    chunk_segs: HashMap<u32, u32>,
    /// Arena pages: title address -> (chunk, offset).
    arena_pages: HashMap<u64, (u32, u64)>,
    /// Captured pre pages (title address of the page) and their title hash.
    taken: BTreeMap<u64, u64>,
    /// Pages of tracked copies the title had written before the snapshot.
    changed_before: BTreeSet<u64>,
    left_out: u64,
    started: std::time::Instant,
    /// Table views' bytes before the captured program's first list: (run,
    /// record, argument, view) -> bytes.
    view_pre: HashMap<(usize, usize, usize, usize), Vec<u8>>,
}

fn meta(c: &mut Capture, k: &str, v: impl ToString) {
    c.meta.push((k.to_string(), v.to_string()));
}

/// Writes the pages of `source` at `pages` (title page addresses): GPU
/// bytes into `out` at `offset` (zero pages only noted), records into
/// `capture`.
#[allow(clippy::too_many_arguments)]
fn take(
    capture: &mut Capture,
    stage: Stage,
    source: &Source,
    seg: u32,
    seg_base: u64,
    pages: &[u64],
    arena_pages: &HashMap<u64, (u32, u64)>,
    out: &mut impl std::io::Write,
    offset: &mut u64,
) -> Vec<(u64, u64)> {
    let mut got: Vec<(u64, Vec<u8>, u64)> = Vec::new();
    match source {
        Source::Copy(copy) => {
            let wanted: Vec<usize> = pages.iter().map(|&p| p as usize).collect();
            for chunk in wanted.chunks(1024) {
                got.extend(
                    copy.capture_pages(chunk)
                        .into_iter()
                        .map(|(a, b, h)| (a as u64, b, h)),
                );
            }
        }
        Source::Arena => {
            for chunk in pages.chunks(2048) {
                got.extend(
                    crate::arena::fetch(chunk)
                        .into_iter()
                        .map(|(a, b)| (a, b, 0)),
                );
            }
        }
    }
    let mut hashes = Vec::new();
    for (at, bytes, cpu_hash) in got {
        let at_seg = match source {
            Source::Copy(_) => at - seg_base,
            Source::Arena => arena_pages
                .get(&(at & !(PAGE - 1)))
                .map_or(0, |&(_, off)| off + (at & (PAGE - 1))),
        };
        let zero = bytes.iter().all(|&b| b == 0);
        let th = title_hash(at, bytes.len());
        let file_offset = if zero {
            u64::MAX
        } else {
            let mut page = bytes.clone();
            page.resize(PAGE as usize, 0);
            if out.write_all(&page).is_err() {
                tracing::warn!("BVH capture: a page could not be written");
            }
            let o = *offset;
            *offset += PAGE;
            o
        };
        capture.pages.push(PageRec {
            stage,
            seg,
            at_seg,
            address: at,
            file_offset,
            hash: hash8(&bytes),
            title_hash: th,
            cpu_hash,
        });
        hashes.push((at, th));
    }
    hashes
}

/// The first part has run: takes the pre pages. None if nothing could be
/// taken (logged).
pub(crate) fn pre(runs: &[&HostList], first: usize, submission: u64) -> Option<Pending> {
    let started = std::time::Instant::now();
    let base = folder()?;
    let dir = base.join(format!("s{submission}"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %e, dir = %dir.display(), "BVH capture: no folder");
        return None;
    }
    let keys: BTreeSet<usize> = runs[first..].iter().flat_map(|l| l.memory_keys()).collect();
    let Some(&key) = keys.iter().next() else {
        tracing::warn!("BVH capture: the builder's lists read no page table");
        return None;
    };
    let mut capture = Capture::default();
    meta(&mut capture, "submission", submission);
    meta(&mut capture, "frame", crate::queue::presented());
    meta(
        &mut capture,
        "builder submission",
        BUILDER_SUBMISSIONS.load(Ordering::Relaxed),
    );
    meta(&mut capture, "page tables read", keys.len());
    meta(&mut capture, "translator", victory_shader::SOURCE_HASH);
    if let Some(prefix) = reader_prefix() {
        meta(&mut capture, "captured program", prefix);
        meta(&mut capture, "from frame", from_frame());
    }
    if let Some(host) = crate::host::host() {
        meta(&mut capture, "gpu", super::adapter_text(&host.device));
    }
    if let Ok(exe) = std::env::current_exe() {
        meta(
            &mut capture,
            "exe",
            exe.file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
    }
    let mut switches: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("VICTORY_"))
        .collect();
    switches.sort();
    for (k, v) in switches {
        // Paths are local: only that one is set.
        let v = if v.contains(':') || v.contains('\\') {
            "<path>".to_string()
        } else {
            v
        };
        meta(&mut capture, &k, v);
    }
    let layout = crate::memory::capture_layout(key);
    let arena_pages: HashMap<u64, (u32, u64)> = crate::arena::capture_pages()
        .into_iter()
        .map(|(at, c, o)| (at, (c, o)))
        .collect();
    // Segments: every copy the table names, every arena chunk with a page.
    let mut copy_segs: HashMap<usize, (u32, Arc<Shadow>)> = HashMap::new();
    for c in &layout.copies {
        let (start, len) = c.range();
        let seg = capture.segments.len() as u32;
        capture.segments.push(Segment::Copy {
            start,
            len,
            card: c.carded(),
        });
        copy_segs.insert(Arc::as_ptr(c) as usize, (seg, c.clone()));
    }
    let mut chunk_segs: HashMap<u32, u32> = HashMap::new();
    let chunks: BTreeSet<u32> = arena_pages.values().map(|&(c, _)| c).collect();
    for c in chunks {
        chunk_segs.insert(c, capture.segments.len() as u32);
        capture.segments.push(Segment::Arena {
            chunk: c,
            len: crate::arena::CHUNK_BYTES,
        });
    }
    // The table.
    let block = crate::memory::CAPTURE_BLOCK;
    capture.table.blocks = layout
        .blocks
        .iter()
        .map(|&b| if b == usize::MAX { 0 } else { b as u32 })
        .collect();
    capture.table.entries = vec![Entry::Null; layout.blocks.len() * block];
    for (i, e) in capture.table.entries.iter_mut().enumerate() {
        if i < block {
            continue;
        }
        if let Some(Some(k)) = layout.owner.get(i) {
            let ptr = Arc::as_ptr(&layout.copies[*k]) as usize;
            *e = Entry::Copy {
                seg: copy_segs[&ptr].0,
            };
        } else if layout.arena.get(i).copied().unwrap_or(false) {
            let at = u64::from(capture.table.blocks[i / block]) << 28 | ((i % block) as u64) << 16;
            if let Some(&(chunk, offset)) = arena_pages.get(&at) {
                *e = Entry::Arena {
                    seg: chunk_segs[&chunk],
                    offset: offset as u32,
                };
            }
        }
    }
    // The 256 MiB blocks to take: arena pages, root views and constant
    // addresses of the builder's dispatches, indirect arguments.
    let records: Vec<Recorded> = runs[first..].iter().flat_map(|l| l.bvh_records()).collect();
    let mut touched: BTreeSet<u64> = arena_pages.keys().map(|a| a >> 28).collect();
    let mut views: Vec<u64> = Vec::new();
    for r in &records {
        // `VICTORY_BVH_CAPTURE_READER`: only the captured program's
        // dispatches name blocks (the others of its lists are not replayed;
        // their root views and constants named most of 7.7 GB, onCap2).
        if reader_prefix().is_some() && !r.builder {
            continue;
        }
        for a in &r.args {
            match a {
                RootArg::View { address, .. } => views.push(*address),
                RootArg::Constants { dwords, .. } => {
                    for d in dwords {
                        if d.address != 0 {
                            views.push(d.address);
                        }
                    }
                    if r.builder {
                        let values: Vec<u32> = dwords.iter().map(|d| d.value).collect();
                        touched.extend(
                            crate::arena::constant_addresses(&values)
                                .into_iter()
                                .map(|a| a >> 28),
                        );
                    }
                }
                RootArg::Special { .. } | RootArg::Table { .. } => {}
            }
        }
        if let Groups::Indirect { args, count, .. } = &r.groups {
            touched.insert(args >> 28);
            if *count != 0 {
                touched.insert(count >> 28);
            }
        }
    }
    touched.extend(views.iter().map(|a| a >> 28));
    // Table views (`VICTORY_BVH_CAPTURE_READER`): their bytes now, their
    // blocks, and the block of the BVH descriptor's base address.
    let mut view_pre = HashMap::new();
    let mut titles = BTreeSet::new();
    // Blocks of views without a host view (the top levels): taken first.
    let mut priority: BTreeSet<u64> = BTreeSet::new();
    for (i, l) in runs.iter().enumerate().skip(first) {
        for (k, r) in l.bvh_records().iter().enumerate() {
            for a in &r.args {
                if let RootArg::Table { views, .. } = a {
                    touched.extend(view_blocks(views));
                    priority.extend(view_blocks(views));
                    titles.extend(views.iter().map(|v| (v.kind, v.register, v.title)));
                }
            }
            for v in &r.views {
                view_pre.insert((i, k, v.arg, v.view), super::reader::gpu_range(&v.copy, v.address, v.len));
            }
        }
    }
    if !titles.is_empty() {
        meta(
            &mut capture,
            "table view addresses",
            titles
                .iter()
                .map(|(k, r, t)| format!("{}{r}={t:#x}", ["b", "t", "u"].get(usize::from(*k)).unwrap_or(&"?")))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    // The first 256 MiB block too: in all eight captures of 2026-11-26 the
    // builder read 2-5 pages of it (0xb64...-0xb90... MiB) that no root
    // view or constant names, and the harness refuses such a capture.
    touched.insert(0);
    // Pages by source, then taken.
    let mut by_copy: BTreeMap<u32, BTreeSet<u64>> = BTreeMap::new();
    let mut arena_wanted: BTreeSet<u64> = BTreeSet::new();
    for (i, e) in capture.table.entries.iter().enumerate() {
        let Some(at) = capture.table.address_of(i) else {
            continue;
        };
        if !touched.contains(&(at >> 28)) {
            continue;
        }
        match *e {
            Entry::Copy { seg } => {
                by_copy.entry(seg).or_default().insert(at);
            }
            Entry::Arena { .. } => {
                arena_wanted.insert(at);
            }
            Entry::Null => {}
        }
    }
    // Root views: their copy for 1 MiB from the address, block or not.
    for &v in &views {
        if let Some((_, Entry::Copy { seg })) = capture.table.entry_at(v) {
            let set = by_copy.entry(seg).or_default();
            let mut p = v & !(PAGE - 1);
            while p < v + (1 << 20) {
                set.insert(p);
                p += PAGE;
            }
        }
    }
    let file = std::fs::File::create(dir.join("pre.pages")).ok()?;
    let mut pending = Pending {
        dir,
        capture,
        first,
        submission,
        pre: std::io::BufWriter::with_capacity(8 << 20, file),
        pre_offset: 0,
        copy_segs,
        chunk_segs,
        arena_pages,
        taken: BTreeMap::new(),
        changed_before: BTreeSet::new(),
        left_out: 0,
        started,
        view_pre,
    };
    let mut budget = most_bytes() / PAGE;
    let seg_of: HashMap<u32, Arc<Shadow>> = pending
        .copy_segs
        .values()
        .map(|(s, c)| (*s, c.clone()))
        .collect();
    // The top levels' blocks first (arena pages, then copies), then the
    // rest: the budget cuts the rest, not the structures the reader walks
    // from (onCap2: its two top-level pages were cut).
    let first_block = |a: &u64| priority.contains(&(a >> 28));
    let (arena_first, arena_rest): (Vec<u64>, Vec<u64>) = arena_wanted.into_iter().partition(first_block);
    let mut copies_first: Vec<(u32, Vec<u64>)> = Vec::new();
    let mut copies_rest: Vec<(u32, Vec<u64>)> = Vec::new();
    for (seg, pages) in by_copy {
        let (a, b): (Vec<u64>, Vec<u64>) = pages.into_iter().partition(first_block);
        if !a.is_empty() {
            copies_first.push((seg, a));
        }
        if !b.is_empty() {
            copies_rest.push((seg, b));
        }
    }
    meta(
        &mut pending.capture,
        "pages of the top levels' blocks",
        arena_first.len() + copies_first.iter().map(|(_, p)| p.len()).sum::<usize>(),
    );
    for (arena_list, copies) in [(arena_first, copies_first), (arena_rest, copies_rest)] {
    let n = (arena_list.len() as u64).min(budget) as usize;
    pending.left_out += (arena_list.len() - n) as u64;
    budget -= n as u64;
    let h = take(
        &mut pending.capture,
        Stage::Pre,
        &Source::Arena,
        0,
        0,
        &arena_list[..n],
        &pending.arena_pages,
        &mut pending.pre,
        &mut pending.pre_offset,
    );
    pending.taken.extend(h);
    for (seg, pages) in copies {
        let Some(copy) = seg_of.get(&seg) else {
            continue;
        };
        let n = (pages.len() as u64).min(budget) as usize;
        pending.left_out += (pages.len() - n) as u64;
        budget -= n as u64;
        for &p in &pages[..n] {
            if copy.title_changed(p as usize) == Some(true) {
                pending.changed_before.insert(p);
            }
        }
        let (start, _) = copy.range();
        let h = take(
            &mut pending.capture,
            Stage::Pre,
            &Source::Copy(copy.clone()),
            seg,
            start,
            &pages[..n],
            &pending.arena_pages,
            &mut pending.pre,
            &mut pending.pre_offset,
        );
        pending.taken.extend(h);
    }
    }
    // The page records name arena pages by their chunk's segment.
    for p in pending.capture.pages.iter_mut() {
        if let Some(&(chunk, _)) = pending.arena_pages.get(&(p.address & !(PAGE - 1))) {
            if !matches!(
                pending.capture.table.entry_at(p.address),
                Some((_, Entry::Copy { .. }))
            ) {
                p.seg = pending.chunk_segs[&chunk];
            }
        }
    }
    meta(
        &mut pending.capture,
        "blocks taken",
        touched
            .iter()
            .map(|l| format!("{:#x}", l << 28))
            .collect::<Vec<_>>()
            .join(" "),
    );
    meta(
        &mut pending.capture,
        "pages left out (budget)",
        pending.left_out,
    );
    tracing::info!(
        target: "victory_gpu::bvh_capture",
        "BVH capture s{submission}: pre {} pages ({} MB written), blocks {}, left out {}, {} ms",
        pending.taken.len(),
        pending.pre_offset >> 20,
        touched.len(),
        pending.left_out,
        started.elapsed().as_millis()
    );
    Some(pending)
}

/// The submission's fence passed, nothing went back yet: the pages the
/// lists marked, the bitmaps, and everything else; writes the capture.
pub(crate) fn post(mut p: Pending, runs: &[&HostList]) {
    let _ = p.pre.flush();
    let submission = p.submission;
    let mut post = match std::fs::File::create(p.dir.join("post.pages")) {
        Ok(f) => std::io::BufWriter::with_capacity(8 << 20, f),
        Err(e) => {
            tracing::warn!(error = %e, "BVH capture: no post file");
            return;
        }
    };
    let mut post_offset = 0;
    let block = crate::memory::CAPTURE_BLOCK;
    let mut written: BTreeSet<u64> = BTreeSet::new();
    let pipelines: BTreeMap<String, Pipeline> =
        PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (i, l) in runs.iter().enumerate().skip(p.first) {
        for (_, set) in l.bvh_bitmaps() {
            for &e in &set {
                if (e as usize) >= block
                    && let Some(at) = p.capture.table.address_of(e as usize)
                {
                    written.insert(at);
                }
            }
            p.capture.bitmaps.push((i as u32, set));
        }
        for (k, r) in l.bvh_records().iter().enumerate() {
            let index = pipelines.keys().position(|k| *k == r.pipeline).unwrap_or(0) as u32;
            let mut args = r.args.clone();
            // Table views: bytes before (taken then) and, unordered ones,
            // after the fence (now).
            for v in &r.views {
                if let Some(RootArg::Table { views, .. }) = args.get_mut(v.arg)
                    && let Some(tv) = views.get_mut(v.view)
                {
                    tv.pre = p.view_pre.remove(&(i, k, v.arg, v.view)).unwrap_or_default();
                    if v.unordered {
                        tv.post = super::reader::gpu_range(&v.copy, v.address, v.len);
                    }
                }
            }
            p.capture.dispatches.push(Dispatch {
                list: i as u32,
                pipeline: index,
                builder: r.builder,
                barrier: r.barrier,
                groups: r.groups.clone(),
                args,
            });
        }
        p.capture.refreshed.extend(l.refreshed_ranges());
        for t in l.note_texts() {
            if !(t.starts_with("dispatch") || t.starts_with("indirect kind 2")) {
                p.capture
                    .others
                    .push((i as u32, t.chars().take(160).collect()));
            }
        }
    }
    p.capture.pipelines = pipelines.into_values().collect();
    // G of the marked pages, by source.
    let mut by_copy: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    let mut arena: Vec<u64> = Vec::new();
    for &at in &written {
        match p.capture.table.entry_at(at) {
            Some((_, Entry::Copy { seg })) => by_copy.entry(seg).or_default().push(at),
            Some((_, Entry::Arena { .. })) => arena.push(at),
            _ => {}
        }
    }
    let first_post = p.capture.pages.len();
    let _ = take(
        &mut p.capture,
        Stage::Post,
        &Source::Arena,
        0,
        0,
        &arena,
        &p.arena_pages,
        &mut post,
        &mut post_offset,
    );
    let seg_of: HashMap<u32, Arc<Shadow>> =
        p.copy_segs.values().map(|(s, c)| (*s, c.clone())).collect();
    for (seg, pages) in &by_copy {
        if let Some(copy) = seg_of.get(seg) {
            let (start, _) = copy.range();
            let _ = take(
                &mut p.capture,
                Stage::Post,
                &Source::Copy(copy.clone()),
                *seg,
                start,
                pages,
                &p.arena_pages,
                &mut post,
                &mut post_offset,
            );
        }
    }
    for rec in p.capture.pages[first_post..].iter_mut() {
        if let Some(&(chunk, _)) = p.arena_pages.get(&(rec.address & !(PAGE - 1)))
            && !matches!(
                p.capture.table.entry_at(rec.address),
                Some((_, Entry::Copy { .. }))
            )
        {
            rec.seg = p.chunk_segs[&chunk];
        }
    }
    let _ = post.flush();
    // Title writes between the snapshot and the fence: tracked copies'
    // pages newly changed, and pages whose title bytes hash differently.
    let mut dirty = BTreeSet::new();
    for (&at, &th) in &p.taken {
        let page = at & !(PAGE - 1);
        if let Some((_, Entry::Copy { seg })) = p.capture.table.entry_at(at)
            && let Some(copy) = seg_of.get(&seg)
            && copy.title_changed(page as usize) == Some(true)
            && !p.changed_before.contains(&page)
        {
            dirty.insert(page);
            continue;
        }
        let len = p
            .capture
            .pages
            .iter()
            .find(|r| r.stage == Stage::Pre && r.address == at)
            .map_or(PAGE as usize, |r| {
                let seg_len = p
                    .capture
                    .segments
                    .get(r.seg as usize)
                    .map_or(PAGE, Segment::len);
                (seg_len.saturating_sub(r.at_seg)).min(PAGE) as usize
            });
        if th != 0 && title_hash(at, len) != th {
            dirty.insert(page);
        }
    }
    p.capture.cpu_dirty = dirty.into_iter().collect();
    let written_n = written.len();
    let missing: usize = written.iter().filter(|a| !p.taken.contains_key(a)).count();
    meta(&mut p.capture, "pages marked written", written_n);
    meta(&mut p.capture, "marked pages not taken before", missing);
    let mut bin = Vec::new();
    if p.capture.write(&mut bin).is_err()
        || std::fs::write(p.dir.join("capture.bin"), &bin).is_err()
    {
        tracing::warn!("BVH capture: capture.bin not written");
        return;
    }
    let _ = std::fs::write(p.dir.join("manifest.txt"), p.capture.manifest());
    for (hash, bytes) in CONTAINERS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let _ = std::fs::write(p.dir.join(format!("container-{hash}.dxbc")), bytes);
    }
    tracing::info!(
        target: "victory_gpu::bvh_capture",
        "BVH capture s{submission} written to {}: pre {} MB, post {} MB ({} pages marked, {} of them not taken before), dispatches {} (builder {}), title writes between snapshot and fence {}, other commands {}, {} ms",
        p.dir.display(),
        p.pre_offset >> 20,
        post_offset >> 20,
        written_n,
        missing,
        p.capture.dispatches.len(),
        p.capture.dispatches.iter().filter(|d| d.builder).count(),
        p.capture.cpu_dirty.len(),
        p.capture.others.len(),
        p.started.elapsed().as_millis()
    );
}

#[cfg(test)]
mod tests {
    use super::{indirect_arguments, is_target, view_blocks};
    use crate::bvh_replay::format::{Groups, RootArg, INDIRECT_ARGUMENTS};
    use crate::bvh_replay::format::TableView;

    /// senior 115: the reader switch makes the reader the captured program
    /// and only it; without it the builder stays.
    #[test]
    fn the_reader_switch_captures_the_reader_instead_of_the_builder() {
        assert!(is_target(Some("f607780a"), "F607780AA8E502D0", false));
        assert!(!is_target(Some("f607780a"), "5552dac6aaaa", true));
        assert!(is_target(None, "5552dac6aaaa", true));
        assert!(!is_target(None, "f607780aa8e502d0", false));
    }

    /// senior 115: every table view's title address and host range give
    /// their 256 MiB block (the reader's top level behind t355), zeros none.
    #[test]
    fn table_views_name_the_blocks_to_take() {
        let t355 = TableView { title: 0xd5f2_0cd00, ..TableView::default() };
        let t356 = TableView { title: 0x10_0274_0000, ..TableView::default() };
        let u52 = TableView { title: 0x12_0b1e_0000, address: 0x12_0b1e_0000, count: 163840, ..TableView::default() };
        let none = TableView::default();
        let blocks = view_blocks([&t355, &t356, &u52, &none]);
        assert_eq!(blocks.into_iter().collect::<Vec<_>>(), vec![0xd5, 0x100]);
    }

    /// senior 115: an indirect dispatch's arguments are recorded as a
    /// table of one argument view over stride x most bytes; a direct one
    /// has none.
    #[test]
    fn indirect_arguments_are_recorded_with_their_bytes() {
        let g = Groups::Indirect { stride: 12, arguments: vec![[2, 0, 0, 0]], args: 0xd_9358_fe60, count: 0, most: 2 };
        let (table, at, len) = indirect_arguments(&g).unwrap();
        assert_eq!((at, len), (0xd_9358_fe60, 24));
        let RootArg::Table { param, views, .. } = table else { panic!("not a table") };
        assert_eq!(param, u32::MAX);
        assert_eq!((views[0].kind, views[0].address, views[0].len()), (INDIRECT_ARGUMENTS, 0xd_9358_fe60, 24));
        assert!(indirect_arguments(&Groups::Direct([1, 1, 1])).is_none());
    }
}
