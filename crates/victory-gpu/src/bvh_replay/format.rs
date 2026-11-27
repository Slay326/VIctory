//! The file format of a BVH builder capture (`VICTORY_BVH_CAPTURE`, local
//! test branch): everything one submission with the builder's dispatches
//! needs to run again on its own (`super::harness`).
//!
//! A capture is a folder:
//! * `capture.bin`: [`Capture`] without page bytes (little endian, see
//!   [`Capture::write`]);
//! * `pre.pages`, `post.pages`: the non-zero 64 KiB pages before the
//!   submission ran and after its fence, before anything went back, each at
//!   the file offset its [`PageRec`] names;
//! * `manifest.txt`: the same in words, for people.
//!
//! Captures hold game data: they are never committed.

use std::io::{self, Read, Write};

/// The first bytes of `capture.bin`.
pub const MAGIC: &[u8; 8] = b"VBVHCAP1";
/// Format version; a reader refuses others.
pub const VERSION: u32 = 1;
/// A page of the page table: 64 KiB.
pub const PAGE: u64 = 1 << 16;
/// Entries per level-2 block of the page table (`crate::memory`).
pub const BLOCK: usize = 4096;
/// Level-1 entries of the page table.
pub const LEVEL1: usize = 4096;
/// Level-2 blocks a table has room for besides the null one.
pub const CAPACITY: usize = 512;
/// Bytes of a list's page bitmap buffer.
pub const BITMAP_BYTES: u64 = (((CAPACITY + 1) * BLOCK / 8) as u64).next_multiple_of(256);

/// A host pipeline a captured dispatch ran.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pipeline {
    /// Its name (`HostPipeline::name`, shader hashes).
    pub name: String,
    /// The compute stage's shader hash.
    pub hash: String,
    /// The DXIL container exactly as the host pipeline was made from it.
    pub dxil: Vec<u8>,
    /// The serialized root signature it was made with.
    pub root: Vec<u8>,
}

/// Where captured pages live: a host copy of a title buffer (laid out as
/// title memory, from `start`; `card`: in the GPU's own memory, else in
/// system memory the GPU reads) or a chunk of the BVH arena (laid out as
/// the arena's chunk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    Copy { start: u64, len: u64, card: bool },
    Arena { chunk: u32, len: u64 },
}

impl Segment {
    pub fn len(&self) -> u64 {
        match *self {
            Segment::Copy { len, .. } | Segment::Arena { len, .. } => len,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A page table entry: no copy (the null pair), a copy (its segment), or an
/// arena page (its chunk's segment and its byte offset in the chunk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Entry {
    #[default]
    Null,
    Copy {
        seg: u32,
    },
    Arena {
        seg: u32,
        offset: u32,
    },
}

/// A view heap's page table as the submission's lists read it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Table {
    /// The level-1 index of each block (block 0, the null one, first: its
    /// value is meaningless).
    pub blocks: Vec<u32>,
    /// `blocks.len() * BLOCK` entries.
    pub entries: Vec<Entry>,
}

impl Table {
    /// The entry of the page at title address `at`, if its block is there.
    pub fn entry_at(&self, at: u64) -> Option<(usize, Entry)> {
        let l1 = (at >> 28) as u32;
        let b = self.blocks.iter().skip(1).position(|&x| x == l1)? + 1;
        let i = b * BLOCK + ((at >> 16) & 0xFFF) as usize;
        Some((i, *self.entries.get(i)?))
    }
    /// The title address of entry `i` (none in block 0).
    pub fn address_of(&self, i: usize) -> Option<u64> {
        let b = i / BLOCK;
        if b == 0 {
            return None;
        }
        Some(u64::from(*self.blocks.get(b)?) << 28 | ((i % BLOCK) as u64) << 16)
    }
}

/// When a page was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Before the builder's first list ran, after the refresh and the
    /// uploads into copies in the GPU's memory.
    Pre = 0,
    /// After the submission's fence, before anything went back (G).
    Post = 1,
}

/// A captured page: its segment and byte offset there, the title address it
/// stands for, its bytes (`None`: all zero) at `offset` of its stage's
/// pages file, and hashes of the CPU side (copy's CPU side or title memory)
/// to find inputs older than what the GPU read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRec {
    pub stage: Stage,
    pub seg: u32,
    pub at_seg: u64,
    pub address: u64,
    /// Offset in `pre.pages` / `post.pages`; `u64::MAX`: a zero page.
    pub file_offset: u64,
    /// blake3 (first 8 bytes) of the GPU bytes as captured.
    pub hash: u64,
    /// The same of title memory at the page, if readable (0: not).
    pub title_hash: u64,
    /// The same of a copy's CPU side, for copies in the GPU's memory (0:
    /// none).
    pub cpu_hash: u64,
}

impl PageRec {
    pub fn is_zero(&self) -> bool {
        self.file_offset == u64::MAX
    }
}

/// How a root constant dword got its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DwordKind {
    /// A value as the title set it (or a constant of ours).
    #[default]
    Plain = 0,
    /// The heap index of a raw SRV of the copy holding `address`.
    RootViewSrv = 1,
    /// The heap index of a raw UAV of the copy holding `address`.
    RootViewUav = 2,
    /// `address`'s byte offset in its copy.
    RootViewOffset = 3,
    /// A descriptor table's byte offset in its heap (not replayable).
    TableOffset = 4,
    /// The split dispatch's group offset (0 for a direct dispatch).
    GroupXOffset = 5,
}

/// A root constant dword.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dword {
    pub kind: DwordKind,
    pub value: u32,
    pub address: u64,
}

/// What a root parameter held.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootArg {
    Constants {
        param: u32,
        dwords: Vec<Dword>,
    },
    /// A root view at a title address: 0 CBV, 1 SRV, 2 UAV.
    View {
        param: u32,
        kind: u8,
        address: u64,
    },
    /// Something of ours ([`Special`]).
    Special {
        param: u32,
        what: Special,
    },
}

/// Root parameters our shaders read that are not the title's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    MemoryTable = 0,
    MemoryDirty = 1,
    MarksTable = 2,
    MarksBitmap = 3,
    ViewStrides = 4,
    ViewAddresses = 5,
}

/// How many groups a dispatch ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Groups {
    Direct([u32; 3]),
    /// An indirect dispatch: the command signature's stride and arguments
    /// (type and three fields each), the argument and count addresses
    /// (title memory) and the most commands.
    Indirect {
        stride: u32,
        arguments: Vec<[u32; 4]>,
        args: u64,
        count: u64,
        most: u32,
    },
}

/// A dispatch of a captured list, in recorded order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    /// The run (host list) it was recorded in, in submission order.
    pub list: u32,
    /// Index in [`Capture::pipelines`].
    pub pipeline: u32,
    /// One of the builder's programs.
    pub builder: bool,
    /// A global UAV barrier came before it (`VICTORY_BVH_UAV_BARRIER`).
    pub barrier: bool,
    pub groups: Groups,
    pub args: Vec<RootArg>,
}

/// A captured submission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Capture {
    /// Key, value (submission, frame, GPU, driver, build, switches, ...).
    pub meta: Vec<(String, String)>,
    pub pipelines: Vec<Pipeline>,
    pub segments: Vec<Segment>,
    pub table: Table,
    pub pages: Vec<PageRec>,
    pub dispatches: Vec<Dispatch>,
    /// Per run with title memory: the page bitmap's set bits after the game
    /// ran it (entries as the table numbers them).
    pub bitmaps: Vec<(u32, Vec<u32>)>,
    /// Ranges (address, bytes) the runs' refresh brought up to date.
    pub refreshed: Vec<(u64, u64)>,
    /// Captured pages the title wrote between the pre snapshot and the
    /// fence (title addresses).
    pub cpu_dirty: Vec<u64>,
    /// Commands in the captured runs that are no dispatch (by text).
    pub others: Vec<(u32, String)>,
}

struct W<'a, T: Write>(&'a mut T);

impl<T: Write> W<'_, T> {
    fn u8(&mut self, v: u8) -> io::Result<()> {
        self.0.write_all(&[v])
    }
    fn u32(&mut self, v: u32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn bytes(&mut self, v: &[u8]) -> io::Result<()> {
        self.u64(v.len() as u64)?;
        self.0.write_all(v)
    }
    fn str(&mut self, v: &str) -> io::Result<()> {
        self.bytes(v.as_bytes())
    }
    fn len(&mut self, n: usize) -> io::Result<()> {
        self.u64(n as u64)
    }
}

struct R<'a, T: Read>(&'a mut T);

fn bad(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("BVH capture: {what}"))
}

impl<T: Read> R<'_, T> {
    fn u8(&mut self) -> io::Result<u8> {
        let mut b = [0; 1];
        self.0.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        let mut b = [0; 4];
        self.0.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> io::Result<u64> {
        let mut b = [0; 8];
        self.0.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn len(&mut self, most: u64) -> io::Result<usize> {
        let n = self.u64()?;
        if n > most {
            return Err(bad("a length past its bound"));
        }
        Ok(n as usize)
    }
    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let n = self.len(1 << 30)?;
        let mut v = vec![0; n];
        self.0.read_exact(&mut v)?;
        Ok(v)
    }
    fn str(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?).map_err(|_| bad("text not UTF-8"))
    }
}

impl Capture {
    /// Writes `capture.bin`'s bytes.
    pub fn write(&self, out: &mut impl Write) -> io::Result<()> {
        let mut w = W(out);
        w.0.write_all(MAGIC)?;
        w.u32(VERSION)?;
        w.len(self.meta.len())?;
        for (k, v) in &self.meta {
            w.str(k)?;
            w.str(v)?;
        }
        w.len(self.pipelines.len())?;
        for p in &self.pipelines {
            w.str(&p.name)?;
            w.str(&p.hash)?;
            w.bytes(&p.dxil)?;
            w.bytes(&p.root)?;
        }
        w.len(self.segments.len())?;
        for s in &self.segments {
            match *s {
                Segment::Copy { start, len, card } => {
                    w.u8(if card { 2 } else { 0 })?;
                    w.u64(start)?;
                    w.u64(len)?;
                }
                Segment::Arena { chunk, len } => {
                    w.u8(1)?;
                    w.u64(u64::from(chunk))?;
                    w.u64(len)?;
                }
            }
        }
        w.len(self.table.blocks.len())?;
        for &b in &self.table.blocks {
            w.u32(b)?;
        }
        // Entries run-length coded: most are null.
        let mut i = 0;
        let e = &self.table.entries;
        w.len(e.len())?;
        while i < e.len() {
            let mut n = 1;
            while i + n < e.len() && e[i + n] == e[i] && n < u32::MAX as usize {
                n += 1;
            }
            w.u32(n as u32)?;
            match e[i] {
                Entry::Null => w.u8(0)?,
                Entry::Copy { seg } => {
                    w.u8(1)?;
                    w.u32(seg)?;
                }
                Entry::Arena { seg, offset } => {
                    w.u8(2)?;
                    w.u32(seg)?;
                    w.u32(offset)?;
                }
            }
            i += n;
        }
        w.len(self.pages.len())?;
        for p in &self.pages {
            w.u8(p.stage as u8)?;
            w.u32(p.seg)?;
            w.u64(p.at_seg)?;
            w.u64(p.address)?;
            w.u64(p.file_offset)?;
            w.u64(p.hash)?;
            w.u64(p.title_hash)?;
            w.u64(p.cpu_hash)?;
        }
        w.len(self.dispatches.len())?;
        for d in &self.dispatches {
            w.u32(d.list)?;
            w.u32(d.pipeline)?;
            w.u8(u8::from(d.builder) | u8::from(d.barrier) << 1)?;
            match &d.groups {
                Groups::Direct(g) => {
                    w.u8(0)?;
                    for &x in g {
                        w.u32(x)?;
                    }
                }
                Groups::Indirect {
                    stride,
                    arguments,
                    args,
                    count,
                    most,
                } => {
                    w.u8(1)?;
                    w.u32(*stride)?;
                    w.len(arguments.len())?;
                    for a in arguments {
                        for &x in a {
                            w.u32(x)?;
                        }
                    }
                    w.u64(*args)?;
                    w.u64(*count)?;
                    w.u32(*most)?;
                }
            }
            w.len(d.args.len())?;
            for a in &d.args {
                match a {
                    RootArg::Constants { param, dwords } => {
                        w.u8(0)?;
                        w.u32(*param)?;
                        w.len(dwords.len())?;
                        for x in dwords {
                            w.u8(x.kind as u8)?;
                            w.u32(x.value)?;
                            w.u64(x.address)?;
                        }
                    }
                    RootArg::View {
                        param,
                        kind,
                        address,
                    } => {
                        w.u8(1)?;
                        w.u32(*param)?;
                        w.u8(*kind)?;
                        w.u64(*address)?;
                    }
                    RootArg::Special { param, what } => {
                        w.u8(2)?;
                        w.u32(*param)?;
                        w.u8(*what as u8)?;
                    }
                }
            }
        }
        w.len(self.bitmaps.len())?;
        for (list, set) in &self.bitmaps {
            w.u32(*list)?;
            w.len(set.len())?;
            for &x in set {
                w.u32(x)?;
            }
        }
        w.len(self.refreshed.len())?;
        for &(a, n) in &self.refreshed {
            w.u64(a)?;
            w.u64(n)?;
        }
        w.len(self.cpu_dirty.len())?;
        for &a in &self.cpu_dirty {
            w.u64(a)?;
        }
        w.len(self.others.len())?;
        for (list, text) in &self.others {
            w.u32(*list)?;
            w.str(text)?;
        }
        Ok(())
    }

    /// Reads what [`Capture::write`] wrote.
    pub fn read(input: &mut impl Read) -> io::Result<Capture> {
        let mut r = R(input);
        let mut magic = [0; 8];
        r.0.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not a capture (magic)"));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(bad(&format!(
                "version {version}, this reader knows {VERSION}"
            )));
        }
        let mut c = Capture::default();
        for _ in 0..r.len(1 << 16)? {
            let k = r.str()?;
            c.meta.push((k, r.str()?));
        }
        for _ in 0..r.len(1 << 16)? {
            c.pipelines.push(Pipeline {
                name: r.str()?,
                hash: r.str()?,
                dxil: r.bytes()?,
                root: r.bytes()?,
            });
        }
        for _ in 0..r.len(1 << 24)? {
            let s = match r.u8()? {
                k @ (0 | 2) => Segment::Copy {
                    start: r.u64()?,
                    len: r.u64()?,
                    card: k == 2,
                },
                1 => Segment::Arena {
                    chunk: r.u64()? as u32,
                    len: r.u64()?,
                },
                _ => return Err(bad("segment kind")),
            };
            c.segments.push(s);
        }
        for _ in 0..r.len((CAPACITY + 1) as u64)? {
            c.table.blocks.push(r.u32()?);
        }
        let n = r.len(((CAPACITY + 1) * BLOCK) as u64)?;
        while c.table.entries.len() < n {
            let run = r.u32()? as usize;
            let e = match r.u8()? {
                0 => Entry::Null,
                1 => Entry::Copy { seg: r.u32()? },
                2 => Entry::Arena {
                    seg: r.u32()?,
                    offset: r.u32()?,
                },
                _ => return Err(bad("entry kind")),
            };
            if run == 0 || c.table.entries.len() + run > n {
                return Err(bad("entry run"));
            }
            c.table.entries.extend(std::iter::repeat_n(e, run));
        }
        for _ in 0..r.len(1 << 26)? {
            let stage = match r.u8()? {
                0 => Stage::Pre,
                1 => Stage::Post,
                _ => return Err(bad("page stage")),
            };
            c.pages.push(PageRec {
                stage,
                seg: r.u32()?,
                at_seg: r.u64()?,
                address: r.u64()?,
                file_offset: r.u64()?,
                hash: r.u64()?,
                title_hash: r.u64()?,
                cpu_hash: r.u64()?,
            });
        }
        for _ in 0..r.len(1 << 24)? {
            let list = r.u32()?;
            let pipeline = r.u32()?;
            let flags = r.u8()?;
            let groups = match r.u8()? {
                0 => Groups::Direct([r.u32()?, r.u32()?, r.u32()?]),
                1 => {
                    let stride = r.u32()?;
                    let mut arguments = Vec::new();
                    for _ in 0..r.len(64)? {
                        arguments.push([r.u32()?, r.u32()?, r.u32()?, r.u32()?]);
                    }
                    Groups::Indirect {
                        stride,
                        arguments,
                        args: r.u64()?,
                        count: r.u64()?,
                        most: r.u32()?,
                    }
                }
                _ => return Err(bad("groups kind")),
            };
            let mut args = Vec::new();
            for _ in 0..r.len(256)? {
                args.push(match r.u8()? {
                    0 => {
                        let param = r.u32()?;
                        let mut dwords = Vec::new();
                        for _ in 0..r.len(256)? {
                            let kind = match r.u8()? {
                                0 => DwordKind::Plain,
                                1 => DwordKind::RootViewSrv,
                                2 => DwordKind::RootViewUav,
                                3 => DwordKind::RootViewOffset,
                                4 => DwordKind::TableOffset,
                                5 => DwordKind::GroupXOffset,
                                _ => return Err(bad("dword kind")),
                            };
                            dwords.push(Dword {
                                kind,
                                value: r.u32()?,
                                address: r.u64()?,
                            });
                        }
                        RootArg::Constants { param, dwords }
                    }
                    1 => RootArg::View {
                        param: r.u32()?,
                        kind: r.u8()?,
                        address: r.u64()?,
                    },
                    2 => {
                        let param = r.u32()?;
                        let what = match r.u8()? {
                            0 => Special::MemoryTable,
                            1 => Special::MemoryDirty,
                            2 => Special::MarksTable,
                            3 => Special::MarksBitmap,
                            4 => Special::ViewStrides,
                            5 => Special::ViewAddresses,
                            _ => return Err(bad("special kind")),
                        };
                        RootArg::Special { param, what }
                    }
                    _ => return Err(bad("root argument kind")),
                });
            }
            c.dispatches.push(Dispatch {
                list,
                pipeline,
                builder: flags & 1 != 0,
                barrier: flags & 2 != 0,
                groups,
                args,
            });
        }
        for _ in 0..r.len(1 << 16)? {
            let list = r.u32()?;
            let mut set = Vec::new();
            for _ in 0..r.len(((CAPACITY + 1) * BLOCK) as u64)? {
                set.push(r.u32()?);
            }
            c.bitmaps.push((list, set));
        }
        for _ in 0..r.len(1 << 26)? {
            c.refreshed.push((r.u64()?, r.u64()?));
        }
        for _ in 0..r.len(1 << 26)? {
            c.cpu_dirty.push(r.u64()?);
        }
        for _ in 0..r.len(1 << 20)? {
            let list = r.u32()?;
            c.others.push((list, r.str()?));
        }
        Ok(c)
    }

    /// The value of meta key `key`.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The pages of `stage`.
    pub fn pages_of(&self, stage: Stage) -> impl Iterator<Item = &PageRec> {
        self.pages.iter().filter(move |p| p.stage == stage)
    }

    /// Pages the capture's budget left out; `None` when the capture does
    /// not say (counts as incomplete).
    pub fn left_out(&self) -> Option<u64> {
        self.meta("pages left out (budget)")?.trim().parse().ok()
    }

    /// Pages the lists marked that have a source (a copy or an arena page:
    /// G holds them) but were not taken before: the harness would run them
    /// on poison or the null pair where the game had bytes.
    pub fn marked_not_taken(&self) -> usize {
        self.marked_not_taken_pages().len()
    }

    /// The pages [`Capture::marked_not_taken`] counts.
    pub fn marked_not_taken_pages(&self) -> Vec<u64> {
        let pre: std::collections::BTreeSet<u64> = self
            .pages_of(Stage::Pre)
            .map(|p| p.address & !(PAGE - 1))
            .collect();
        let post: std::collections::BTreeSet<u64> = self
            .pages_of(Stage::Post)
            .map(|p| p.address & !(PAGE - 1))
            .collect();
        post.difference(&pre).copied().collect()
    }
}

/// Why a capture cannot stand for its submission; empty: complete. A
/// capture with pages left out, marked pages with a source not taken
/// before, or harness accesses outside the capture is refused: "valid in
/// the harness" from such a capture would be a wrong conclusion.
/// The BVH builder's stage that rehydrates packed bottom levels into
/// triangles (shader hash prefix); its root constant dword 5 is the y scale.
pub const REHYDRATE_HASH: &str = "5e687674";

/// Where [`REHYDRATE_HASH`]'s y scale (`cb5`) lies in a captured
/// [`RootArg::Constants`]: the dwords are indexed by user-data index, and its
/// constants start at user data 16 (16 + 5).
pub const REHYDRATE_Y_SCALE: usize = 21;

pub fn incomplete(left_out: Option<u64>, marked_not_taken: usize, outside: usize) -> Vec<String> {
    let mut why = Vec::new();
    match left_out {
        None => why.push("pages left out by the budget unknown".to_string()),
        Some(0) => {}
        Some(n) => why.push(format!("{n} pages left out by the budget")),
    }
    if marked_not_taken > 0 {
        why.push(format!("{marked_not_taken} marked pages with a source not taken before"));
    }
    if outside > 0 {
        why.push(format!("{outside} accesses outside the capture"));
    }
    why
}

/// Marks an [`outside`] key whose entry lies in the null block: the 256 MiB
/// region had no block, so only the page's offset in its region is known
/// (`NULL_BLOCK | offset`), not the region.
pub const NULL_BLOCK: u64 = 1 << 63;

/// The accesses outside the capture of one harness run: entries its page
/// bitmap `bits` set that the game's `game` did not, whose page was not
/// captured before the run. Title addresses, or [`NULL_BLOCK`] keys.
pub fn outside(
    c: &Capture,
    bits: &std::collections::BTreeSet<u32>,
    game: &std::collections::BTreeSet<u32>,
) -> std::collections::BTreeSet<u64> {
    let pre: std::collections::BTreeSet<u64> = c
        .pages_of(Stage::Pre)
        .map(|p| p.address & !(PAGE - 1))
        .collect();
    bits.difference(game)
        .filter_map(|&e| {
            if (e as usize) < BLOCK {
                return Some(NULL_BLOCK | u64::from(e) << 16);
            }
            c.table.address_of(e as usize).filter(|a| !pre.contains(a))
        })
        .collect()
}

/// An [`outside`] key as text: an address, or `null block +offset`.
pub fn outside_text(k: u64) -> String {
    if k & NULL_BLOCK != 0 {
        format!("null block +{:#x}", k & !NULL_BLOCK)
    } else {
        format!("{k:#x}")
    }
}

impl Capture {
    /// [`incomplete`] for what the capture alone says (before a run).
    pub fn incomplete_before_run(&self) -> Vec<String> {
        let mut why = incomplete(self.left_out(), self.marked_not_taken(), 0);
        why.extend(self.rehydrate_scale_zeroed());
        why
    }

    /// Fix B acceptance (b): the rehydrating stage [`REHYDRATE_HASH`] reads
    /// its y scale from root constant dword 5 (`cb5`). When the capture has
    /// dispatches of it and none carries cb5 != 0, the host zeroed the
    /// title's value (as before Fix B) and the capture cannot show the y
    /// scale: a reason it is invalid. Without such dispatches: no reason.
    pub fn rehydrate_scale_zeroed(&self) -> Option<String> {
        let mut seen = 0usize;
        for d in &self.dispatches {
            let Some(p) = self.pipelines.get(d.pipeline as usize) else { continue };
            if !p.hash.to_ascii_lowercase().starts_with(REHYDRATE_HASH) {
                continue;
            }
            seen += 1;
            let nonzero = d.args.iter().any(|a| {
                matches!(a, RootArg::Constants { dwords, .. }
                    if dwords.get(REHYDRATE_Y_SCALE).is_some_and(|x| x.value != 0))
            });
            if nonzero {
                return None;
            }
        }
        (seen > 0).then(|| {
            format!("{seen} dispatches of {REHYDRATE_HASH}, none with root constant dword 5 (y scale) != 0")
        })
    }

    /// A summary for `manifest.txt`.
    pub fn manifest(&self) -> String {
        use std::fmt::Write as _;
        let mut t = String::new();
        let _ = writeln!(
            t,
            "BVH builder capture (format {VERSION}); game data, never commit"
        );
        for (k, v) in &self.meta {
            let _ = writeln!(t, "{k}: {v}");
        }
        let pre = self.pages_of(Stage::Pre).count();
        let pre_zero = self.pages_of(Stage::Pre).filter(|p| p.is_zero()).count();
        let post = self.pages_of(Stage::Post).count();
        let _ = writeln!(
            t,
            "pipelines {}, dispatches {} (builder {}), segments {} (arena chunks {}), table blocks {}",
            self.pipelines.len(),
            self.dispatches.len(),
            self.dispatches.iter().filter(|d| d.builder).count(),
            self.segments.len(),
            self.segments
                .iter()
                .filter(|s| matches!(s, Segment::Arena { .. }))
                .count(),
            self.table.blocks.len().saturating_sub(1)
        );
        let _ = writeln!(
            t,
            "pages pre {pre} (zero {pre_zero}), post {post}; refreshed ranges {}, title writes between snapshot and fence {}, other commands {}",
            self.refreshed.len(),
            self.cpu_dirty.len(),
            self.others.len()
        );
        for (list, set) in &self.bitmaps {
            let _ = writeln!(
                t,
                "run {list}: page bitmap bits set after the game ran it {}",
                set.len()
            );
        }
        for (i, d) in self.dispatches.iter().enumerate() {
            let p = self
                .pipelines
                .get(d.pipeline as usize)
                .map_or("?", |p| p.hash.as_str());
            let _ = writeln!(
                t,
                "dispatch {i}: run {} {p} builder {} {:?}",
                d.list, d.builder, d.groups
            );
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Capture {
        let mut table = Table {
            blocks: vec![0, 0xdd],
            entries: vec![Entry::Null; 2 * BLOCK],
        };
        table.entries[BLOCK + 3] = Entry::Copy { seg: 0 };
        table.entries[BLOCK + 4] = Entry::Arena {
            seg: 1,
            offset: 0x30000,
        };
        Capture {
            meta: vec![("submission".into(), "5427".into())],
            pipelines: vec![Pipeline {
                name: "cs 5552dac6".into(),
                hash: "5552dac6".into(),
                dxil: vec![1, 2, 3],
                root: vec![9],
            }],
            segments: vec![
                Segment::Copy {
                    start: 0xdd0030000,
                    len: 0x10000,
                    card: true,
                },
                Segment::Arena {
                    chunk: 2,
                    len: 64 << 20,
                },
            ],
            table,
            pages: vec![
                PageRec {
                    stage: Stage::Pre,
                    seg: 0,
                    at_seg: 0,
                    address: 0xdd0030000,
                    file_offset: 0,
                    hash: 7,
                    title_hash: 8,
                    cpu_hash: 0,
                },
                PageRec {
                    stage: Stage::Post,
                    seg: 1,
                    at_seg: 0x30000,
                    address: 0xdd0040000,
                    file_offset: u64::MAX,
                    hash: 1,
                    title_hash: 0,
                    cpu_hash: 0,
                },
            ],
            dispatches: vec![Dispatch {
                list: 0,
                pipeline: 0,
                builder: true,
                barrier: false,
                groups: Groups::Indirect {
                    stride: 12,
                    arguments: vec![[2, 0, 0, 0]],
                    args: 0xdd0030010,
                    count: 0,
                    most: 1,
                },
                args: vec![
                    RootArg::Constants {
                        param: 0,
                        dwords: vec![
                            Dword {
                                kind: DwordKind::RootViewUav,
                                value: 17,
                                address: 0xdd0030000,
                            },
                            Dword::default(),
                        ],
                    },
                    RootArg::View {
                        param: 1,
                        kind: 1,
                        address: 0xdd0030040,
                    },
                    RootArg::Special {
                        param: 2,
                        what: Special::MemoryTable,
                    },
                ],
            }],
            bitmaps: vec![(0, vec![BLOCK as u32 + 3])],
            refreshed: vec![(0xdd0030000, 0x10000)],
            cpu_dirty: vec![0xdd0030000],
            others: vec![(0, "copy".into())],
        }
    }

    #[test]
    fn a_capture_reads_back_as_written() {
        let c = sample();
        let mut bytes = Vec::new();
        c.write(&mut bytes).unwrap();
        let back = Capture::read(&mut &bytes[..]).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn a_truncated_or_foreign_capture_is_refused() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        assert!(Capture::read(&mut &bytes[..bytes.len() - 3]).is_err());
        bytes[0] = b'X';
        assert!(Capture::read(&mut &bytes[..]).is_err());
    }

    #[test]
    fn table_entries_name_their_title_addresses() {
        let c = sample();
        assert_eq!(
            c.table.entry_at(0xdd0030000),
            Some((BLOCK + 3, Entry::Copy { seg: 0 }))
        );
        assert_eq!(c.table.address_of(BLOCK + 4), Some(0xdd0040000));
        assert_eq!(c.table.entry_at(0xde0000000), None);
        assert_eq!(c.table.address_of(5), None);
    }

    #[test]
    fn a_marked_page_with_a_source_not_taken_before_is_found() {
        let mut c = sample();
        // The arena page 0xdd0040000 is in G but not in the pre pages.
        assert_eq!(c.marked_not_taken(), 1);
        let mut pre = c.pages[1].clone();
        pre.stage = Stage::Pre;
        c.pages.push(pre);
        assert_eq!(c.marked_not_taken(), 0);
    }

    #[test]
    fn the_budget_count_comes_from_the_meta_and_unknown_is_incomplete() {
        let mut c = sample();
        assert_eq!(c.left_out(), None);
        assert!(!c.incomplete_before_run().is_empty());
        c.meta.push(("pages left out (budget)".into(), "0".into()));
        assert_eq!(c.left_out(), Some(0));
        c.meta.last_mut().unwrap().1 = "29000".into();
        assert_eq!(c.left_out(), Some(29000));
    }

    #[test]
    fn an_incomplete_capture_is_refused_for_each_reason() {
        assert!(incomplete(Some(0), 0, 0).is_empty());
        assert_eq!(incomplete(Some(5), 0, 0).len(), 1);
        assert_eq!(incomplete(None, 0, 0).len(), 1);
        assert_eq!(incomplete(Some(0), 3, 0).len(), 1);
        assert_eq!(incomplete(Some(0), 0, 121).len(), 1);
        assert_eq!(incomplete(Some(1), 1, 1).len(), 3);
    }

    /// A2 step 1 (determinism of the accesses outside the capture): only
    /// entries the harness set beyond the game's, whose page was not taken
    /// before, count; one in the null block keeps its offset, not a region.
    #[test]
    fn accesses_outside_the_capture_are_the_harness_only_pages_not_taken() {
        let c = sample();
        let set = |v: &[u32]| v.iter().copied().collect::<std::collections::BTreeSet<u32>>();
        let b = BLOCK as u32;
        // 0x110: a page in a region without a block; b+3 captured (pre);
        // b+4 not captured (an arena page only in G); b+5 not captured but
        // the game set it too; b+6 not captured and the harness only.
        let harness = set(&[0x110, b + 3, b + 4, b + 5, b + 6]);
        let game = set(&[b + 5]);
        let got = outside(&c, &harness, &game);
        let want: std::collections::BTreeSet<u64> =
            [NULL_BLOCK | 0x110_0000, 0xdd0040000, 0xdd0060000].into();
        assert_eq!(got, want);
        assert_eq!(outside_text(NULL_BLOCK | 0x110_0000), "null block +0x1100000");
        assert_eq!(outside_text(0xdd0040000), "0xdd0040000");
        // Same bits as the game: nothing outside.
        assert!(outside(&c, &game, &game).is_empty());
    }

    /// Fix B acceptance (b): a capture whose 5e687674 dispatches all carry
    /// cb5 = 0 (the host zeroed the title's y scale) is refused; one dispatch
    /// with cb5 != 0 passes; a capture without the stage is not affected.
    #[test]
    fn a_capture_with_the_rehydrating_y_scale_zeroed_is_refused() {
        let mut c = sample();
        c.meta.push(("pages left out (budget)".into(), "0".into()));
        let mut pre = c.pages[1].clone();
        pre.stage = Stage::Pre;
        c.pages.push(pre);
        assert!(c.incomplete_before_run().is_empty(), "no 5e687674: no reason");
        c.pipelines.push(Pipeline {
            name: "cs 5e68767498c57eb6".into(),
            hash: "5e68767498c57eb6".into(),
            dxil: vec![4],
            root: vec![9],
        });
        let cb = |y: u32| RootArg::Constants {
            param: 0,
            dwords: (0..35)
                .map(|i| Dword {
                    // As the old build recorded it: dword 5 (y scale) as the group offset.
                    kind: if i == REHYDRATE_Y_SCALE { DwordKind::GroupXOffset } else { DwordKind::Plain },
                    // Index 5 nonzero too: only the y scale counts.
                    value: match i { 5 | 20 | 22 => 0x4900_0000, REHYDRATE_Y_SCALE => y, _ => 0 },
                    address: 0,
                })
                .collect(),
        };
        let rehydrate = |y: u32| Dispatch {
            list: 0,
            pipeline: 1,
            builder: true,
            barrier: false,
            groups: Groups::Direct([1, 1, 1]),
            args: vec![cb(y)],
        };
        c.dispatches.push(rehydrate(0));
        c.dispatches.push(rehydrate(0));
        let why = c.incomplete_before_run();
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(why[0].contains("2 dispatches of 5e687674"), "{why:?}");
        c.dispatches.push(rehydrate(0x4880_0000));
        assert!(c.incomplete_before_run().is_empty());
    }

    #[test]
    fn a_complete_capture_passes_before_the_run() {
        let mut c = sample();
        c.meta.push(("pages left out (budget)".into(), "0".into()));
        let mut pre = c.pages[1].clone();
        pre.stage = Stage::Pre;
        c.pages.push(pre);
        assert!(c.incomplete_before_run().is_empty());
    }
}
