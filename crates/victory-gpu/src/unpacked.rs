//! Bottom levels the BVH builder unpacked in place go into title memory
//! (`VICTORY_BVH_UNPACK_BACK=1`; off by default, local test).
//!
//! The title's bottom-level structures come packed from its files (header
//! dword 0 = 0x8102....). Once per structure the title has the builder
//! unpack it in place on the GPU (4ddd6f97 clears bit 24 of dword 0, then
//! 5e687674 and the box stages write the nodes) and builds top levels over
//! it from then on. On the console that memory is the CPU's as well: the
//! unpacked structure is simply what is there, and the title's own check
//! before it asks again reads the header's packed bit there.
//!
//! Here the builder writes the host copy, and with the arena on what it
//! writes in title buffers never goes into title memory
//! (`crate::arena::BACK`, and nothing while lists miss memory). Title memory
//! kept the packed structure, so every later take of the copy from title
//! memory packed it again: the page's next refresh after the title wrote
//! anything in the same 64 KiB (a neighbour it loaded), the retake of
//! withheld pages at the frame's end (all of a small copy's, and a large
//! copy's unless kept: `crate::memory::keeps_pages`), a copy made anew, a
//! copy-back that takes title memory's other bytes. Run 233538: all 541
//! packed bottom levels the readers entered had been unpacked once, in
//! place; 5,143 kept pages taken from title memory again.
//!
//! With the switch, after a submission whose lists bound an unpacking
//! stage, each structure it names (source and destination the same
//! address) is looked at in the copy holding it: if the copy holds it
//! unpacked and title memory holds the same structure packed (dword 0 but
//! for bit 24, the size and the identifier equal), its bytes
//! `[address, address + dword 1 * 64)` go into title memory where they
//! differ ([`Shadow::put_range`]), and no other byte does: not what else
//! the builder's lists wrote in those pages (results computed from memory
//! with no copy stay out of title memory as before). From then on any take
//! from title memory holds the unpacked structure, in every copy, and the
//! title reads what the console's does.
//!
//! `=2`: not the structures of a list that itself missed memory.
//
// A structure may lie in several copies (capture C1 of 2026-11-28: three
// bottom levels of 5.3-5.6 MB, each over two 4 MiB title buffers, one ending
// in an arena page; they were never put and stayed packed, their instances'
// boxes empty). Its parts go back each from the copy the page table names
// for the page ([`owner_of`]), a part no copy holds from the arena's page;
// the header's part last, so that a part that cannot be written leaves the
// header packed in title memory.
// shortcut: a structure unpacked to another address than its source is left
// as before (counted; none in run 233538: 7,570 of 7,570 binds in place).
// shortcut: only the header says the stages ran; nodes computed from
// geometry with no copy go into title memory with it (they were the GPU's
// already, and are a structure's contents, no pointer the title follows).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::shadow::Shadow;

/// The switch's level: 0 off, 1 every structure, 2 not those of lists that
/// missed memory themselves.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = level_of(std::env::var("VICTORY_BVH_UNPACK_BACK").ok().as_deref());
        if level != 0 {
            tracing::warn!(
                level,
                "bottom levels the BVH builder unpacks in place go into title memory (VICTORY_BVH_UNPACK_BACK)"
            );
        }
        level
    })
}

fn level_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    }
}

/// The stages that unpack a bottom level: the header's and the leaves'
/// (shader hash prefixes; the box stages follow in the same list).
const STAGES: [&str; 2] = ["4ddd6f97", "5e687674"];

/// The structure a bind of the pipeline named `name` unpacks, from its
/// root constants `values` by user-data index: (source, destination), the
/// builder's root slots 1 and 2 at user data 35/36 and 37/38. None for
/// other programs and for addresses no header lies at.
pub(crate) fn pair(name: &str, values: &[u32]) -> Option<(u64, u64)> {
    if !STAGES.iter().any(|s| name.contains(s)) {
        return None;
    }
    let at = |i: usize| Some(u64::from(*values.get(i)?) | u64::from(*values.get(i + 1)?) << 32);
    let (source, destination) = (at(35)?, at(37)?);
    let header = |a: u64| a != 0 && a & 0x3f == 0;
    (header(source) && header(destination)).then_some((source, destination))
}

/// Header dword 0, bit 24: the structure is still packed.
const PACKED: u32 = 1 << 24;
/// The most 64-byte units a structure put back has (64 MiB).
const MOST: u32 = 1 << 20;

/// What a copy and title memory hold at a structure's address.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Head {
    /// The copy holds it unpacked, title memory the same structure packed,
    /// or unpacked already: its bytes (so many) go into title memory.
    Unpacked(usize),
    /// The copy holds it packed: no stage ran on this copy, or the copy
    /// took title memory again since.
    Packed,
    /// The copy holds it unpacked, title memory something else (another
    /// structure, loaded there meanwhile): the title's stays.
    Other,
    /// No bottom level's header in the copy.
    Unknown,
}

/// Judges the 16 header dwords the copy (`gpu`) and title memory (`title`)
/// hold at a structure's address.
fn judge(gpu: &[u32; 16], title: &[u32; 16]) -> Head {
    match gpu[0] >> 16 {
        0x8102 => return Head::Packed,
        0x8002 => {}
        _ => return Head::Unknown,
    }
    if gpu[1] == 0 || gpu[1] > MOST {
        return Head::Unknown;
    }
    // The same structure: dword 0 but for the packed bit, its size and its
    // identifier.
    if title[0] | PACKED == gpu[0] | PACKED && title[1..3] == gpu[1..3] {
        Head::Unpacked(gpu[1] as usize * 64)
    } else {
        Head::Other
    }
}

/// Since the last frame line: structures put into title memory, the bytes
/// of the lines written, structures title memory held as the copy already;
/// not put: packed in the copy, another structure in title memory, no
/// header, not unpacked in place, in no copy, of a list that missed (level
/// 2), title memory not readable or writable. And of those put, how many
/// had been put before: the title wrote the same structure there packed
/// again, or another one. With the `diagnostics` feature: of those put, the
/// ones whose root (a 112-byte box node, dword 12) ends past the size, and
/// of these the ones whose bytes there differ in the copy from title
/// memory's (they do not go back: an open question of the format).
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) struct Stats {
    pub(crate) put: u64,
    pub(crate) bytes: u64,
    pub(crate) there: u64,
    pub(crate) packed: u64,
    pub(crate) other: u64,
    pub(crate) unknown: u64,
    pub(crate) moved: u64,
    pub(crate) no_copy: u64,
    pub(crate) missed: u64,
    pub(crate) failed: u64,
    pub(crate) again_same: u64,
    pub(crate) again_other: u64,
    pub(crate) past: u64,
    pub(crate) tail: u64,
}

static STATS: Mutex<Stats> = Mutex::new(Stats {
    put: 0,
    bytes: 0,
    there: 0,
    packed: 0,
    other: 0,
    unknown: 0,
    moved: 0,
    no_copy: 0,
    missed: 0,
    failed: 0,
    again_same: 0,
    again_other: 0,
    past: 0,
    tail: 0,
});

/// The structures put into title memory so far: address → header dwords 1
/// and 2 (size, identifier). A structure found packed in title memory again
/// was written there by the title since (loaded again, or its address used
/// for another).
static PUT: Mutex<BTreeMap<u64, (u32, u32)>> = Mutex::new(BTreeMap::new());

/// After submission `submission` ran and its copy-backs are done: the
/// structures its lists' unpacking binds named (`pairs`: source,
/// destination, whether the list itself missed memory) go into title
/// memory ([`level`]). Only on the submitting thread, while no submission
/// is on the GPU.
pub(crate) fn back(submission: u64, pairs: &[(u64, u64, bool)]) {
    let level = level();
    if level == 0 || pairs.is_empty() {
        return;
    }
    let (t, put, not) = back_at(level, pairs, &crate::shadow::all());
    // `VICTORY_BVH_GT` (diagnostics): by address, to tell a structure the
    // title packed again from one never put (the readers' packed bottom
    // levels).
    #[cfg(not(feature = "diagnostics"))]
    let _ = (submission, &put, &not);
    #[cfg(feature = "diagnostics")]
    if crate::bvh_gt::target().is_some() && put.len() + not.len() > 0 {
        let list = |v: &[u64]| v.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" ");
        tracing::info!(
            "BVH GT s{submission} unpacked into title memory: {} [{}]; not put {} [{}]",
            put.len(),
            list(&put),
            not.len(),
            list(&not),
        );
    }
    let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
    s.put += t.put;
    s.bytes += t.bytes;
    s.there += t.there;
    s.packed += t.packed;
    s.other += t.other;
    s.unknown += t.unknown;
    s.moved += t.moved;
    s.no_copy += t.no_copy;
    s.missed += t.missed;
    s.failed += t.failed;
    s.again_same += t.again_same;
    s.again_other += t.again_other;
    s.past += t.past;
    s.tail += t.tail;
}

/// Logs why the structure at `at` counts as "no header" (the first 64 of a
/// run: three structures of capture C1 stayed packed this way).
fn why(at: u64, text: impl Fn() -> String) {
    static LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(64);
    if LEFT.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |n| n.checked_sub(1)).is_ok() {
        tracing::info!("unpacked bottom level at {at:#x} not put: {}", text());
    }
}

/// The 16 header dwords title memory holds at `at`, if readable.
fn title_head(at: u64) -> Option<[u32; 16]> {
    let mut title = [0u32; 16];
    // SAFETY: 64 bytes of ours to fill.
    let bytes = unsafe { std::slice::from_raw_parts_mut(title.as_mut_ptr().cast::<u8>(), 64) };
    crate::host::read_into(at as *const u8, bytes).then_some(title)
}

/// A part of a structure to put: `[at, at + len)` as a copy holds it, or
/// as the arena does (its bytes).
enum Part {
    Copy(Arc<Shadow>, u64, usize),
    Arena(u64, Vec<u8>),
}

/// A structure to put: its parts in address order, its address and bytes,
/// title memory's header as judged, and the copy's root pointer (dword 12).
type Chosen = (Vec<Part>, u64, usize, [u32; 16], u32);

/// Of copies with `ranges` (start, bytes), the one the page table names for
/// the page of `[at, at + len)` (one page's bytes): the smallest that holds
/// the whole page, else the largest that holds the bytes (`crate::memory`,
/// `write_table`: larger copies first, a copy holding the whole page takes
/// it from them).
fn owner_of(at: u64, len: u64, ranges: impl Iterator<Item = (u64, u64)>) -> Option<usize> {
    let page = at & !0xffff;
    let mut best: Option<(bool, u64, usize)> = None;
    for (k, (start, n)) in ranges.enumerate() {
        if start > at || at + len > start + n {
            continue;
        }
        let whole = start <= page && page + 0x1_0000 <= start + n;
        let better = match best {
            None => true,
            Some((w, size, _)) => (whole && !w) || (whole == w && if whole { n < size } else { n > size }),
        };
        if better {
            best = Some((whole, n, k));
        }
    }
    best.map(|(_, _, k)| k)
}

/// The parts of `[at, at + len)`: all of it in `first` (the copy whose
/// header was judged) if that holds it; else page by page from `first`
/// where it holds the page's bytes, from the page's copy ([`owner_of`]), or
/// from the arena (`arena(at, len)`: the bytes it holds there). None if
/// some part is held by nothing: the GPU did not write the structure whole.
fn parts(
    first: &Arc<Shadow>,
    at: u64,
    len: usize,
    copies: &[Arc<Shadow>],
    arena: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) -> Option<Vec<Part>> {
    let holds = |c: &Shadow, a: u64, n: usize| {
        let (from, to) = c.offsets(a as usize, n);
        to - from == n && c.range().0 <= a
    };
    if holds(first, at, len) {
        return Some(vec![Part::Copy(first.clone(), at, len)]);
    }
    let mut out: Vec<Part> = Vec::new();
    let (mut a, end) = (at, at + len as u64);
    while a < end {
        let n = ((a | 0xffff) + 1).min(end) - a;
        let copy = if holds(first, a, n as usize) {
            Some(first.clone())
        } else {
            owner_of(a, n, copies.iter().map(|c| c.range())).map(|k| copies[k].clone())
        };
        match copy {
            Some(c) => match out.last_mut() {
                Some(Part::Copy(last, from, bytes)) if Arc::ptr_eq(last, &c) && *from + *bytes as u64 == a => *bytes += n as usize,
                _ => out.push(Part::Copy(c, a, n as usize)),
            },
            None => out.push(Part::Arena(a, arena(a, n as usize).filter(|b| b.len() == n as usize)?)),
        }
        a += n;
    }
    Some(out)
}

/// `[at, at + len)` as the arena holds it, if it holds the page.
fn arena_bytes(at: u64, len: usize) -> Option<Vec<u8>> {
    let page = at & !0xffff;
    let (_, bytes) = crate::arena::fetch(&[page]).into_iter().next()?;
    let o = (at - page) as usize;
    bytes.get(o..o + len).map(<[u8]>::to_vec)
}

/// [`back`] at `level` among `copies`: what was done, the addresses put
/// and those not put.
fn back_at(level: u8, pairs: &[(u64, u64, bool)], copies: &[Arc<Shadow>]) -> (Stats, Vec<u64>, Vec<u64>) {
    back_with(level, pairs, copies, &arena_bytes)
}

/// [`back_at`] with the arena's bytes from `arena`.
fn back_with(
    level: u8,
    pairs: &[(u64, u64, bool)],
    copies: &[Arc<Shadow>],
    arena: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) -> (Stats, Vec<u64>, Vec<u64>) {
    let mut s = Stats::default();
    let (mut put, mut not) = (Vec::new(), Vec::new());
    // Each destination once (a structure has a bind per stage); a list that
    // missed counts for all of them.
    let mut wanted: BTreeMap<u64, bool> = BTreeMap::new();
    for &(source, destination, missed) in pairs {
        if source != destination {
            s.moved += 1;
            not.push(destination);
            continue;
        }
        *wanted.entry(destination).or_default() |= missed;
    }
    if level == 2 {
        wanted.retain(|&at, missed| {
            if *missed {
                s.missed += 1;
                not.push(at);
            }
            !*missed
        });
    }
    // The copies holding each header, the larger first (as the page table
    // names them: `crate::memory`), and what the GPU left there in each.
    let mut asked: Vec<(Arc<Shadow>, u64)> = Vec::new();
    for &at in wanted.keys() {
        let mut holding: Vec<&Arc<Shadow>> = copies
            .iter()
            .filter(|c| {
                let (start, len) = c.range();
                start <= at && at + 64 <= start + len
            })
            .collect();
        holding.sort_by_key(|c| std::cmp::Reverse(c.range().1));
        if holding.is_empty() {
            s.no_copy += 1;
            not.push(at);
        }
        asked.extend(holding.into_iter().map(|c| (c.clone(), at)));
    }
    let heads = crate::shadow::gpu_heads(&asked);
    // Per structure, the first copy that holds it unpacked over the same
    // structure in title memory; else what the first copy says.
    let mut chosen: Vec<Chosen> = Vec::new();
    let mut k = 0;
    while k < asked.len() {
        let at = asked[k].1;
        let n = asked[k..].iter().take_while(|a| a.1 == at).count();
        let Some(title) = title_head(at) else {
            s.failed += 1;
            not.push(at);
            k += n;
            continue;
        };
        let judged: Vec<Head> = (k..k + n).map(|i| heads[i].map_or(Head::Unknown, |gpu| judge(&gpu, &title))).collect();
        let unpacked = judged.iter().enumerate().find_map(|(i, h)| match h {
            Head::Unpacked(len) => Some((i, *len)),
            _ => None,
        });
        match unpacked {
            Some((i, len)) => match parts(&asked[k + i].0, at, len, copies, arena) {
                Some(p) => chosen.push((p, at, len, title, heads[k + i].map_or(0, |gpu| gpu[12]))),
                // A part of it in no copy and not in the arena.
                None => {
                    why(at, || format!("unpacked in its copy, {len} bytes, but a part of it is in no copy and in no arena page"));
                    s.unknown += 1;
                    not.push(at);
                }
            },
            None => {
                match judged[0] {
                    Head::Packed => s.packed += 1,
                    Head::Other => s.other += 1,
                    _ => {
                        why(at, || format!("no bottom level's header in {n} copies: dwords 0 and 1 {:x?}", heads[k].map(|h| [h[0], h[1]])));
                        s.unknown += 1;
                    }
                }
                not.push(at);
            }
        }
        k += n;
    }
    // Of copies in the GPU's memory the structures come down, in one fetch.
    let fetch: Vec<(&Shadow, usize, usize)> = chosen
        .iter()
        .flat_map(|(parts, ..)| parts)
        .filter_map(|part| match part {
            Part::Copy(c, at, len) => {
                let (from, to) = c.offsets(*at as usize, *len);
                Some((&**c, from, to))
            }
            Part::Arena(..) => None,
        })
        .collect();
    crate::host::card_fetch(&fetch);
    drop(fetch);
    let mut before = PUT.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    let mut roots: Vec<(Arc<Shadow>, u64, u32)> = Vec::new();
    for (parts, at, len, title, root) in chosen {
        // Title memory must still hold the header that was judged: the
        // fetch above is a round trip to the GPU while the title runs, and
        // what it wrote there meanwhile (another structure) stays.
        #[cfg(test)]
        tests::between();
        match title_head(at) {
            None => {
                s.failed += 1;
                not.push(at);
                continue;
            }
            Some(now) if now != title => {
                s.other += 1;
                not.push(at);
                continue;
            }
            Some(_) => {}
        }
        // The last part first, the header's last: a part that cannot be
        // written leaves the header as it was.
        let (mut bytes, mut all) = (0, true);
        for part in parts.iter().rev() {
            let (n, ok) = match part {
                Part::Copy(copy, from, n) => copy.put_range(*from as usize, *n),
                Part::Arena(from, held) => crate::shadow::put_bytes(*from as usize, held),
            };
            bytes += n;
            all &= ok;
            if !ok {
                break;
            }
        }
        if !all {
            s.failed += 1;
            not.push(at);
            continue;
        }
        if title[0] & PACKED == 0 {
            s.there += 1;
        } else if let Some(&(size, id)) = before.get(&at) {
            if (size, id) == (title[1], title[2]) {
                s.again_same += 1;
            } else {
                s.again_other += 1;
            }
        }
        s.put += 1;
        s.bytes += bytes as u64;
        before.insert(at, (title[1], title[2]));
        put.push(at);
        // A box node of 112 bytes as the root, ending past the size.
        #[cfg(feature = "diagnostics")]
        if root & 7 == 5 {
            let end = (u64::from(root & !7) << 3) + 112;
            if end > len as u64
                && end <= len as u64 + 64
                && let Some(Part::Copy(copy, ..)) = parts.last()
            {
                s.past += 1;
                roots.push((copy.clone(), at + len as u64, (end - len as u64) as u32));
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (root, len);
    }
    drop(before);
    // Once every structure is put (the next one's header may lie there):
    // whether the copy holds other bytes past the size than title memory.
    // Not of copies in the GPU's memory, whose bytes there were not fetched.
    #[cfg(feature = "diagnostics")]
    for (copy, at, n) in roots {
        if copy.carded() {
            continue;
        }
        let mut title = [0u8; 64];
        let n = n as usize;
        if let Some(gpu) = copy.host_dwords(at, n / 4, false)
            && crate::host::read_into(at as *const u8, &mut title[..n])
            && gpu.iter().flat_map(|d| d.to_le_bytes()).ne(title[..n].iter().copied())
        {
            s.tail += 1;
        }
    }
    (s, put, not)
}

/// A line per presented frame in which structures were looked at.
pub(crate) fn log_frame() {
    if level() == 0 {
        return;
    }
    let s = std::mem::take(&mut *STATS.lock().unwrap_or_else(|e| e.into_inner()));
    if s == Stats::default() {
        return;
    }
    tracing::info!(
        "unpacked bottom levels this frame: {} put into title memory ({} bytes written; {} held unpacked there already, \
         {} the title had packed again: {} the same structure, {} another); not put: {} packed in the copy, \
         {} another structure in title memory, {} no header, {} not in place, {} in no copy, {} their list missed memory, \
         {} title memory not readable or writable",
        s.put,
        s.bytes,
        s.there,
        s.again_same + s.again_other,
        s.again_same,
        s.again_other,
        s.packed,
        s.other,
        s.unknown,
        s.moved,
        s.no_copy,
        s.missed,
        s.failed,
    );
    #[cfg(feature = "diagnostics")]
    tracing::info!(
        "unpacked bottom levels this frame: of those put, {} have a root ending past the size, {} of them with other bytes there than title memory",
        s.past,
        s.tail,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::{find, register, unregister};
    use crate::watch::PAGE;

    thread_local! {
        /// Run once between the look at title memory's header and the write
        /// (what the title does meanwhile).
        static BETWEEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn between() {
        if let Some(f) = BETWEEN.with(|b| b.borrow_mut().take()) {
            f();
        }
    }

    /// Where the test's structure lies in its buffer, its 64-byte units,
    /// and its identifier.
    const AT: usize = PAGE + 0x100;
    const UNITS: u32 = 3;
    const ID: u32 = 0x1234_5678;

    fn header(dword0: u32, units: u32, id: u32) -> [u32; 16] {
        let mut h = [0u32; 16];
        (h[0], h[1], h[2]) = (dword0, units, id);
        h
    }

    fn bytes_of(h: &[u32; 16]) -> Vec<u8> {
        h.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    /// Title memory `[p, p + len)` filled with 0x11, a packed structure of
    /// `UNITS` units at `AT` (its body 0x11 too).
    fn fill(p: usize, len: usize) {
        // SAFETY: bytes of the test's memory, not guarded yet.
        unsafe {
            std::ptr::write_bytes(p as *mut u8, 0x11, len);
            std::ptr::copy_nonoverlapping(bytes_of(&header(0x8102_0006, UNITS, ID)).as_ptr(), (p + AT) as *mut u8, 64);
        }
    }

    /// The GPU unpacks the structure in `copy`: the header's bit falls, its
    /// body becomes 0x22; and it writes a byte right after the structure
    /// (0x77: something else of the builder's in the same page).
    fn unpack(copy: &Shadow, p: usize) {
        copy.write_host((p + AT) as u64, &bytes_of(&header(0x8002_0006, UNITS, ID)));
        copy.write_host((p + AT + 64) as u64, &vec![0x22; (UNITS as usize - 1) * 64]);
        copy.write_host((p + AT + UNITS as usize * 64) as u64, &[0x77]);
    }

    fn title_dword(at: usize) -> u32 {
        // SAFETY: four bytes of the test's memory.
        unsafe { (at as *const u32).read_volatile() }
    }

    fn title_byte(at: usize) -> u8 {
        // SAFETY: a byte of the test's memory.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// The switch's values, and the binds that name a structure.
    #[test]
    fn the_switch_and_the_unpacking_binds() {
        assert_eq!((level_of(None), level_of(Some("0")), level_of(Some(" 1 ")), level_of(Some("2")), level_of(Some("3"))), (0, 0, 1, 2, 0));
        let mut values = vec![0u32; 40];
        (values[35], values[36], values[37], values[38]) = (0x0405_e200, 7, 0x0405_e200, 7);
        assert_eq!(pair("[compute 4ddd6f97aa]", &values), Some((0x7_0405_e200, 0x7_0405_e200)));
        assert_eq!(pair("[compute 5e687674bb]", &values), Some((0x7_0405_e200, 0x7_0405_e200)));
        assert_eq!(pair("[compute 190fbf0e]", &values), None, "another stage");
        values[37] = 0x0405_e300;
        assert_eq!(pair("4ddd6f97", &values), Some((0x7_0405_e200, 0x7_0405_e300)), "not in place: both named");
        values[37] = 0x0405_e308;
        assert_eq!(pair("4ddd6f97", &values), None, "no header lies there");
        assert_eq!(pair("4ddd6f97", &values[..38]), None, "too few constants");
    }

    /// Only a structure the copy holds unpacked and title memory holds as
    /// the same structure goes back.
    #[test]
    fn a_header_is_judged_against_title_memorys() {
        let gpu = header(0x8002_0006, UNITS, ID);
        assert_eq!(judge(&gpu, &header(0x8102_0006, UNITS, ID)), Head::Unpacked(192));
        assert_eq!(judge(&gpu, &gpu), Head::Unpacked(192), "there already: its bytes still compared");
        assert_eq!(judge(&header(0x8102_0006, UNITS, ID), &header(0x8102_0006, UNITS, ID)), Head::Packed);
        assert_eq!(judge(&gpu, &header(0x8102_0006, UNITS, ID + 1)), Head::Other, "another identifier");
        assert_eq!(judge(&gpu, &header(0x8102_0006, UNITS + 1, ID)), Head::Other, "another size");
        assert_eq!(judge(&gpu, &header(0x8102_0007, UNITS, ID)), Head::Other, "another dword 0");
        assert_eq!(judge(&gpu, &[0x1111_1111; 16]), Head::Other, "no structure in title memory");
        assert_eq!(judge(&[0x2222_2222; 16], &header(0x8102_0006, UNITS, ID)), Head::Unknown);
        assert_eq!(judge(&header(0x8002_0006, 0, ID), &header(0x8102_0006, 0, ID)), Head::Unknown, "no size");
        assert_eq!(judge(&header(0x8002_0006, MOST + 1, ID), &header(0x8102_0006, MOST + 1, ID)), Head::Unknown);
    }

    /// Invariant "what the GPU wrote is visible to the CPU at the sync
    /// point": a structure unpacked in a copy is in title memory after its
    /// submission, its bytes and no others; and "no take from title memory
    /// undoes GPU writes the CPU did not overwrite": the copy holds it
    /// unpacked after the title wrote the same page, after the frame's end
    /// dropped the withheld page, and a copy made anew of the memory does.
    #[test]
    fn an_unpacked_structure_is_in_title_memory_and_survives_every_take() {
        let pages = 6;
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        fill(p, len);
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let at = (p + AT) as u64;
        let end = p + AT + UNITS as usize * 64;
        copy.update(p, len);
        copy.writing(p, len, false);
        unpack(&copy, p);
        // The submission withholds the range: nothing went into title memory.
        copy.keep_writes(p, len);
        assert_eq!(title_dword(p + AT), 0x8102_0006);
        // The sync point (32-byte lines: the header's second is the same in both).
        let (s, put, not) = back_at(1, &[(at, at, true), (at, at, false)], std::slice::from_ref(&copy));
        assert_eq!((s.put, s.bytes, put, not), (1, 160, vec![at], vec![]), "{s:?}");
        assert_eq!(title_dword(p + AT), 0x8002_0006, "the CPU reads the header unpacked");
        assert_eq!((title_dword(p + AT + 4), title_dword(p + AT + 8)), (UNITS, ID));
        assert!((p + AT + 64..end).all(|a| title_byte(a) == 0x22), "and the structure's body");
        assert_eq!((title_byte(p + AT - 1), title_byte(end)), (0x11, 0x11), "nothing beside the structure went back");
        // The frame's end drops what the frame withheld, and the title writes
        // the same page (a neighbour it loads): the copy takes the page.
        crate::memory::drop_withheld(p, len);
        crate::memory::log_frame();
        // SAFETY: a byte of our pool (a guarded page faults into the handler).
        unsafe { ((p + AT + 0x800) as *mut u8).write_volatile(0x33) };
        copy.update(p, len);
        assert_eq!(copy.read_host(at, 3), [0x8002_0006, UNITS, ID], "the copy holds it unpacked after the take");
        assert_eq!(copy.read_host(at + 64, 1), [0x2222_2222]);
        assert_eq!(copy.read_host((p + AT + 0x800) as u64, 1)[0] & 0xff, 0x33, "with the title's write");
        // A copy made anew of the same memory.
        register(p + 1, p as u64, len as u64);
        let anew = find(p as u64 + PAGE as u64, PAGE as u64).filter(|c| !Arc::ptr_eq(c, &copy));
        if let Some(anew) = &anew {
            anew.update(p, len);
            assert_eq!(anew.read_host(at, 1), [0x8002_0006], "a new copy takes it unpacked");
        }
        // The title loads the structure again, packed: its bytes win, and
        // the next unpacking is counted as one put before.
        // SAFETY: bytes of our pool (guarded: faults into the handler).
        unsafe { std::ptr::copy_nonoverlapping(bytes_of(&header(0x8102_0006, UNITS, ID)).as_ptr(), (p + AT) as *mut u8, 64) };
        copy.update(p, len);
        assert_eq!(copy.read_host(at, 1), [0x8102_0006], "what the CPU wrote is the copy's");
        copy.write_host(at, &0x8002_0006u32.to_le_bytes());
        let (s, ..) = back_at(1, &[(at, at, false)], std::slice::from_ref(&copy));
        assert_eq!((s.put, s.again_same, s.again_other), (1, 1, 0), "{s:?}");
        PUT.lock().unwrap().remove(&at);
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// In memory the runtime maps as views (as the title's GPU heaps), the
    /// structure goes back through our own view: the copy's page stays
    /// taken, and what else shaders wrote in it stays in the copy.
    #[test]
    fn beside_the_guard_the_pages_other_gpu_bytes_stay_in_the_copy() {
        let pages = 3;
        let (section, a, _) = crate::watch::tests::reported_section(pages);
        let len = pages * PAGE;
        fill(a, len);
        register(a, a as u64, len as u64);
        let Some(copy) = find(a as u64, len as u64) else {
            unregister(a, a as u64);
            crate::watch::title_memory_section_gone(section);
            return; // no host GPU here
        };
        let at = (a + AT) as u64;
        let end = a + AT + UNITS as usize * 64;
        copy.update(a, len);
        unpack(&copy, a);
        let (s, ..) = back_at(1, &[(at, at, false)], std::slice::from_ref(&copy));
        assert_eq!((s.put, s.bytes), (1, 160), "{s:?}");
        assert_eq!((title_dword(a + AT), title_byte(a + AT + 64), title_byte(end)), (0x8002_0006, 0x22, 0x11));
        assert_eq!(copy.title_changed(a + AT), Some(false), "our write is not the title's: nothing to take");
        copy.update(a, len);
        assert_eq!(copy.read_host(end as u64, 1)[0] & 0xff, 0x77, "the builder's other byte is still the copy's");
        assert_eq!(copy.read_host(at, 1), [0x8002_0006]);
        PUT.lock().unwrap().remove(&at);
        unregister(a, a as u64);
        crate::watch::title_memory_section_gone(section);
    }

    /// Nothing goes into title memory when the copy still holds the
    /// structure packed, when title memory holds another structure (the
    /// title loaded one there meanwhile), when it was unpacked to another
    /// address, when no copy holds it, or at level 2 when its list missed.
    #[test]
    fn what_is_not_the_same_structure_unpacked_in_place_stays_out() {
        let pages = 6;
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        fill(p, len);
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let at = (p + AT) as u64;
        let copies = std::slice::from_ref(&copy);
        let title: Vec<u8> = (0..len).map(|o| title_byte(p + o)).collect();
        let untouched = |why: &str| assert!((0..len).all(|o| title_byte(p + o) == title[o]), "title memory written: {why}");
        copy.update(p, len);
        // Still packed in the copy (the stages did not run).
        copy.write_host(at + 64, &[0x22; 64]);
        let (s, put, not) = back_at(1, &[(at, at, false)], copies);
        assert_eq!((s.packed, s.put, put, not), (1, 0, vec![], vec![at]), "{s:?}");
        untouched("packed in the copy");
        unpack(&copy, p);
        // Level 2, and the list missed memory.
        let (s, ..) = back_at(2, &[(at, at, false), (at, at, true)], copies);
        assert_eq!((s.missed, s.put), (1, 0), "{s:?}");
        untouched("its list missed");
        // Unpacked to another address.
        let (s, ..) = back_at(1, &[(at + 0x1000, at, false)], copies);
        assert_eq!((s.moved, s.put), (1, 0), "{s:?}");
        untouched("not in place");
        // No copy holds it.
        let (s, ..) = back_at(1, &[(at, at, false)], &[]);
        assert_eq!((s.no_copy, s.put), (1, 0), "{s:?}");
        untouched("no copy");
        // The title loaded another structure there since.
        // SAFETY: bytes of our pool (guarded: faults into the handler).
        unsafe { ((p + AT + 8) as *mut u32).write_volatile(ID + 1) };
        let (s, ..) = back_at(1, &[(at, at, false)], copies);
        assert_eq!((s.other, s.put), (1, 0), "{s:?}");
        assert_eq!((title_dword(p + AT), title_dword(p + AT + 8), title_byte(p + AT + 64)), (0x8102_0006, ID + 1, 0x11), "the title's structure stays");
        unregister(p, p as u64);
    }

    /// Fable (BVH proof, capture C1: three bottom levels of 5.3-5.6 MB stayed
    /// packed, each over two 4 MiB copies): a structure that lies in two
    /// copies goes into title memory whole, each page's part from the copy
    /// the page table names for it, and nothing beside it does.
    #[test]
    fn a_structure_over_two_copies_goes_back_whole() {
        let pages = 6;
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        fill(p, len);
        let half = 3 * PAGE;
        register(p, p as u64, half as u64);
        register(p + 1, (p + half) as u64, half as u64);
        let (Some(a), Some(b)) = (find(p as u64, half as u64), find((p + half) as u64, half as u64)) else {
            unregister(p + 1, (p + half) as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // 16 units from 0x200 before the second copy: 8 in each.
        let units = 16u32;
        let at = p + half - 0x200;
        let head = header(0x8102_0006, units, ID);
        // SAFETY: bytes of the test's memory, not guarded yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes_of(&head).as_ptr(), at as *mut u8, 64) };
        a.update(p, half);
        b.update(p + half, half);
        a.write_host(at as u64, &bytes_of(&header(0x8002_0006, units, ID)));
        a.write_host(at as u64 + 64, &vec![0x22; 0x200 - 64]);
        b.write_host((p + half) as u64, &vec![0x33; 0x200]);
        // Something else of the builder's right after the structure.
        b.write_host((p + half + 0x200) as u64, &[0x77]);
        let copies = [a.clone(), b.clone()];
        let (s, put, not) = back_at(1, &[(at as u64, at as u64, false)], &copies);
        assert_eq!((s.put, s.unknown, put, not), (1, 0, vec![at as u64], vec![]), "{s:?}");
        assert_eq!(title_dword(at), 0x8002_0006, "the header unpacked");
        assert!((at + 64..p + half).all(|x| title_byte(x) == 0x22), "the first copy's part");
        assert!((p + half..p + half + 0x200).all(|x| title_byte(x) == 0x33), "the second copy's part");
        assert_eq!((title_byte(at - 1), title_byte(p + half + 0x200)), (0x11, 0x11), "nothing beside the structure");
        PUT.lock().unwrap().remove(&(at as u64));
        unregister(p + 1, (p + half) as u64);
        unregister(p, p as u64);
    }

    /// A structure whose end lies in no copy: with the arena holding those
    /// pages their bytes go back from there; with nothing holding them the
    /// structure stays out whole (its header stays packed in title memory).
    #[test]
    fn a_part_in_no_copy_comes_from_the_arena_or_keeps_the_structure_out() {
        let pages = 6;
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        fill(p, len);
        let half = 3 * PAGE;
        register(p, p as u64, half as u64);
        let Some(a) = find(p as u64, half as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let units = 16u32;
        let at = p + half - 0x200;
        // SAFETY: bytes of the test's memory, not guarded yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes_of(&header(0x8102_0006, units, ID)).as_ptr(), at as *mut u8, 64) };
        a.update(p, half);
        a.write_host(at as u64, &bytes_of(&header(0x8002_0006, units, ID)));
        a.write_host(at as u64 + 64, &vec![0x22; 0x200 - 64]);
        let copies = std::slice::from_ref(&a);
        let title: Vec<u8> = (0..len).map(|o| title_byte(p + o)).collect();
        // Nothing holds the second half.
        let (s, put, not) = back_with(1, &[(at as u64, at as u64, false)], copies, &|_, _| None);
        assert_eq!((s.put, s.unknown, put, not), (0, 1, vec![], vec![at as u64]), "{s:?}");
        assert!((0..len).all(|o| title_byte(p + o) == title[o]), "title memory written");
        // The arena holds it (a page of 0x44).
        let asked = std::cell::RefCell::new(Vec::new());
        let arena = |page: u64, n: usize| {
            asked.borrow_mut().push((page, n));
            Some(vec![0x44; n])
        };
        let (s, put, _) = back_with(1, &[(at as u64, at as u64, false)], copies, &arena);
        assert_eq!((s.put, put), (1, vec![at as u64]), "{s:?}");
        assert_eq!(*asked.borrow(), vec![((p + half) as u64, 0x200)]);
        assert_eq!((title_dword(at), title_byte(at + 64)), (0x8002_0006, 0x22));
        assert!((p + half..p + half + 0x200).all(|x| title_byte(x) == 0x44), "the arena's part");
        assert_eq!(title_byte(p + half + 0x200), 0x11, "nothing past the structure");
        PUT.lock().unwrap().remove(&(at as u64));
        unregister(p, p as u64);
    }

    /// The copy the page table names for a page: the smallest one holding
    /// the whole page, else the largest holding the bytes asked for
    /// (`crate::memory`'s rule).
    #[test]
    fn a_pages_copy_is_the_smallest_holding_it_whole() {
        let ranges = [(0x10_0000u64, 0x80_0000u64), (0x10_0000, 0x40_0000), (0x50_0000, 0x40_0000), (0x90_8000, 0x1_0000)];
        let pick = |at: u64, n: u64| owner_of(at, n, ranges.iter().copied());
        assert_eq!(pick(0x10_0100, 0x100), Some(1), "the 4 MiB copy before the 8 MiB one over it");
        assert_eq!(pick(0x50_0000, 0x1_0000), Some(2));
        assert_eq!(pick(0x90_8000, 0x100), Some(3), "a part of the page: the only holder");
        assert_eq!(pick(0x90_7f00, 0x100), None);
        assert_eq!(pick(0xa0_0000, 0x100), None);
    }

    /// The title wrote another structure there after its header was looked
    /// at and before the write: its bytes stay, nothing of the copy's goes
    /// over them.
    #[test]
    fn a_structure_the_title_writes_meanwhile_keeps_the_titles_bytes() {
        let pages = 6;
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        fill(p, len);
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let at = (p + AT) as u64;
        copy.update(p, len);
        unpack(&copy, p);
        // SAFETY: a dword of our pool (guarded: faults into the handler).
        BETWEEN.with(|b| *b.borrow_mut() = Some(Box::new(move || unsafe { ((p + AT + 8) as *mut u32).write_volatile(ID + 1) })));
        let (s, put, not) = back_at(1, &[(at, at, false)], std::slice::from_ref(&copy));
        assert_eq!((s.other, s.put, put, not), (1, 0, vec![], vec![at]), "{s:?}");
        assert_eq!((title_dword(p + AT), title_dword(p + AT + 8), title_byte(p + AT + 64)), (0x8102_0006, ID + 1, 0x11), "the title's structure stays");
        unregister(p, p as u64);
    }
}
