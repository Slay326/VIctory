//! A CPU reference for the RT reader `f607780a`
//! (`indirect_light_probes:CS_IrradianceCache_ShadowTrace_WithFar`) over a
//! reader capture (Fable, BVH proof 2026-11-28): the program's rays are
//! rebuilt from its views, every ray is tested in f64 against every triangle
//! the acceleration structures hold (their boxes ignored), and the verdict
//! is compared with the bytes the game's GPU wrote. None of the translator
//! is used: what the program does is taken from its disassembly, the node
//! layouts from the public sources `emit/lower/bvh.rs` names.
//!
//! Three verdicts a ray:
//! * **triangles** ([`Scene::brute`]): any triangle of any instance the
//!   program enters, within the ray's extent. Boxes, tree order and parent
//!   links play no part: what the geometry says.
//! * **tree** ([`Scene::walk`]): the same triangles, reached only through
//!   the structures' boxes (f64, no stack limit). Differs from the first
//!   where a box is too small or a node is not reached.
//! * **game**: `u52` after the fence (0 a hit, 127 none).
//!
//! and one check without rays ([`Scene::check`]): every box holds what is
//! below it down to the triangles, every instance's box holds its
//! transformed bottom level, every node is reached once, and the parent
//! links the program's stackless way back reads name the parent.
//!
//! The program, from its disassembly (offsets in the native code):
//! * 0x030 thread = group * 32 + lane, active below `t379[s6]`.
//! * 0x114 `s6 != 16`: `w = t339[thread].x`, probe `w & 0x0fffffff`,
//!   cascade `w >> 29`; 0x08c `s6 == 16`: `v = t373[thread]`, probe `v % N`,
//!   cascade `v / N` (`N` = `b3` dword 5).
//! * 0x13c `t12`: `s4`, `s5` at 0x58c, 0x590; `O` (four floats) at 0x594 +
//!   16 * cascade, `H` at 0x5f0 + 16 * cascade, `R0` at 0x7c8. With
//!   `x = probe % s5`, `y = (probe / s5) % s4`, `z = probe / (s5 * s4)` and
//!   `n(k, dim) = 2 (k + 0.5) / dim - 1`: origin
//!   `(O.x + O.w n(z, s4), O.y + O.w n(y, s4), O.z + H n(x, s5))`.
//! * 0x310 `L` = `t10` at 0x110, direction `-L`; inside the sphere of
//!   radius `0.7 R0`: `v8` = the way out of it along the direction, else 0.
//! * Near: the top level behind `t355`, from the origin, t in `[0, E)`,
//!   `E` = `b3` at 0x44. Far, only without a near hit and with
//!   `E - v8 > 0`: the top level behind `t356`, from `origin - v8 L`, t in
//!   `[0, E - v8)`.
//! * The walk (0x430-0xb34): box nodes give their hit children nearest
//!   first; an instance node (pointer type 6) is entered only with bit 30
//!   of its dword 12: the bottom level's root is dword 14 & 0x3fffffff
//!   (0x3fffffff: none), its base dword 15 << 8, the ray becomes
//!   `M ray + T` with `T` dwords 16-18 and the rows of `M` dwords 19-27, `M`
//!   left out where bit 31 of dword 14 is set. A triangle hit counts with
//!   `0 <= t < E` whichever side it is met from (the face term is
//!   multiplied by 0 at 0x940), and the first one ends the ray. After a
//!   type-0 triangle whose node has bit 7 of dword 12 set, the same node's
//!   type-1 triangle is tested.
//! * The stack holds 16 entries a lane; past that the program finds its way
//!   back over parent links: of a box node at slot `s` the dword at
//!   `base + D14 + 4 (s - D13 / 64)`, of an instance its dword 29.
//! * 0x134c `u52[cascade * N + probe]` = 0.5 without a hit, 0 with one.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};

use super::format::{Capture, Dispatch, Entry, INDIRECT_ARGUMENTS, PAGE, RootArg, Stage, Table, TableView};

/// No node.
pub const NONE: u32 = u32::MAX;
/// The vertices (of V0-V4) of a triangle node's triangle, by pointer type.
const TRIANGLE: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
/// The most nodes a structure is read to before it gives up (a cycle).
const MOST_NODES: usize = 1 << 22;
/// The most nodes a ray's walk visits (the program's own limit is 0xffff
/// rounds a top level).
const MOST_VISITS: usize = 1 << 17;
/// A barycentric coordinate within this of an edge is neither in nor out.
const EDGE: f64 = 1e-3;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn float(w: u32) -> f64 {
    f64::from(f32::from_bits(w))
}
/// An IEEE half as f64.
fn half(h: u32) -> f64 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let v = match e {
        0 => f64::from(m) * (2f64).powi(-24),
        31 => {
            if m == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => (1.0 + f64::from(m) / 1024.0) * (2f64).powi(e as i32 - 15),
    };
    if s == 1 { -v } else { v }
}

/// Title memory as the program reads it over the page table.
pub trait Memory {
    /// The dword at `at`: 0 where the page table names nothing (the program
    /// reads zeros there), `None` where it names a page that is not held.
    fn word(&self, at: u64) -> Option<u32>;

    /// `N` dwords from `at`.
    fn words<const N: usize>(&self, at: u64) -> Option<[u32; N]>
    where
        Self: Sized,
    {
        let mut out = [0u32; N];
        for (i, w) in out.iter_mut().enumerate() {
            *w = self.word(at + 4 * i as u64)?;
        }
        Some(out)
    }
}

/// Which of the page records at one address the page table's entry means:
/// a copy's by its segment, an arena page's by its chunk's segment (or the
/// only record there).
pub fn pick(entry: Entry, records: &[(u32, u64)]) -> Option<u64> {
    let seg = match entry {
        Entry::Null => return None,
        Entry::Copy { seg } | Entry::Arena { seg, .. } => seg,
    };
    records
        .iter()
        .find(|(s, _)| *s == seg)
        .or(match (entry, records) {
            (Entry::Arena { .. }, [one]) => Some(one),
            _ => None,
        })
        .map(|&(_, offset)| offset)
}

/// The pages a capture took before the program ran, read over its page
/// table.
pub struct CaptureMemory {
    table: Table,
    /// Page address -> (segment, offset in the pages file; `u64::MAX`: zeros).
    records: HashMap<u64, Vec<(u32, u64)>>,
    file: RefCell<std::fs::File>,
    cache: RefCell<HashMap<u64, Option<Box<[u8]>>>>,
    /// Pages the table names that the capture does not hold.
    pub missing: RefCell<BTreeSet<u64>>,
    /// Pages read that the table names nothing for.
    pub null: RefCell<BTreeSet<u64>>,
}

impl CaptureMemory {
    pub fn new(capture: &Capture, pre: std::fs::File) -> CaptureMemory {
        let mut records: HashMap<u64, Vec<(u32, u64)>> = HashMap::new();
        for p in capture.pages_of(Stage::Pre) {
            records.entry(p.address & !(PAGE - 1)).or_default().push((p.seg, p.file_offset));
        }
        CaptureMemory {
            table: capture.table.clone(),
            records,
            file: RefCell::new(pre),
            cache: RefCell::new(HashMap::new()),
            missing: RefCell::new(BTreeSet::new()),
            null: RefCell::new(BTreeSet::new()),
        }
    }

    fn load(&self, page: u64) -> Option<Box<[u8]>> {
        let entry = self.table.entry_at(page).map_or(Entry::Null, |(_, e)| e);
        if entry == Entry::Null {
            self.null.borrow_mut().insert(page);
            return Some(vec![0u8; PAGE as usize].into_boxed_slice());
        }
        let offset = self.records.get(&page).and_then(|r| pick(entry, r));
        let Some(offset) = offset else {
            self.missing.borrow_mut().insert(page);
            return None;
        };
        let mut bytes = vec![0u8; PAGE as usize];
        if offset != u64::MAX {
            let mut f = self.file.borrow_mut();
            if f.seek(SeekFrom::Start(offset)).is_err() || f.read_exact(&mut bytes).is_err() {
                self.missing.borrow_mut().insert(page);
                return None;
            }
        }
        Some(bytes.into_boxed_slice())
    }
}

impl Memory for CaptureMemory {
    fn word(&self, at: u64) -> Option<u32> {
        let page = at & !(PAGE - 1);
        let mut cache = self.cache.borrow_mut();
        let bytes = cache.entry(page).or_insert_with(|| self.load(page)).as_ref()?;
        let o = (at - page) as usize;
        let b = bytes.get(o..o + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Memory of dwords by address (tests): unwritten ones read 0.
#[derive(Default)]
pub struct WordMemory(pub HashMap<u64, u32>);

impl Memory for WordMemory {
    fn word(&self, at: u64) -> Option<u32> {
        Some(self.0.get(&at).copied().unwrap_or(0))
    }
}

/// A ray of the program.
#[derive(Debug, Clone, PartialEq)]
pub struct Ray {
    pub thread: u32,
    pub probe: u32,
    pub cascade: u32,
    /// The element of `u52` it writes.
    pub slot: u64,
    pub origin: V3,
    pub direction: V3,
    pub extent: f64,
    /// The far ray: its origin and extent (not shot unless the extent is
    /// above 0).
    pub far_origin: V3,
    pub far_extent: f64,
}

/// The rays of a dispatch of `groups` groups of 32 with root constant `s6`,
/// from the views' bytes before it ran (`view("t12")` and so on).
pub fn rays(view: &dyn Fn(&str) -> Option<Vec<u8>>, s6: u32, groups: u32) -> Result<Vec<Ray>, String> {
    let get = |name: &str| view(name).filter(|b| !b.is_empty()).ok_or_else(|| format!("no bytes of {name}"));
    let u = |b: &[u8], o: usize| -> Result<u32, String> {
        b.get(o..o + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| format!("a view ends before byte {o}"))
    };
    let f = |b: &[u8], o: usize| -> Result<f32, String> { u(b, o).map(f32::from_bits) };
    let (t379, t12, t10, b3) = (get("t379")?, get("t12")?, get("t10")?, get("b3")?);
    let ids = get(if s6 == 16 { "t373" } else { "t339" })?;
    let count = u(&t379, 4 * s6 as usize)?;
    let threads = count.min(groups.saturating_mul(32));
    let (n, extent) = (u(&b3, 0x14)?, f(&b3, 0x44)?);
    let (s4, s5) = (u(&t12, 0x58c)?, u(&t12, 0x590)?);
    if n == 0 || s4 == 0 || s5 == 0 {
        return Err(format!("N {n}, grid {s5} x {s4}"));
    }
    let radius = f32::from_bits(0x3f33_3333) * f(&t12, 0x7c8)?;
    let l = [f(&t10, 0x110)?, f(&t10, 0x114)?, f(&t10, 0x118)?];
    let mut out = Vec::with_capacity(threads as usize);
    for thread in 0..threads {
        let (probe, cascade) = if s6 == 16 {
            let v = u(&ids, 4 * thread as usize)?;
            (v % n, v / n)
        } else {
            let w = u(&ids, 8 * thread as usize)?;
            (w & 0x0fff_ffff, w >> 29)
        };
        let (x, y, z) = (probe % s5, (probe / s5) % s4, probe / (s5 * s4));
        let at = 0x594 + 16 * cascade as usize;
        let o = [f(&t12, at)?, f(&t12, at + 4)?, f(&t12, at + 8)?, f(&t12, at + 12)?];
        let h = f(&t12, 0x5f0 + 16 * cascade as usize)?;
        // As the program: f32, (k + 0.5) / dim, then * 2 - 1, then the fma.
        let norm = |k: u32, dim: u32| ((k as f32 + 0.5) * (1.0 / dim as f32)).mul_add(2.0, -1.0);
        let p = [
            o[3].mul_add(norm(z, s4), o[0]),
            o[3].mul_add(norm(y, s4), o[1]),
            h.mul_add(norm(x, s5), o[2]),
        ];
        let pp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let v8 = if pp < radius * radius {
            let b = -(l[0] * p[0] + l[1] * p[1] + l[2] * p[2]);
            (((radius * radius - pp) + b * b).sqrt() - b).max(0.0)
        } else {
            0.0
        };
        let origin = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
        let far = [
            f64::from((-l[0]).mul_add(v8, p[0])),
            f64::from((-l[1]).mul_add(v8, p[1])),
            f64::from((-l[2]).mul_add(v8, p[2])),
        ];
        out.push(Ray {
            thread,
            probe,
            cascade,
            slot: u64::from(cascade) * u64::from(n) + u64::from(probe),
            origin,
            direction: [f64::from(-l[0]), f64::from(-l[1]), f64::from(-l[2])],
            extent: f64::from(extent),
            far_origin: far,
            far_extent: f64::from(extent - v8),
        });
    }
    Ok(out)
}

/// What a ray meets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Hit,
    /// Only within [`EDGE`] of a triangle's edge or of the extent's ends.
    Edge,
    Miss,
}

/// A triangle met: where on the ray, in which instance, its node's pointer
/// in the bottom level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Met {
    pub t: f64,
    pub instance: u32,
    pub node: u32,
}

/// f64 Moeller-Trumbore, both sides: (t, the least barycentric coordinate).
fn triangle(v: &[V3; 3], o: V3, d: V3) -> Option<(f64, f64)> {
    let (e1, e2) = (sub(v[1], v[0]), sub(v[2], v[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    let scale = (dot(e1, e1) * dot(e2, e2) * dot(d, d)).sqrt();
    if det.is_nan() || det.abs() <= 1e-12 * scale {
        return None;
    }
    let s = sub(o, v[0]);
    let a = dot(s, p) / det;
    let q = cross(s, e1);
    let b = dot(d, q) / det;
    let t = dot(e2, q) / det;
    if !t.is_finite() {
        return None;
    }
    Some((t, a.min(b).min(1.0 - a - b)))
}

/// The triangle's verdict within `[0, extent)`.
fn judge(t: f64, least: f64, extent: f64) -> Verdict {
    let slack = 1e-4 * t.abs().max(1.0);
    if least > EDGE && t > slack && t < extent - slack {
        Verdict::Hit
    } else if least > -EDGE && t > -slack && t < extent + slack {
        Verdict::Edge
    } else {
        Verdict::Miss
    }
}

/// Whether the ray meets the box within `[0, extent]`, the box grown by
/// `grow` of its size and place.
fn slab(lo: V3, hi: V3, o: V3, d: V3, extent: f64, grow: f64) -> bool {
    let (mut near, mut far) = (0.0f64, extent);
    for a in 0..3 {
        let pad = grow * (hi[a] - lo[a]).abs().max(lo[a].abs()).max(hi[a].abs()).max(1e-6);
        let (l, h) = (lo[a] - pad, hi[a] + pad);
        if l.is_nan() || h.is_nan() || l > h {
            return false;
        }
        if d[a] == 0.0 {
            if o[a] < l || o[a] > h {
                return false;
            }
            continue;
        }
        let (mut x, mut y) = ((l - o[a]) / d[a], (h - o[a]) / d[a]);
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        near = near.max(x);
        far = far.min(y);
    }
    near <= far
}

/// A box: min, max.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub lo: V3,
    pub hi: V3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3] };

    pub fn add(&mut self, p: V3) {
        for a in 0..3 {
            self.lo[a] = self.lo[a].min(p[a]);
            self.hi[a] = self.hi[a].max(p[a]);
        }
    }
    pub fn join(&mut self, o: &Aabb) {
        if o.proper() {
            self.add(o.lo);
            self.add(o.hi);
        }
    }
    /// Finite, min <= max.
    pub fn proper(&self) -> bool {
        (0..3).all(|a| self.lo[a].is_finite() && self.hi[a].is_finite() && self.lo[a] <= self.hi[a])
    }
    /// By how much `inner` sticks out, over the larger of the box's size and
    /// place (0: held).
    pub fn short(&self, inner: &Aabb) -> f64 {
        let mut worst = 0.0f64;
        for a in 0..3 {
            let scale = (self.hi[a] - self.lo[a]).abs().max(self.lo[a].abs()).max(self.hi[a].abs()).max(1e-6);
            worst = worst.max((self.lo[a] - inner.lo[a]) / scale).max((inner.hi[a] - self.hi[a]) / scale);
        }
        worst
    }
}

/// The triangles of one bottom level with a tree of the reference's own
/// over them (median splits; the structure's boxes play no part).
pub struct Bottom {
    pub base: u64,
    pub root: u32,
    /// The header's marks (dword 0 >> 16).
    pub marks: u32,
    /// Vertices, the node's pointer.
    pub triangles: Vec<([V3; 3], u32)>,
    /// Each node's parent (pointer) and slot there; the root has none.
    pub parents: HashMap<u32, (u32, usize)>,
    pub bounds: Aabb,
    own: Vec<Own>,
    order: Vec<u32>,
    /// What the pointers gave besides: nodes reached again, pointers of
    /// type 6 or 7, nodes not held by the capture.
    pub again: u32,
    pub other: u32,
    pub unread: u32,
}

struct Own {
    bounds: Aabb,
    /// Children, or `first..first + count` of `order`.
    children: Option<(u32, u32)>,
    first: u32,
    count: u32,
}

/// A structure's node address.
fn node(base: u64, pointer: u32) -> u64 {
    base + u64::from((pointer & 0x1fff_ffff) >> 3) * 64
}

/// A box node's child boxes (`pointer`'s type 4: f16, else f32).
fn boxes<M: Memory>(m: &M, base: u64, pointer: u32) -> Option<([u32; 4], [Aabb; 4])> {
    let at = node(base, pointer);
    let children: [u32; 4] = m.words(at)?;
    let mut out = [Aabb::EMPTY; 4];
    if pointer & 7 == 4 {
        let w: [u32; 12] = m.words(at + 16)?;
        for (i, b) in out.iter_mut().enumerate() {
            let h = |j: usize| half((w[3 * i + j / 2] >> (16 * (j % 2))) & 0xffff);
            *b = Aabb { lo: [h(0), h(1), h(2)], hi: [h(3), h(4), h(5)] };
        }
    } else {
        let w: [u32; 24] = m.words(at + 16)?;
        for (i, b) in out.iter_mut().enumerate() {
            let f = |j: usize| float(w[6 * i + j]);
            *b = Aabb { lo: [f(0), f(1), f(2)], hi: [f(3), f(4), f(5)] };
        }
    }
    Some((children, out))
}

/// The triangles a triangle pointer stands for: its own, and for type 0
/// with bit 7 of the node's dword 12 the node's type-1 one.
fn triangles_of<M: Memory>(m: &M, base: u64, pointer: u32) -> Option<Vec<[V3; 3]>> {
    let w: [u32; 15] = m.words(node(base, pointer))?;
    let v = |k: usize| [float(w[3 * k]), float(w[3 * k + 1]), float(w[3 * k + 2])];
    let ty = (pointer & 7) as usize;
    let mut out = vec![TRIANGLE[ty].map(v)];
    if ty == 0 && w[12] & 0x80 != 0 {
        out.push(TRIANGLE[1].map(v));
    }
    Some(out)
}

impl Bottom {
    /// The bottom level at `base` walked from `root` over its pointers.
    pub fn read<M: Memory>(m: &M, base: u64, root: u32) -> Bottom {
        let mut b = Bottom {
            base,
            root,
            marks: m.word(base).map_or(0, |w| w >> 16),
            triangles: Vec::new(),
            parents: HashMap::new(),
            bounds: Aabb::EMPTY,
            own: Vec::new(),
            order: Vec::new(),
            again: 0,
            other: 0,
            unread: 0,
        };
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(p) = stack.pop() {
            if p == NONE {
                continue;
            }
            if !seen.insert(p & 0x1fff_ffff) || seen.len() > MOST_NODES {
                b.again += 1;
                continue;
            }
            match p & 7 {
                4 | 5 => match boxes(m, base, p) {
                    Some((children, _)) => {
                        for (k, &c) in children.iter().enumerate() {
                            if c != NONE {
                                b.parents.entry(c & 0x1fff_ffff).or_insert((p & 0x1fff_ffff, k));
                                stack.push(c);
                            }
                        }
                    }
                    None => b.unread += 1,
                },
                6 | 7 => b.other += 1,
                _ => match triangles_of(m, base, p) {
                    Some(list) => {
                        for t in list {
                            if t.iter().flatten().all(|x| x.is_finite()) {
                                for v in t {
                                    b.bounds.add(v);
                                }
                                b.triangles.push((t, p & 0x1fff_ffff));
                            }
                        }
                    }
                    None => b.unread += 1,
                },
            }
        }
        b.order = (0..b.triangles.len() as u32).collect();
        if !b.triangles.is_empty() {
            let n = b.order.len();
            b.split(0, n);
        }
        b
    }

    /// Makes the node over `order[first..first + count]`, returns its index.
    fn split(&mut self, first: usize, count: usize) -> u32 {
        let mut bounds = Aabb::EMPTY;
        let mut centres = Aabb::EMPTY;
        for &i in &self.order[first..first + count] {
            let t = &self.triangles[i as usize].0;
            for v in t {
                bounds.add(*v);
            }
            centres.add([0, 1, 2].map(|a| (t[0][a] + t[1][a] + t[2][a]) / 3.0));
        }
        let index = self.own.len() as u32;
        self.own.push(Own { bounds, children: None, first: first as u32, count: count as u32 });
        if count > 4 {
            let axis = (0..3)
                .max_by(|&a, &b| (centres.hi[a] - centres.lo[a]).total_cmp(&(centres.hi[b] - centres.lo[b])))
                .unwrap_or(0);
            let tris = &self.triangles;
            let centre = |i: u32| {
                let t = &tris[i as usize].0;
                t[0][axis] + t[1][axis] + t[2][axis]
            };
            self.order[first..first + count].sort_by(|&a, &b| centre(a).total_cmp(&centre(b)));
            let half = count / 2;
            let left = self.split(first, half);
            let right = self.split(first + half, count - half);
            self.own[index as usize].children = Some((left, right));
        }
        index
    }

    /// Any triangle the ray meets within `[0, extent)`: a sure one if there
    /// is one, and whether one was met at an edge.
    pub fn meet(&self, o: V3, d: V3, extent: f64) -> (Option<(f64, u32)>, bool) {
        let mut edge = false;
        if self.own.is_empty() {
            return (None, false);
        }
        let mut stack = vec![0u32];
        while let Some(i) = stack.pop() {
            let n = &self.own[i as usize];
            if !slab(n.bounds.lo, n.bounds.hi, o, d, extent, 1e-3) {
                continue;
            }
            if let Some((l, r)) = n.children {
                stack.push(l);
                stack.push(r);
                continue;
            }
            for &k in &self.order[n.first as usize..(n.first + n.count) as usize] {
                let (v, pointer) = &self.triangles[k as usize];
                if let Some((t, least)) = triangle(v, o, d) {
                    match judge(t, least, extent) {
                        Verdict::Hit => return (Some((t, *pointer)), edge),
                        Verdict::Edge => edge = true,
                        Verdict::Miss => {}
                    }
                }
            }
        }
        (None, edge)
    }

    /// The nearest triangle the ray meets within `(0, extent)`: where, and
    /// the triangle's normal (stored space, not of length 1).
    pub fn nearest(&self, o: V3, d: V3, extent: f64) -> Option<(f64, V3)> {
        let mut best: Option<(f64, V3)> = None;
        if self.own.is_empty() {
            return None;
        }
        let mut stack = vec![0u32];
        while let Some(i) = stack.pop() {
            let n = &self.own[i as usize];
            let reach = best.map_or(extent, |(t, _)| t);
            if !slab(n.bounds.lo, n.bounds.hi, o, d, reach, 1e-6) {
                continue;
            }
            if let Some((l, r)) = n.children {
                stack.push(l);
                stack.push(r);
                continue;
            }
            for &k in &self.order[n.first as usize..(n.first + n.count) as usize] {
                let v = &self.triangles[k as usize].0;
                if let Some((t, least)) = triangle(v, o, d)
                    && least >= 0.0
                    && t > 0.0
                    && t < best.map_or(extent, |(t, _)| t)
                {
                    best = Some((t, cross(sub(v[1], v[0]), sub(v[2], v[0]))));
                }
            }
        }
        best
    }
}

/// An instance of a top level.
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub index: u32,
    /// Bit 30 of dword 12: the program enters it.
    pub enter: bool,
    /// The bottom level: its base (dword 15 << 8) and root (dword 14 &
    /// 0x3fffffff; [`NONE`] for 0x3fffffff).
    pub base: u64,
    pub root: u32,
    /// The rows of `M` (world to stored), none with bit 31 of dword 14.
    pub rows: Option<[f64; 9]>,
    pub translation: V3,
    /// Dword 29: the parent's pointer.
    pub parent: u32,
    /// Dwords 0-11.
    pub first: [u32; 12],
}

impl Instance {
    /// The ray in the bottom level's space.
    pub fn stored(&self, o: V3, d: V3) -> (V3, V3) {
        let t = self.translation;
        match &self.rows {
            Some(m) => {
                let row = |r: usize, v: V3| m[3 * r] * v[0] + m[3 * r + 1] * v[1] + m[3 * r + 2] * v[2];
                ([row(0, o) + t[0], row(1, o) + t[1], row(2, o) + t[2]], [row(0, d), row(1, d), row(2, d)])
            }
            None => ([o[0] + t[0], o[1] + t[1], o[2] + t[2]], d),
        }
    }

    /// A stored point in the world (`M` inverted); none if `M` has no
    /// inverse.
    pub fn world(&self, p: V3) -> Option<V3> {
        let q = sub(p, self.translation);
        let Some(m) = &self.rows else { return Some(q) };
        let (a, b, c) = ([m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]);
        let det = dot(a, cross(b, c));
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        // Columns of the inverse: b x c, c x a, a x b over det.
        let (x, y, z) = (cross(b, c), cross(c, a), cross(a, b));
        Some([0, 1, 2].map(|k| (x[k] * q[0] + y[k] * q[1] + z[k] * q[2]) / det))
    }
}

/// Why the tree loses a triangle the ray meets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lost {
    /// No pointer leads from the top level's root to the instance.
    InstanceNotReached,
    /// A box above the instance's leaf slot misses the ray.
    TopBox,
    /// The instance's own box (its slot in its parent) misses the ray.
    InstanceBox,
    /// A box of the bottom level misses the ray.
    BottomBox,
    /// Every box on the way holds the ray.
    Nothing,
}

/// A top level with its instances and their bottom levels.
pub struct Scene<'m, M: Memory> {
    pub m: &'m M,
    pub base: u64,
    pub header: [u32; 16],
    pub instances: Vec<Instance>,
    /// By (base, root).
    pub bottoms: HashMap<(u64, u32), Bottom>,
    /// Each top-level node's parent (pointer) and slot there.
    pub parents: HashMap<u32, (u32, usize)>,
    /// How often the pointers reach each instance.
    pub reached: Vec<u32>,
    pub again: u32,
    pub unread: u32,
}

impl<'m, M: Memory> Scene<'m, M> {
    /// The top level whose view names `address` (the program takes
    /// `address & !0xff` as its base), or why it cannot be read.
    pub fn read(m: &'m M, address: u64) -> Result<Scene<'m, M>, String> {
        let base = address & !0xff;
        let header: [u32; 16] = m.words(base).ok_or_else(|| format!("the top level's header at {base:#x} is not held"))?;
        let (count, list) = (header[3], u64::from(header[13]));
        if count > 1 << 20 {
            return Err(format!("{count} instances at {base:#x}"));
        }
        let mut s = Scene {
            m,
            base,
            header,
            instances: Vec::new(),
            bottoms: HashMap::new(),
            parents: HashMap::new(),
            reached: vec![0; count as usize],
            again: 0,
            unread: 0,
        };
        for index in 0..count {
            let e: [u32; 32] = m
                .words(base + list + 128 * u64::from(index))
                .ok_or_else(|| format!("instance {index} of {base:#x} is not held"))?;
            let root = e[14] & 0x3fff_ffff;
            s.instances.push(Instance {
                index,
                enter: e[12] >> 30 & 1 == 1,
                base: u64::from(e[15]) << 8,
                root: if root == 0x3fff_ffff { NONE } else { root },
                rows: (e[14] >> 31 == 0).then(|| std::array::from_fn(|k| float(e[19 + k]))),
                translation: [float(e[16]), float(e[17]), float(e[18])],
                parent: e[29],
                first: std::array::from_fn(|k| e[k]),
            });
        }
        // The pointers from the root.
        let mut seen = HashSet::new();
        let mut stack = vec![header[12]];
        while let Some(p) = stack.pop() {
            if p == NONE {
                continue;
            }
            let ty = p & 7;
            if ty == 6 {
                let off = node(base, p).wrapping_sub(base + list);
                if off % 128 == 0 && off / 128 < u64::from(count) {
                    s.reached[(off / 128) as usize] += 1;
                }
                continue;
            }
            if !seen.insert(p & 0x1fff_ffff) || seen.len() > MOST_NODES {
                s.again += 1;
                continue;
            }
            if ty == 4 || ty == 5 {
                match boxes(m, base, p) {
                    Some((children, _)) => {
                        for (k, &c) in children.iter().enumerate() {
                            if c != NONE {
                                s.parents.entry(c & 0x1fff_ffff).or_insert((p & 0x1fff_ffff, k));
                                stack.push(c);
                            }
                        }
                    }
                    None => s.unread += 1,
                }
            }
        }
        for i in 0..s.instances.len() {
            let (base, root) = (s.instances[i].base, s.instances[i].root);
            if s.instances[i].enter && root != NONE {
                s.bottoms.entry((base, root)).or_insert_with(|| Bottom::read(m, base, root));
            }
        }
        Ok(s)
    }

    /// The pointer of instance `index`'s node.
    pub fn leaf(&self, index: u32) -> u32 {
        ((self.header[13] + 128 * index) >> 3) | 6
    }

    /// The triangles' verdict, with a sure hit if there is one.
    pub fn brute(&self, o: V3, d: V3, extent: f64) -> (Verdict, Option<Met>) {
        let mut edge = false;
        for i in self.instances.iter().filter(|i| i.enter) {
            let Some(b) = self.bottoms.get(&(i.base, i.root)) else { continue };
            let (so, sd) = i.stored(o, d);
            let (met, e) = b.meet(so, sd, extent);
            edge |= e;
            if let Some((t, pointer)) = met {
                return (Verdict::Hit, Some(Met { t, instance: i.index, node: pointer }));
            }
        }
        (if edge { Verdict::Edge } else { Verdict::Miss }, None)
    }

    /// The nearest triangle of the instances the program enters within
    /// `(0, extent)`: where, in which instance, and its normal in the world
    /// (not of length 1; the rows of `M` transposed times the stored one).
    pub fn nearest(&self, o: V3, d: V3, extent: f64) -> Option<(f64, u32, V3)> {
        let mut best: Option<(f64, u32, V3)> = None;
        for i in self.instances.iter().filter(|i| i.enter) {
            let Some(b) = self.bottoms.get(&(i.base, i.root)) else { continue };
            let (so, sd) = i.stored(o, d);
            if let Some((t, n)) = b.nearest(so, sd, best.map_or(extent, |(t, _, _)| t)) {
                let world = match &i.rows {
                    Some(m) => [0, 1, 2].map(|c| m[c] * n[0] + m[3 + c] * n[1] + m[6 + c] * n[2]),
                    None => n,
                };
                best = Some((t, i.index, world));
            }
        }
        best
    }

    /// The tree's verdict: the structures' own boxes decide what is
    /// reached. Also the deepest the walk's stack got (every hit child
    /// pushed, in slot order: a measure, not the program's own depth).
    pub fn walk(&self, o: V3, d: V3, extent: f64) -> (Verdict, usize) {
        let mut edge = false;
        let mut deepest = 0usize;
        let mut visited = 0usize;
        // Pointer, level (none: the top one).
        let mut stack: Vec<(u32, Option<usize>)> = vec![(self.header[12], None)];
        while let Some((p, level)) = stack.pop() {
            visited += 1;
            if visited > MOST_VISITS {
                break;
            }
            if p == NONE {
                continue;
            }
            let (base, ro, rd) = match level {
                None => (self.base, o, d),
                Some(i) => {
                    let (so, sd) = self.instances[i].stored(o, d);
                    (self.instances[i].base, so, sd)
                }
            };
            match p & 7 {
                4 | 5 => {
                    let Some((children, b)) = boxes(self.m, base, p) else { continue };
                    for (&child, b) in children.iter().zip(&b) {
                        if child != NONE && slab(b.lo, b.hi, ro, rd, extent, 1e-5) {
                            stack.push((child, level));
                        }
                    }
                    deepest = deepest.max(stack.len());
                }
                6 if level.is_none() => {
                    let off = node(base, p).wrapping_sub(self.base + u64::from(self.header[13]));
                    let index = (off / 128) as usize;
                    if off % 128 == 0 && index < self.instances.len() && self.instances[index].enter {
                        stack.push((self.instances[index].root, Some(index)));
                        deepest = deepest.max(stack.len());
                    }
                }
                6 | 7 => {}
                _ => {
                    for v in triangles_of(self.m, base, p).unwrap_or_default() {
                        if let Some((t, least)) = triangle(&v, ro, rd) {
                            match judge(t, least, extent) {
                                Verdict::Hit => return (Verdict::Hit, deepest),
                                Verdict::Edge => edge = true,
                                Verdict::Miss => {}
                            }
                        }
                    }
                }
            }
        }
        (if edge { Verdict::Edge } else { Verdict::Miss }, deepest)
    }

    /// Where the tree loses `met` for the ray: the boxes on the way from
    /// the root to the triangle's node, top down, tested up to the hit.
    pub fn lost(&self, o: V3, d: V3, met: &Met) -> Lost {
        let i = &self.instances[met.instance as usize];
        let reach = met.t * (1.0 + 1e-4) + 1e-4;
        if self.reached.get(met.instance as usize).copied().unwrap_or(0) == 0 {
            return Lost::InstanceNotReached;
        }
        // The top level's boxes, from the leaf up; judged top down.
        let mut first = Lost::Nothing;
        let mut at = self.leaf(i.index) & 0x1fff_ffff;
        let mut steps = 0;
        while let Some(&(parent, slot)) = self.parents.get(&at) {
            if let Some((_, b)) = boxes(self.m, self.base, parent)
                && !slab(b[slot].lo, b[slot].hi, o, d, reach, 1e-5)
            {
                first = if at == self.leaf(i.index) & 0x1fff_ffff { Lost::InstanceBox } else { Lost::TopBox };
            }
            at = parent;
            steps += 1;
            if steps > 4096 {
                break;
            }
        }
        if first != Lost::Nothing {
            return first;
        }
        let Some(b) = self.bottoms.get(&(i.base, i.root)) else { return Lost::Nothing };
        let (so, sd) = i.stored(o, d);
        let mut at = met.node;
        let mut steps = 0;
        while let Some(&(parent, slot)) = b.parents.get(&at) {
            if let Some((_, bx)) = boxes(self.m, i.base, parent)
                && !slab(bx[slot].lo, bx[slot].hi, so, sd, reach, 1e-5)
            {
                return Lost::BottomBox;
            }
            at = parent;
            steps += 1;
            if steps > 4096 {
                break;
            }
        }
        Lost::Nothing
    }

    /// What an instance's bottom level is like (for the instances whose
    /// box is empty or too small): its marks, whether its root's child
    /// boxes are written, whether it has triangles.
    fn describe(&self, i: &Instance) -> String {
        let marks = self.m.word(i.base).map_or("header not held".to_string(), |w| format!("marks {:#x}", w >> 16));
        let root = match boxes(self.m, i.base, i.root) {
            _ if i.root == NONE => "no root",
            None => "root not held",
            Some((children, b)) => {
                let named: Vec<&Aabb> = children.iter().zip(&b).filter(|(c, _)| **c != NONE).map(|(_, b)| b).collect();
                if named.is_empty() {
                    "root without children"
                } else if named.iter().all(|b| b.proper() && b.lo != b.hi) {
                    "root boxes written"
                } else if named.iter().all(|b| b.lo == [0.0; 3] && b.hi == [0.0; 3]) {
                    "root boxes all zero"
                } else {
                    "root boxes partly written"
                }
            }
        };
        let triangles = self.bottoms.get(&(i.base, i.root)).map_or(0, |b| b.triangles.len());
        format!("{marks}, {root}, {}", if triangles > 0 { "triangles" } else { "no triangles" })
    }

    /// The check without rays.
    pub fn check(&self) -> Check {
        let mut c = Check { instances: self.instances.len() as u32, ..Check::default() };
        c.entered = self.instances.iter().filter(|i| i.enter).count() as u32;
        c.never = self.reached.iter().zip(&self.instances).filter(|(n, i)| **n == 0 && (i.base != 0 || i.enter)).count() as u32;
        c.twice = self.reached.iter().filter(|n| **n > 1).count() as u32;
        c.top_again = self.again;
        c.unread = self.unread;
        // What each instance holds, in the world.
        let world: Vec<Option<Aabb>> = self
            .instances
            .iter()
            .map(|i| {
                let b = self.bottoms.get(&(i.base, i.root))?;
                let mut w = Aabb::EMPTY;
                for (t, _) in &b.triangles {
                    for v in t {
                        w.add(i.world(*v)?);
                    }
                }
                w.proper().then_some(w)
            })
            .collect();
        // The top level's boxes, bottom up over the pointers.
        let mut holds: HashMap<u32, Aabb> = HashMap::new();
        let mut order = Vec::new();
        let mut stack = vec![self.header[12]];
        let mut seen = HashSet::new();
        while let Some(p) = stack.pop() {
            if p == NONE || p & 6 != 4 || !seen.insert(p & 0x1fff_ffff) {
                continue;
            }
            order.push(p);
            if let Some((children, _)) = boxes(self.m, self.base, p) {
                stack.extend(children);
            }
        }
        let table = |slot: u32, base: u64, header: &[u32; 16]| -> Option<u32> {
            let index = u64::from(slot).wrapping_sub(u64::from(header[13] >> 6));
            self.m.word(base.wrapping_add(u64::from(header[14])).wrapping_add(index.wrapping_mul(4)))
        };
        for &p in order.iter().rev() {
            let Some((children, b)) = boxes(self.m, self.base, p) else { continue };
            let mut all = Aabb::EMPTY;
            for k in 0..4 {
                let child = children[k];
                if child == NONE {
                    continue;
                }
                all.join(&b[k]);
                if child & 7 == 6 {
                    let off = node(self.base, child).wrapping_sub(self.base + u64::from(self.header[13]));
                    let Some(i) = self.instances.get((off / 128) as usize).filter(|_| off % 128 == 0) else {
                        c.top_outside += 1;
                        continue;
                    };
                    if i.parent & 0x1fff_ffff == p & 0x1fff_ffff {
                        c.parent_instance.0 += 1;
                    } else {
                        c.parent_instance.1 += 1;
                    }
                    if !i.enter {
                        continue;
                    }
                    if !b[k].proper() || world[i.index as usize].as_ref().is_some_and(|w| b[k].short(w) > 1e-3) {
                        *c.unsound_instances.entry(self.describe(i)).or_default() += 1;
                    }
                    match &world[i.index as usize] {
                        _ if !b[k].proper() => c.instance_box.empty += 1,
                        Some(w) => c.instance_box.count(b[k].short(w), || format!("instance {} of {:#x}", i.index, self.base)),
                        None => c.instance_box.unknown += 1,
                    }
                } else if child & 6 == 4 {
                    match table(child >> 3 & 0x03ff_ffff, self.base, &self.header) {
                        Some(w) if w & 0x1fff_ffff == p & 0x1fff_ffff => c.parent_top.0 += 1,
                        _ => c.parent_top.1 += 1,
                    }
                    match holds.get(&(child & 0x1fff_ffff)) {
                        _ if !b[k].proper() => c.top_box.empty += 1,
                        Some(inner) if inner.proper() => c.top_box.count(b[k].short(inner), || format!("node {child:#x} of {:#x}", self.base)),
                        _ => c.top_box.unknown += 1,
                    }
                }
            }
            holds.insert(p & 0x1fff_ffff, all);
        }
        // The bottom levels the program enters.
        for b in self.bottoms.values() {
            c.bottoms += 1;
            c.triangles += b.triangles.len() as u64;
            c.bottom_again += b.again;
            c.unread += b.unread;
            if b.marks & 0x100 != 0 {
                c.packed += 1;
            }
            if b.triangles.is_empty() {
                c.without_triangles += 1;
            }
            let header: Option<[u32; 16]> = self.m.words(b.base);
            let mut order = Vec::new();
            let mut stack = vec![b.root];
            let mut seen = HashSet::new();
            while let Some(p) = stack.pop() {
                if p == NONE || p & 6 != 4 || !seen.insert(p & 0x1fff_ffff) {
                    continue;
                }
                order.push(p);
                if let Some((children, _)) = boxes(self.m, b.base, p) {
                    stack.extend(children);
                }
            }
            let mut holds: HashMap<u32, Aabb> = HashMap::new();
            for &p in order.iter().rev() {
                let Some((children, bx)) = boxes(self.m, b.base, p) else { continue };
                let mut all = Aabb::EMPTY;
                for k in 0..4 {
                    let child = children[k];
                    if child == NONE {
                        continue;
                    }
                    all.join(&bx[k]);
                    let inner = if child & 6 == 4 {
                        if let Some(h) = &header {
                            match table(child >> 3 & 0x03ff_ffff, b.base, h) {
                                Some(w) if w & 0x1fff_ffff == p & 0x1fff_ffff => c.parent_bottom.0 += 1,
                                _ => c.parent_bottom.1 += 1,
                            }
                        }
                        holds.get(&(child & 0x1fff_ffff)).copied()
                    } else if child & 4 == 0 {
                        triangles_of(self.m, b.base, child).map(|list| {
                            let mut w = Aabb::EMPTY;
                            for v in list.iter().flatten() {
                                w.add(*v);
                            }
                            w
                        })
                    } else {
                        None
                    };
                    let set = if child & 4 == 0 { &mut c.leaf_box } else { &mut c.bottom_box };
                    match inner {
                        _ if !bx[k].proper() => set.empty += 1,
                        Some(inner) if inner.proper() => set.count(bx[k].short(&inner), || format!("node {child:#x} of {:#x}", b.base)),
                        _ => set.unknown += 1,
                    }
                }
                holds.insert(p & 0x1fff_ffff, all);
            }
        }
        // Transformations: `M` (world to stored) times the 3 x 3 of dwords
        // 0-11 (rows of three and a fourth: object to world) is the bottom
        // level's own scale, one positive factor an axis (onCap3: 8192,
        // 524288, 8192): nothing off the diagonal.
        for i in self.instances.iter().filter(|i| i.enter) {
            let Some(m) = &i.rows else {
                c.transform.2 += 1;
                continue;
            };
            let a: [f64; 9] = std::array::from_fn(|k| float(i.first[4 * (k / 3) + k % 3]));
            let mut p = [0.0f64; 9];
            for r in 0..3 {
                for col in 0..3 {
                    p[3 * r + col] = (0..3).map(|k| m[3 * r + k] * a[3 * k + col]).sum();
                }
            }
            let diagonal = (0..3).all(|r| {
                let own = p[4 * r];
                own.is_finite() && own > 0.0 && (0..3).all(|col| col == r || p[3 * r + col].abs() <= 2e-3 * own)
            });
            if diagonal {
                c.transform.0 += 1;
            } else {
                c.transform.1 += 1;
            }
        }
        c
    }
}

/// Boxes against what they hold.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Held {
    pub good: u32,
    /// Sticking out by more than a thousandth.
    pub small: u32,
    /// Not finite, or min above max.
    pub empty: u32,
    /// What is below could not be read.
    pub unknown: u32,
    pub worst: f64,
    pub first: Vec<String>,
}

impl Held {
    fn count(&mut self, short: f64, what: impl Fn() -> String) {
        if short > 1e-3 || short.is_nan() {
            self.small += 1;
            if self.first.len() < 8 {
                self.first.push(format!("{} (by {short:.3})", what()));
            }
        } else {
            self.good += 1;
        }
        if short > self.worst {
            self.worst = short;
        }
    }
    /// Nothing a ray could be lost by.
    pub fn sound(&self) -> bool {
        self.small == 0 && self.empty == 0
    }
}

impl std::fmt::Display for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "good {}, too small {}, empty {}, unknown {} (worst {:.4})",
            self.good, self.small, self.empty, self.unknown, self.worst
        )
    }
}

/// [`Scene::check`]'s counts.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Check {
    pub instances: u32,
    pub entered: u32,
    /// Instances with a bottom level no pointer reaches / reached more than
    /// once.
    pub never: u32,
    pub twice: u32,
    /// Nodes the pointers reach again (a shared node or a cycle).
    pub top_again: u32,
    pub bottom_again: u32,
    /// Leaf pointers of the top level outside its instance list.
    pub top_outside: u32,
    pub unread: u32,
    pub bottoms: u32,
    pub packed: u32,
    pub without_triangles: u32,
    pub triangles: u64,
    /// Inner slots of the top level over their child's boxes.
    pub top_box: Held,
    /// Instance slots over the instance's triangles in the world.
    pub instance_box: Held,
    /// Inner slots of the bottom levels over their child's boxes.
    pub bottom_box: Held,
    /// Leaf slots of the bottom levels over their triangles.
    pub leaf_box: Held,
    /// Parent links that name the parent / do not: box nodes of the top
    /// level, instances, box nodes of the bottom levels.
    pub parent_top: (u32, u32),
    pub parent_instance: (u32, u32),
    pub parent_bottom: (u32, u32),
    /// `M` agrees with dwords 0-11 / does not / has no `M`.
    pub transform: (u32, u32, u32),
    /// The instances whose box is empty or too small, by what their
    /// bottom level is like.
    pub unsound_instances: BTreeMap<String, u32>,
}

impl Check {
    /// No ray can be lost: every instance reached once, no node twice, no
    /// box too small or empty, nothing unread.
    pub fn sound(&self) -> bool {
        self.never == 0
            && self.twice == 0
            && self.top_again == 0
            && self.bottom_again == 0
            && self.top_outside == 0
            && self.unread == 0
            && self.packed == 0
            && self.top_box.sound()
            && self.instance_box.sound()
            && self.bottom_box.sound()
            && self.leaf_box.sound()
    }
}

impl std::fmt::Display for Check {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "  instances {} (entered {}), never reached {}, reached more than once {}, leaf pointers outside the list {}",
            self.instances, self.entered, self.never, self.twice, self.top_outside
        )?;
        writeln!(
            f,
            "  nodes reached again: top level {}, bottom levels {}; nodes not held {}",
            self.top_again, self.bottom_again, self.unread
        )?;
        writeln!(
            f,
            "  bottom levels entered {} (packed {}, without triangles {}), triangles {}",
            self.bottoms, self.packed, self.without_triangles, self.triangles
        )?;
        writeln!(f, "  top-level inner boxes: {}", self.top_box)?;
        writeln!(f, "  instance boxes over their triangles in the world: {}", self.instance_box)?;
        writeln!(f, "  bottom-level inner boxes: {}", self.bottom_box)?;
        writeln!(f, "  bottom-level leaf boxes over their triangles: {}", self.leaf_box)?;
        for (name, h) in [("top", &self.top_box), ("instance", &self.instance_box), ("bottom", &self.bottom_box), ("leaf", &self.leaf_box)] {
            if !h.first.is_empty() {
                writeln!(f, "    first too small ({name}): {}", h.first.join("; "))?;
            }
        }
        writeln!(
            f,
            "  parent links right / wrong: top-level boxes {} / {}, instances {} / {}, bottom-level boxes {} / {}",
            self.parent_top.0, self.parent_top.1, self.parent_instance.0, self.parent_instance.1, self.parent_bottom.0, self.parent_bottom.1
        )?;
        writeln!(
            f,
            "  transformations agreeing with dwords 0-11: {}, not {}, without a matrix {}",
            self.transform.0, self.transform.1, self.transform.2
        )?;
        write!(f, "  instances with an empty or too small box, by bottom level: {:?}", self.unsound_instances)
    }
}

/// Why the game and the triangles disagree on a ray.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// The triangles are met, the tree loses them ([`Lost`]), the game
    /// misses.
    Tree(Lost),
    /// The triangles are met and the tree reaches them, the game misses:
    /// the program's walk or its node tests, or other bytes than captured.
    /// `true`: the reference's walk went deeper than the program's stack.
    Reader(bool),
    /// The game hits what the triangles do not hold.
    GameOnly,
}

/// A ray's three verdicts.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub ray: Ray,
    pub triangles: Verdict,
    pub tree: Verdict,
    /// The game's byte: 0 a hit, 127 none.
    pub game: Option<u8>,
    pub met: Option<(bool, Met)>,
    pub deepest: usize,
    pub class: Option<Class>,
}

/// A dispatch's rays against its two top levels.
pub fn compare<M: Memory>(near: &Scene<'_, M>, far: &Scene<'_, M>, rays: &[Ray], game: &[u8]) -> Vec<Row> {
    let mut rows = Vec::with_capacity(rays.len());
    for r in rays {
        let shoot_far = r.far_extent > 0.0;
        let (bn, mn) = near.brute(r.origin, r.direction, r.extent);
        let (bf, mf) = if bn != Verdict::Hit && shoot_far {
            far.brute(r.far_origin, r.direction, r.far_extent)
        } else {
            (Verdict::Miss, None)
        };
        let (wn, dn) = near.walk(r.origin, r.direction, r.extent);
        let (wf, df) = if wn != Verdict::Hit && shoot_far {
            far.walk(r.far_origin, r.direction, r.far_extent)
        } else {
            (Verdict::Miss, 0)
        };
        let both = |a: Verdict, b: Verdict| match (a, b) {
            (Verdict::Hit, _) | (_, Verdict::Hit) => Verdict::Hit,
            (Verdict::Edge, _) | (_, Verdict::Edge) => Verdict::Edge,
            _ => Verdict::Miss,
        };
        let (triangles, tree) = (both(bn, bf), both(wn, wf));
        let met = mn.map(|m| (false, m)).or(mf.map(|m| (true, m)));
        let byte = game.get(r.slot as usize).copied();
        let class = match (triangles, byte) {
            (Verdict::Hit, Some(127)) => Some(match (tree, &met) {
                (Verdict::Hit, _) | (_, None) => Class::Reader(dn.max(df) > 16),
                (_, Some((false, m))) => Class::Tree(near.lost(r.origin, r.direction, m)),
                (_, Some((true, m))) => Class::Tree(far.lost(r.far_origin, r.direction, m)),
            }),
            (Verdict::Miss, Some(0)) => Some(Class::GameOnly),
            _ => None,
        };
        rows.push(Row { ray: r.clone(), triangles, tree, game: byte, met, deepest: dn.max(df), class });
    }
    rows
}

/// The rows counted.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Summary {
    pub rays: u32,
    /// Rays whose triangles' verdict is sure and whose game byte is 0 or
    /// 127.
    pub sure: u32,
    /// Of those: game and triangles say the same.
    pub same: u32,
    pub edge: u32,
    /// Game bytes that are neither 0 nor 127 (or outside the view).
    pub other_bytes: u32,
    pub game_hits: u32,
    pub triangle_hits: u32,
    /// Sure rays where tree and triangles differ.
    pub tree_differs: u32,
    pub classes: BTreeMap<String, u32>,
    pub deepest: usize,
    pub deeper_than_16: u32,
}

impl Summary {
    pub fn of(rows: &[Row]) -> Summary {
        let mut s = Summary { rays: rows.len() as u32, ..Summary::default() };
        for r in rows {
            s.deepest = s.deepest.max(r.deepest);
            s.deeper_than_16 += u32::from(r.deepest > 16);
            let game = match r.game {
                Some(0) => true,
                Some(127) => false,
                _ => {
                    s.other_bytes += 1;
                    continue;
                }
            };
            s.game_hits += u32::from(game);
            if r.triangles == Verdict::Edge {
                s.edge += 1;
                continue;
            }
            s.sure += 1;
            s.triangle_hits += u32::from(r.triangles == Verdict::Hit);
            s.same += u32::from(game == (r.triangles == Verdict::Hit));
            s.tree_differs += u32::from(r.tree != Verdict::Edge && r.tree != r.triangles);
            if let Some(c) = r.class {
                *s.classes.entry(format!("{c:?}")).or_default() += 1;
            }
        }
        s
    }

    /// Per cent of the sure rays where the game says what the triangles say.
    pub fn agreement(&self) -> f64 {
        if self.sure == 0 { 0.0 } else { 100.0 * f64::from(self.same) / f64::from(self.sure) }
    }

    /// Fable Aufruf 22 point 14: at least 99 % the same on at least 1000
    /// sure rays.
    pub fn accepted(&self) -> bool {
        self.sure >= 1000 && self.same * 100 >= self.sure * 99
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "  rays {}: sure {}, at an edge {}, game bytes neither 0 nor 127 {}",
            self.rays, self.sure, self.edge, self.other_bytes
        )?;
        writeln!(
            f,
            "  game hits {}, triangle hits (sure rays) {}, game = triangles {} of {} = {:.2} %",
            self.game_hits,
            self.triangle_hits,
            self.same,
            self.sure,
            self.agreement()
        )?;
        writeln!(
            f,
            "  tree differs from triangles on {} sure rays; deepest stack {} (deeper than 16: {} rays)",
            self.tree_differs, self.deepest, self.deeper_than_16
        )?;
        write!(f, "  classes of game != triangles: {:?}", self.classes)
    }
}

/// The reader's dispatches of a capture: (dispatch index, `s6`, groups in x,
/// its table views).
pub fn reader_dispatches<'c>(c: &'c Capture, prefix: &str) -> Vec<(usize, u32, u32, Vec<&'c TableView>)> {
    let mut out = Vec::new();
    for (k, d) in c.dispatches.iter().enumerate() {
        let hash = c.pipelines.get(d.pipeline as usize).map_or("", |p| p.hash.as_str());
        if !hash.to_ascii_lowercase().starts_with(&prefix.to_ascii_lowercase()) {
            continue;
        }
        let (s6, groups, views) = dispatch_views(d);
        out.push((k, s6, groups, views));
    }
    out
}

fn dispatch_views(d: &Dispatch) -> (u32, u32, Vec<&TableView>) {
    let mut s6 = 0;
    let mut groups = match &d.groups {
        super::format::Groups::Direct(g) => g[0],
        super::format::Groups::Indirect { .. } => 0,
    };
    let mut views = Vec::new();
    for a in &d.args {
        match a {
            RootArg::Constants { dwords, .. } if dwords.len() > 6 => s6 = dwords[6].value,
            RootArg::Table { views: v, .. } => {
                for view in v {
                    if view.kind == INDIRECT_ARGUMENTS {
                        if let Some(b) = view.pre.get(..4) {
                            groups = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        }
                    } else {
                        views.push(view);
                    }
                }
            }
            _ => {}
        }
    }
    (s6, groups, views)
}

/// A table view's name as the program's register (`t355`, `u52`, `b3`).
pub fn view_name(v: &TableView) -> String {
    format!("{}{}", ["b", "t", "u"].get(usize::from(v.kind)).unwrap_or(&"?"), v.register)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f64) -> u32 {
        (x as f32).to_bits()
    }

    /// A structure being written: slot 0 the header, nodes from slot 1.
    struct Writer<'a> {
        m: &'a mut WordMemory,
        base: u64,
        next: u32,
        /// (child pointer, parent pointer) of box nodes.
        links: Vec<(u32, u32)>,
    }

    impl Writer<'_> {
        fn put(&mut self, at: u64, w: u32) {
            self.m.0.insert(at, w);
        }
        /// A box32 node over `children` (pointer, box).
        fn node(&mut self, children: &[(u32, Aabb)]) -> u32 {
            let slot = self.next;
            self.next += 2;
            let at = self.base + u64::from(slot) * 64;
            let pointer = slot << 3 | 5;
            for k in 0..4 {
                let (child, b) = children.get(k).copied().unwrap_or((NONE, Aabb::EMPTY));
                self.put(at + 4 * k as u64, child);
                if child != NONE {
                    for a in 0..3 {
                        self.put(at + 16 + 24 * k as u64 + 4 * a as u64, bits(b.lo[a]));
                        self.put(at + 28 + 24 * k as u64 + 4 * a as u64, bits(b.hi[a]));
                    }
                    if child & 6 == 4 {
                        self.links.push((child, pointer));
                    }
                }
            }
            pointer
        }
        /// A tree of box32 nodes over `leaves`.
        fn tree(&mut self, leaves: &[(u32, Aabb)]) -> (u32, Aabb) {
            let mut all = Aabb::EMPTY;
            for (_, b) in leaves {
                all.join(b);
            }
            if leaves.len() <= 4 {
                return (self.node(leaves), all);
            }
            let per = leaves.len().div_ceil(4);
            let children: Vec<(u32, Aabb)> = leaves.chunks(per).map(|c| self.tree(c)).collect();
            (self.node(&children), all)
        }
        /// The parent table behind the nodes, the header's size and links.
        fn finish(&mut self, first: u32, root: u32) {
            let table = u64::from(self.next) * 64;
            for (child, parent) in self.links.clone() {
                self.put(self.base + table + 4 * (u64::from(child >> 3) - u64::from(first >> 6)), parent);
            }
            self.put(self.base + 4, self.next + 64);
            self.put(self.base + 48, root);
            self.put(self.base + 52, first);
            self.put(self.base + 56, table as u32);
        }
    }

    /// A bottom level of `quads` (two triangles a node: V0 V1 V2 and V1 V3
    /// V2, bit 7 of dword 12) at `base`; its root.
    fn bottom(m: &mut WordMemory, base: u64, quads: &[[V3; 4]]) -> u32 {
        let mut w = Writer { m, base, next: 1, links: Vec::new() };
        w.put(base, 0x8002_0006);
        let mut leaves = Vec::new();
        for q in quads {
            let slot = w.next;
            w.next += 1;
            let at = base + u64::from(slot) * 64;
            let mut b = Aabb::EMPTY;
            for (k, v) in q.iter().enumerate() {
                b.add(*v);
                for a in 0..3 {
                    w.put(at + 12 * k as u64 + 4 * a as u64, bits(v[a]));
                }
            }
            w.put(at + 48, 0x80);
            leaves.push((slot << 3, b));
        }
        let (root, _) = w.tree(&leaves);
        w.finish(64, root);
        root
    }

    /// What a test scene's instance is made from: the bottom level's base
    /// and root, a scale and a translation (stored = scale * world +
    /// translation), whether the program enters it, whether it has a
    /// matrix, and its box in the world.
    #[derive(Clone)]
    struct Put {
        base: u64,
        root: u32,
        scale: f64,
        translation: V3,
        enter: bool,
        matrix: bool,
        world: Aabb,
    }

    /// A top level of `puts` at `base`.
    fn top(m: &mut WordMemory, base: u64, puts: &[Put]) {
        // The parent table from +0x40, the instances after it, then boxes.
        let list = 0x40 + (4 * (2 * puts.len() as u32 + 64)).next_multiple_of(64);
        let mut w = Writer { m, base, next: (list >> 6) + 2 * puts.len() as u32, links: Vec::new() };
        w.put(base, 6);
        w.put(base + 12, puts.len() as u32);
        let leaves: Vec<(u32, Aabb)> =
            puts.iter().enumerate().map(|(i, p)| ((list + 128 * i as u32) >> 3 | 6, p.world)).collect();
        let (root, _) = w.tree(&leaves);
        // Each instance's parent: the box node naming it.
        let mut parent = HashMap::new();
        let mut stack = vec![root];
        while let Some(p) = stack.pop() {
            for k in 0..4 {
                let c = w.m.0.get(&(node(base, p) + 4 * k)).copied().unwrap_or(NONE);
                if c == NONE {
                    continue;
                }
                if c & 7 == 6 {
                    parent.insert(c, p);
                } else {
                    stack.push(c);
                }
            }
        }
        for (i, p) in puts.iter().enumerate() {
            let at = base + u64::from(list) + 128 * i as u64;
            // Dwords 0-11: stored to world, rows of three and a fourth.
            for r in 0..3 {
                w.put(at + 16 * r as u64 + 4 * r as u64, bits(1.0 / p.scale));
                w.put(at + 16 * r as u64 + 12, bits(-p.translation[r] / p.scale));
            }
            w.put(at + 48, u32::from(p.enter) << 30);
            w.put(at + 56, p.root | u32::from(!p.matrix) << 31);
            w.put(at + 60, (p.base >> 8) as u32);
            for a in 0..3 {
                w.put(at + 64 + 4 * a as u64, bits(p.translation[a]));
                w.put(at + 76 + 16 * a as u64, bits(p.scale));
            }
            w.put(at + 116, parent[&((list + 128 * i as u32) >> 3 | 6)]);
        }
        // The table precedes the list: D13 = list, D14 = 0x40.
        let links = w.links.clone();
        for (child, parent) in links {
            w.put(base + 0x40 + 4 * (u64::from(child >> 3) - u64::from(list >> 6)), parent);
        }
        w.put(base + 4, w.next + 64);
        w.put(base + 48, root);
        w.put(base + 52, list);
        w.put(base + 56, 0x40);
    }

    /// A wall of `n` x `n` unit quads in the plane z = `z`, from (0, 0).
    fn wall(n: usize, z: f64) -> Vec<[V3; 4]> {
        let mut q = Vec::new();
        for x in 0..n {
            for y in 0..n {
                let (x, y) = (x as f64, y as f64);
                q.push([[x, y, z], [x + 1.0, y, z], [x, y + 1.0, z], [x + 1.0, y + 1.0, z]]);
            }
        }
        q
    }

    const TOP: u64 = 0xd_5f20_0000;
    const BOTTOM: u64 = 0x7_0300_0000;

    /// Three instances of one 6 x 6 wall: as stored at z = 5; twice the
    /// size, moved to x 100.. (stored = 0.5 world - (50, 0, 0)); and one
    /// without a matrix, moved by (0, 200, 0).
    fn scene(m: &mut WordMemory) -> Vec<Put> {
        let root = bottom(m, BOTTOM, &wall(6, 5.0));
        let b = |lo: V3, hi: V3| Aabb { lo, hi };
        let puts = vec![
            Put { base: BOTTOM, root, scale: 1.0, translation: [0.0; 3], enter: true, matrix: true, world: b([0.0, 0.0, 5.0], [6.0, 6.0, 5.0]) },
            Put {
                base: BOTTOM,
                root,
                scale: 0.5,
                translation: [-50.0, 0.0, 0.0],
                enter: true,
                matrix: true,
                world: b([100.0, 0.0, 10.0], [112.0, 12.0, 10.0]),
            },
            Put {
                base: BOTTOM,
                root,
                scale: 1.0,
                translation: [0.0, -200.0, 0.0],
                enter: true,
                matrix: false,
                world: b([0.0, 200.0, 5.0], [6.0, 206.0, 5.0]),
            },
        ];
        top(m, TOP, &puts);
        puts
    }

    /// Rays up the z axis from a grid under the three walls and beside them.
    fn grid() -> Vec<(V3, V3, f64)> {
        let mut out = Vec::new();
        for (ox, oy, step) in [(0.0, 0.0, 1.0), (100.0, 0.0, 2.0), (0.0, 200.0, 1.0)] {
            for i in -2..8 {
                for j in -2..8 {
                    let o = [ox + step * (f64::from(i) + 0.3), oy + step * (f64::from(j) + 0.4), 0.0];
                    out.push((o, [0.0, 0.0, 1.0], 1000.0));
                }
            }
        }
        out
    }

    fn counts(s: &Scene<'_, WordMemory>) -> (u32, u32, u32) {
        let (mut hits, mut differ, mut misses) = (0, 0, 0);
        for (o, d, e) in grid() {
            let (b, _) = s.brute(o, d, e);
            let (w, _) = s.walk(o, d, e);
            hits += u32::from(b == Verdict::Hit);
            misses += u32::from(b == Verdict::Miss);
            differ += u32::from(b != w);
        }
        (hits, misses, differ)
    }

    /// A sound scene: 36 rays hit each of the three walls (scaled, moved
    /// and matrix-less alike), the tree says what the triangles say, and
    /// the check finds nothing.
    #[test]
    fn a_sound_scene_is_met_alike_by_triangles_and_tree() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP | 0x40).unwrap();
        assert_eq!(s.instances.len(), 3);
        assert_eq!(s.reached, vec![1, 1, 1]);
        assert_eq!(counts(&s), (108, 192, 0));
        let c = s.check();
        assert!(c.sound(), "{c}");
        assert_eq!((c.bottoms, c.triangles), (1, 72));
        assert_eq!(c.instance_box.good, 3);
        assert_eq!(c.leaf_box.good, 36);
        assert!(c.bottom_box.good > 0 && c.parent_bottom.0 > 0);
        assert_eq!((c.parent_instance, c.parent_bottom.1, c.parent_top.1), ((3, 0), 0, 0));
        assert_eq!(c.transform, (2, 0, 1));
    }

    /// The second triangle of a node counts only with bit 7 of its dword
    /// 12 (the program's rule at 0x9a0): without it half of each quad is
    /// missed by triangles and tree alike.
    #[test]
    fn a_nodes_second_triangle_needs_its_bit() {
        let mut m = WordMemory::default();
        scene(&mut m);
        // In the quad at (3, 3): the first triangle, the second one.
        let (first, second, up) = ([3.3, 3.4, 0.0], [3.7, 3.6, 0.0], [0.0, 0.0, 1.0]);
        let both = |m: &WordMemory, o: V3| {
            let s = Scene::read(m, TOP).unwrap();
            (s.brute(o, up, 1000.0).0, s.walk(o, up, 1000.0).0)
        };
        assert_eq!(both(&m, first), (Verdict::Hit, Verdict::Hit));
        assert_eq!(both(&m, second), (Verdict::Hit, Verdict::Hit));
        let nodes: Vec<u64> = m.0.iter().filter(|&(&a, &w)| a >> 28 == BOTTOM >> 28 && a % 64 == 48 && w == 0x80).map(|(&a, _)| a).collect();
        assert_eq!(nodes.len(), 36);
        for a in nodes {
            m.0.insert(a, 0);
        }
        assert_eq!(both(&m, first), (Verdict::Hit, Verdict::Hit));
        assert_eq!(both(&m, second), (Verdict::Miss, Verdict::Miss));
        assert_eq!(Scene::read(&m, TOP).unwrap().check().triangles, 36);
    }

    /// An instance the program does not enter (bit 30 of dword 12 clear)
    /// is no hit, for triangles and tree alike.
    #[test]
    fn an_instance_without_its_bit_is_not_entered() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let list = u64::from(Scene::read(&m, TOP).unwrap().header[13]);
        m.0.insert(TOP + list + 128 + 48, 0);
        let s = Scene::read(&m, TOP).unwrap();
        assert_eq!(counts(&s), (72, 228, 0));
    }

    /// An instance box that is too small: the check names it, the tree
    /// loses rays the triangles meet, and the loss is put on the instance
    /// box.
    #[test]
    fn a_small_instance_box_is_found_and_classed() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP).unwrap();
        // Instance 1's slot in its parent: max x 112 -> 104.
        let (parent, slot) = s.parents[&(s.leaf(1) & 0x1fff_ffff)];
        let at = node(TOP, parent) + 16 + 24 * slot as u64 + 12;
        assert_eq!(m.0[&at], bits(112.0));
        m.0.insert(at, bits(104.0));
        let s = Scene::read(&m, TOP).unwrap();
        let c = s.check();
        assert!(!c.sound());
        assert_eq!((c.instance_box.small, c.instance_box.good), (1, 2));
        assert!(c.top_box.sound() && c.bottom_box.sound() && c.leaf_box.sound());
        let (o, d) = ([109.3, 3.0, 0.0], [0.0, 0.0, 1.0]);
        let (b, met) = s.brute(o, d, 1000.0);
        assert_eq!((b, s.walk(o, d, 1000.0).0), (Verdict::Hit, Verdict::Miss));
        assert_eq!(s.lost(o, d, &met.unwrap()), Lost::InstanceBox);
        // A ray inside the small box is still met.
        assert_eq!(s.walk([101.3, 3.0, 0.0], d, 1000.0).0, Verdict::Hit);
    }

    /// An inner box of the bottom level set to zeros (as a stage that did
    /// not run leaves it): found by the check, classed as a bottom box.
    #[test]
    fn an_unwritten_bottom_box_is_found_and_classed() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP).unwrap();
        let root = s.instances[0].root;
        // The root's first child box.
        for k in 0..6 {
            m.0.insert(node(BOTTOM, root) + 16 + 4 * k, 0);
        }
        let s = Scene::read(&m, TOP).unwrap();
        let c = s.check();
        assert!(!c.sound() && c.bottom_box.small + c.bottom_box.empty >= 1, "{c}");
        assert!(c.instance_box.sound());
        let mut lost = BTreeSet::new();
        for (o, d, e) in grid() {
            if let (Verdict::Hit, Some(met)) = s.brute(o, d, e)
                && s.walk(o, d, e).0 == Verdict::Miss
            {
                lost.insert(s.lost(o, d, &met));
            }
        }
        assert_eq!(lost.into_iter().collect::<Vec<_>>(), vec![Lost::BottomBox]);
    }

    /// A leaf whose triangle lies outside its slot's box (a vertex moved
    /// after the boxes were made) is found as a leaf box.
    #[test]
    fn a_leaf_outside_its_box_is_found() {
        let mut m = WordMemory::default();
        scene(&mut m);
        // The first triangle node (slot 1): V0.x 0 -> -3.
        assert_eq!(m.0[&(BOTTOM + 64)], bits(0.0));
        m.0.insert(BOTTOM + 64, bits(-3.0));
        let c = Scene::read(&m, TOP).unwrap().check();
        assert_eq!(c.leaf_box.small, 1);
        // And the instance boxes over the moved triangle (instances 0 and 2
        // by 3 of 6, instance 1 by 6 of 112).
        assert_eq!(c.instance_box.small, 3);
    }

    /// A wrong matrix (the rows of another scale than dwords 0-11 say) is
    /// counted; the instance without a matrix ignores what its rows hold.
    #[test]
    fn transformations_are_checked_against_the_first_dwords() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let list = u64::from(Scene::read(&m, TOP).unwrap().header[13]);
        // Instance 0: shear the stored x by world y.
        m.0.insert(TOP + list + 76 + 4, bits(0.25));
        // Instance 2 (no matrix): rubbish rows.
        for k in 0..9 {
            m.0.insert(TOP + list + 256 + 76 + 4 * k, bits(7.0));
        }
        let s = Scene::read(&m, TOP).unwrap();
        assert_eq!(s.check().transform, (1, 1, 1));
        assert_eq!(s.brute([3.3, 203.4, 0.0], [0.0, 0.0, 1.0], 1000.0).0, Verdict::Hit);
    }

    /// A child pointer back to the root (a cycle): the walks end, the
    /// check counts the node reached again.
    #[test]
    fn a_cycle_ends_and_is_counted() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP).unwrap();
        let root = s.instances[0].root;
        let (children, _) = boxes(&m, BOTTOM, root).unwrap();
        let inner = children.into_iter().find(|c| c & 6 == 4).unwrap();
        // The inner node's last child: the root again, in a box the rays
        // below meet.
        m.0.insert(node(BOTTOM, inner) + 12, root);
        for (k, x) in [0.0, 0.0, 5.0, 6.0, 6.0, 5.0].into_iter().enumerate() {
            m.0.insert(node(BOTTOM, inner) + 16 + 72 + 4 * k as u64, bits(x));
        }
        let s = Scene::read(&m, TOP).unwrap();
        let c = s.check();
        assert!(c.bottom_again >= 1 && !c.sound(), "{c}");
        // Under the walls and beside them: both walks end.
        let _ = s.walk([3.3, 3.4, 0.0], [0.0, 0.0, 1.0], 1000.0);
        let (o, d) = ([0.5, 0.5, 5.0], [1.0, 1.0, 0.0]);
        let _ = (s.walk(o, d, 1000.0), s.brute(o, d, 1000.0));
        let met = Met { t: 5.0, instance: 0, node: inner };
        let _ = s.lost([3.3, 3.4, 0.0], [0.0, 0.0, 1.0], &met);
    }

    /// Parent links that do not name the parent are counted (a box node's
    /// table entry, an instance's dword 29).
    #[test]
    fn wrong_parent_links_are_counted() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP).unwrap();
        let before = s.check();
        assert_eq!((before.parent_bottom.1, before.parent_instance.1), (0, 0));
        let h: [u32; 16] = m.words(BOTTOM).unwrap();
        let (children, _) = boxes(&m, BOTTOM, s.instances[0].root).unwrap();
        let inner = children.into_iter().find(|c| c & 6 == 4).unwrap();
        let list = u64::from(s.header[13]);
        m.0.insert(BOTTOM + u64::from(h[14]) + 4 * (u64::from(inner >> 3) - u64::from(h[13] >> 6)), 0x1234 << 3 | 5);
        m.0.insert(TOP + list + 116, 0x4321 << 3 | 5);
        let after = Scene::read(&m, TOP).unwrap().check();
        assert_eq!((after.parent_bottom.1, after.parent_instance.1), (1, 1));
        assert_eq!(after.parent_bottom.0 + 1, before.parent_bottom.0);
    }

    /// The nearest hit: the wall at z 5 before one put behind it, with a
    /// normal along z whatever the instance's scale.
    #[test]
    fn the_nearest_triangle_and_its_normal() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let s = Scene::read(&m, TOP).unwrap();
        for (o, want_t, want_i) in [([3.3, 3.4, 0.0], 5.0, 0), ([109.3, 3.0, 2.0], 8.0, 1), ([3.3, 203.4, 9.0], f64::NAN, 9)] {
            match s.nearest(o, [0.0, 0.0, 1.0], 1000.0) {
                Some((t, i, n)) => {
                    assert!((t - want_t).abs() < 1e-9 && i == want_i, "{t} {i}");
                    assert!(n[0].abs() < 1e-12 && n[1].abs() < 1e-12 && n[2].abs() > 0.1, "{n:?}");
                }
                None => assert!(want_t.is_nan()),
            }
        }
        // Looking down from above wall 0: met from the other side at 4.
        assert_eq!(s.nearest([3.3, 3.4, 9.0], [0.0, 0.0, -1.0], 1000.0).map(|h| h.0), Some(4.0));
        assert_eq!(s.nearest([3.3, 3.4, 9.0], [0.0, 0.0, -1.0], 3.9), None);
    }

    /// Triangles: both sides count, the extent's end does not, an edge is
    /// neither in nor out.
    #[test]
    fn triangle_verdicts() {
        let v = [[0.0, 0.0, 5.0], [4.0, 0.0, 5.0], [0.0, 4.0, 5.0]];
        let up = [0.0, 0.0, 1.0];
        let j = |o: V3, d: V3, e: f64| triangle(&v, o, d).map_or(Verdict::Miss, |(t, l)| judge(t, l, e));
        assert_eq!(j([1.0, 1.0, 0.0], up, 1000.0), Verdict::Hit);
        assert_eq!(j([1.0, 1.0, 9.0], [0.0, 0.0, -1.0], 1000.0), Verdict::Hit);
        assert_eq!(j([1.0, 1.0, 0.0], up, 4.9), Verdict::Miss);
        assert_eq!(j([1.0, 1.0, 6.0], up, 1000.0), Verdict::Miss);
        assert_eq!(j([5.0, 1.0, 0.0], up, 1000.0), Verdict::Miss);
        assert_eq!(j([2.0, 2.0, 0.0], up, 1000.0), Verdict::Edge);
        assert_eq!(j([1.0, 1.0, 0.0], up, 5.0), Verdict::Edge);
    }

    /// f16 boxes read as the lowering's sources say: pairs minx|miny,
    /// minz|maxx, maxy|maxz.
    #[test]
    fn half_boxes() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(0x7c00), f64::INFINITY);
        assert_eq!(half(0x0001), (2f64).powi(-24));
        let mut m = WordMemory::default();
        let at = 0x1000u64;
        m.0.insert(at, 0x10);
        for k in 1..4 {
            m.0.insert(at + 4 * k, NONE);
        }
        // min (1, -2, 0.5), max (2, 4, 8).
        m.0.insert(at + 16, 0x3c00 | 0xc000 << 16);
        m.0.insert(at + 20, 0x3800 | 0x4000 << 16);
        m.0.insert(at + 24, 0x4400 | 0x4800 << 16);
        let (children, b) = boxes(&m, 0, (0x1000 >> 3) | 4).unwrap();
        assert_eq!(children[0], 0x10);
        assert_eq!(b[0], Aabb { lo: [1.0, -2.0, 0.5], hi: [2.0, 4.0, 8.0] });
    }

    /// The rays as the program makes them (0x13c-0x340).
    #[test]
    fn rays_as_the_program_makes_them() {
        let mut t12 = vec![0u8; 2044];
        let mut t10 = vec![0u8; 340];
        let mut b3 = vec![0u8; 560];
        let mut t379 = vec![0u8; 88];
        let mut t339 = vec![0u8; 8 * 64];
        let put = |b: &mut [u8], o: usize, w: u32| b[o..o + 4].copy_from_slice(&w.to_le_bytes());
        let putf = |b: &mut [u8], o: usize, x: f32| b[o..o + 4].copy_from_slice(&x.to_le_bytes());
        put(&mut t12, 0x58c, 4);
        put(&mut t12, 0x590, 8);
        putf(&mut t12, 0x7c8, 100.0);
        for (k, x) in [1.0f32, 2.0, 3.0, 16.0].into_iter().enumerate() {
            putf(&mut t12, 0x594 + 32 + 4 * k, x);
        }
        putf(&mut t12, 0x5f0 + 32, 32.0);
        putf(&mut t10, 0x114, -1.0);
        put(&mut b3, 0x14, 32768);
        putf(&mut b3, 0x44, 1000.0);
        put(&mut t379, 60, 3);
        // Probe x 5, y 2, z 1: 5 + 8 * (2 + 4 * 1) = 53, cascade 2.
        for thread in 0..3 {
            put(&mut t339, 8 * thread, 2 << 29 | 53);
        }
        let view = move |name: &str| -> Option<Vec<u8>> {
            Some(match name {
                "t12" => t12.clone(),
                "t10" => t10.clone(),
                "b3" => b3.clone(),
                "t379" => t379.clone(),
                "t339" => t339.clone(),
                _ => return None,
            })
        };
        let r = rays(&view, 15, 2).unwrap();
        assert_eq!(r.len(), 3, "the lesser of the count and 64 threads");
        let (nx, ny, nz) = (2.0 * 5.5 / 8.0 - 1.0, 2.0 * 2.5 / 4.0 - 1.0, 2.0 * 1.5 / 4.0 - 1.0);
        let want = [1.0 + 16.0 * nz, 2.0 + 16.0 * ny, 3.0 + 32.0 * nx];
        for a in 0..3 {
            assert!((r[0].origin[a] - want[a]).abs() < 1e-5, "{:?} {want:?}", r[0].origin);
        }
        assert_eq!(r[0].slot, 2 * 32768 + 53);
        assert_eq!(r[0].direction, [0.0, 1.0, 0.0]);
        // The far origin lies on the sphere of radius 70 along +y.
        let q = r[0].far_origin;
        assert!((dot(q, q).sqrt() - 70.0).abs() < 1e-3, "{q:?}");
        assert!((q[0] - want[0]).abs() < 1e-5 && q[1] > want[1]);
        assert!((r[0].far_extent - (1000.0 - (q[1] - want[1]))).abs() < 1e-2);
        // s6 16 without t373: refused.
        assert!(rays(&view, 16, 2).is_err());
    }

    /// The game against the triangles: the same bytes agree; a hit the game
    /// misses is put on the reader where the tree reaches it and on the
    /// tree where a box loses it; a hit of the game alone is its own class.
    #[test]
    fn disagreements_are_classed() {
        let mut m = WordMemory::default();
        scene(&mut m);
        let far_top = 0xd_6010_0000u64;
        top(&mut m, far_top, &[]);
        let ray = |thread: u32, o: V3| Ray {
            thread,
            probe: thread,
            cascade: 0,
            slot: u64::from(thread),
            origin: o,
            direction: [0.0, 0.0, 1.0],
            extent: 1000.0,
            far_origin: o,
            far_extent: 900.0,
        };
        let list: Vec<Ray> = vec![ray(0, [3.3, 3.4, 0.0]), ray(1, [50.0, 50.0, 0.0]), ray(2, [109.3, 3.0, 0.0]), ray(3, [101.3, 3.0, 0.0])];
        let s = Scene::read(&m, TOP).unwrap();
        let f = Scene::read(&m, far_top).unwrap();
        let rows = compare(&s, &f, &list, &[0, 127, 0, 0]);
        assert!(rows.iter().all(|r| r.class.is_none()));
        let sum = Summary::of(&rows);
        assert_eq!((sum.sure, sum.same, sum.game_hits, sum.triangle_hits), (4, 4, 3, 3));
        let rows = compare(&s, &f, &list, &[127, 0, 127, 9]);
        assert_eq!(rows[0].class, Some(Class::Reader(false)));
        assert_eq!(rows[1].class, Some(Class::GameOnly));
        let sum = Summary::of(&rows);
        assert_eq!((sum.sure, sum.same, sum.other_bytes), (3, 0, 1));
        assert!(!sum.accepted());
        // With instance 1's box too small the loss is the tree's.
        let (parent, slot) = s.parents[&(s.leaf(1) & 0x1fff_ffff)];
        m.0.insert(node(TOP, parent) + 16 + 24 * slot as u64 + 12, bits(104.0));
        let s = Scene::read(&m, TOP).unwrap();
        let f = Scene::read(&m, far_top).unwrap();
        let rows = compare(&s, &f, &list, &[0, 127, 127, 0]);
        assert_eq!(rows[2].class, Some(Class::Tree(Lost::InstanceBox)));
        assert_eq!(rows[3].class, None);
    }

    /// 1000 sure rays with 10 wrong are accepted, 11 wrong are not, 999
    /// rays never.
    #[test]
    fn acceptance_is_99_per_cent_of_1000() {
        let s = |sure, same| Summary { sure, same, ..Summary::default() };
        assert!(s(1000, 990).accepted());
        assert!(!s(1000, 989).accepted());
        assert!(!s(999, 999).accepted());
    }

    /// The page table's entry picks among the records at an address.
    #[test]
    fn the_entry_picks_the_page_record() {
        let records = [(7u32, 0x10000u64), (9, 0x20000)];
        assert_eq!(pick(Entry::Copy { seg: 9 }, &records), Some(0x20000));
        assert_eq!(pick(Entry::Copy { seg: 8 }, &records), None);
        assert_eq!(pick(Entry::Null, &records), None);
        assert_eq!(pick(Entry::Arena { seg: 3, offset: 0 }, &records[..1]), Some(0x10000));
        assert_eq!(pick(Entry::Arena { seg: 3, offset: 0 }, &records), None);
    }
}
