//! `cargo run --release -p victory-gpu --example bvh_ref -- <capture folder> [--program <hash prefix>] [--rows <file>] [--list N] [--panorama <file.bmp>] [--view <file.bmp>] [--width W]`:
//! the CPU reference for the RT reader's rays
//! (`victory_gpu::bvh_replay::reference`) over a reader capture
//! (`VICTORY_BVH_CAPTURE_READER` with `VICTORY_BVH_CAPTURE_READER_BLAS=1`).
//! No GPU, no translator. Exit code 0: the game's result is what the
//! triangles say on at least 99 % of at least 1000 sure rays; 1: not; 2: the
//! capture cannot stand for the game (pages or views missing).
//!
//! `--panorama`: what the near top level's triangles look like from the
//! camera (the origin of the program's coordinates) all around, 360 by 180
//! degrees, z up, lit by the program's own light direction: sunlit
//! surfaces warm, surfaces turned away from the light blue-grey, surfaces
//! another triangle shadows grey, no triangle sky blue. From the triangles
//! alone (no boxes of the structures, no shadow map). `--view`: the same
//! through the game's camera, from the matrix `b3` holds at 0x70 (three
//! rows of four: the world axes' parts of clip x, y, z, w; onCap3: 1.27 and
//! 2.24 on x and y, 16:9), to lay next to a frame of the run.

use std::io::Write;

use victory_gpu::bvh_replay::format::Capture;
use victory_gpu::bvh_replay::reference::{CaptureMemory, Class, Row, Scene, Summary, Verdict, compare, rays, reader_dispatches, view_name};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir = None;
    let mut program = "f607780a".to_string();
    let mut rows_file = None;
    let mut list = 12usize;
    let mut panorama = None;
    let mut view = None;
    let mut width = 1024usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--program" => program = it.next().cloned().unwrap_or_default(),
            "--rows" => rows_file = it.next().cloned(),
            "--list" => list = it.next().and_then(|v| v.parse().ok()).unwrap_or(12),
            "--panorama" => panorama = it.next().cloned(),
            "--view" => view = it.next().cloned(),
            "--width" => width = it.next().and_then(|v| v.parse().ok()).unwrap_or(1024),
            s if s.starts_with("--") => anyhow::bail!("unknown option {s}"),
            s => dir = Some(std::path::PathBuf::from(s)),
        }
    }
    let Some(dir) = dir else {
        anyhow::bail!("usage: bvh_ref <capture folder> [--program <hash prefix>] [--rows <file>] [--list N]");
    };
    let capture = Capture::read(&mut std::io::BufReader::new(std::fs::File::open(dir.join("capture.bin"))?))?;
    let memory = CaptureMemory::new(&capture, std::fs::File::open(dir.join("pre.pages"))?);
    println!(
        "capture {}: submission {}, frame {}, exe {}, translator {}, pages left out {:?}",
        dir.display(),
        capture.meta("submission").unwrap_or("?"),
        capture.meta("frame").unwrap_or("?"),
        capture.meta("exe").unwrap_or("?"),
        capture.meta("translator").unwrap_or("?"),
        capture.left_out()
    );
    for key in ["VICTORY_BVH_UNPACK_BACK", "VICTORY_BVH_ARENA", "VICTORY_BVH_CAPTURE_READER_BLAS", "tops first", "pages of top levels and bottom-level headers", "pages of those with no entry in the page table"] {
        if let Some(v) = capture.meta(key) {
            println!("  {key}: {v}");
        }
    }
    let mut all: Vec<Row> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    let mut unsound = 0;
    let mut checked = std::collections::BTreeSet::new();
    let mut out = match &rows_file {
        Some(f) => Some(std::io::BufWriter::new(std::fs::File::create(f)?)),
        None => None,
    };
    if let Some(o) = out.as_mut() {
        writeln!(o, "dispatch\tthread\tcascade\tprobe\tox\toy\toz\textent\tfar_extent\ttriangles\ttree\tgame\tclass\tdeepest\tmet_far\tmet_t\tmet_instance\tmet_node")?;
    }
    let dispatches = reader_dispatches(&capture, &program);
    if dispatches.is_empty() {
        invalid.push(format!("no dispatch of {program}"));
    }
    for (k, s6, groups, views) in dispatches {
        let find = |name: &str| views.iter().find(|v| view_name(v) == name).copied();
        let pre = |name: &str| find(name).map(|v| v.pre.clone());
        if groups == 0 {
            println!("dispatch {k}: s6 {s6}, no groups");
            continue;
        }
        let list_rays = match rays(&pre, s6, groups) {
            Ok(r) => r,
            Err(e) => {
                invalid.push(format!("dispatch {k}: {e}"));
                continue;
            }
        };
        let (Some(t355), Some(t356), Some(u52)) = (find("t355"), find("t356"), find("u52")) else {
            invalid.push(format!("dispatch {k}: t355, t356 or u52 not captured"));
            continue;
        };
        if u52.post.is_empty() {
            invalid.push(format!("dispatch {k}: no bytes of u52 after the fence"));
            continue;
        }
        println!(
            "dispatch {k}: s6 {s6}, groups {groups}, rays {}, near {:#x}, far {:#x}, u52 {} bytes",
            list_rays.len(),
            t355.title,
            t356.title,
            u52.post.len()
        );
        let scenes = (Scene::read(&memory, t355.title), Scene::read(&memory, t356.title));
        let (near, far) = match scenes {
            (Ok(n), Ok(f)) => (n, f),
            (n, f) => {
                invalid.extend(n.err());
                invalid.extend(f.err());
                continue;
            }
        };
        for (name, s) in [("near", &near), ("far", &far)] {
            if checked.insert(s.base) {
                let c = s.check();
                println!(" {name} top level {:#x}: check {}", s.base, if c.sound() { "SOUND" } else { "NOT SOUND" });
                println!("{c}");
                unsound += u32::from(!c.sound());
            }
        }
        if let (Some(file), Some(first)) = (panorama.take(), list_rays.first()) {
            let height = width / 2;
            let image = render(&near, &far, first.direction, width, height, &|x, y| {
                let elevation = std::f64::consts::FRAC_PI_2 - std::f64::consts::PI * y;
                let azimuth = std::f64::consts::TAU * x - std::f64::consts::PI;
                [elevation.cos() * azimuth.cos(), elevation.cos() * azimuth.sin(), elevation.sin()]
            });
            std::fs::write(&file, image)?;
            println!(" panorama {width} x {height} written to {file} (light towards {:?})", first.direction);
        }
        if let (Some(file), Some(first), Some(b3)) = (view.take(), list_rays.first(), pre("b3")) {
            let f = |o: usize| b3.get(o..o + 4).map_or(0.0, |b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            // Column c of the three rows at 0x70: clip x (0), y (1), w (3).
            let column = |c: usize| [f(0x70 + 4 * c), f(0x80 + 4 * c), f(0x90 + 4 * c)];
            let (right, up, forward) = (column(0), column(1), column(3));
            let sq = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).max(1e-30);
            let height = width * 9 / 16;
            println!(" view: right {right:?}, up {up:?}, forward {forward:?}");
            let image = render(&near, &far, first.direction, width, height, &|x, y| {
                let (u, v) = (2.0 * x - 1.0, 1.0 - 2.0 * y);
                [0, 1, 2].map(|a| forward[a] / sq(forward) + u * right[a] / sq(right) + v * up[a] / sq(up))
            });
            std::fs::write(&file, image)?;
            println!(" view {width} x {height} written to {file}");
        }
        let rows = compare(&near, &far, &list_rays, &u52.post);
        let sum = Summary::of(&rows);
        println!(" dispatch {k} rays:\n{sum}");
        let mut shown = 0;
        for r in rows.iter().filter(|r| r.class.is_some()) {
            if shown == list {
                break;
            }
            shown += 1;
            println!(
                "   thread {} cascade {} probe {} origin ({:.2}, {:.2}, {:.2}): triangles {:?}, tree {:?}, game {:?}, {:?}, met {:?}",
                r.ray.thread, r.ray.cascade, r.ray.probe, r.ray.origin[0], r.ray.origin[1], r.ray.origin[2], r.triangles, r.tree, r.game, r.class, r.met
            );
        }
        if let Some(o) = out.as_mut() {
            for r in &rows {
                let verdict = |v: Verdict| match v {
                    Verdict::Hit => "hit",
                    Verdict::Edge => "edge",
                    Verdict::Miss => "miss",
                };
                let class = match r.class {
                    None => "-".to_string(),
                    Some(Class::Tree(l)) => format!("tree:{l:?}"),
                    Some(Class::Reader(deep)) => format!("reader{}", if deep { ":deep" } else { "" }),
                    Some(Class::GameOnly) => "game-only".to_string(),
                };
                let met = r.met.map_or("-\t-\t-\t-".to_string(), |(far, m)| format!("{}\t{:.4}\t{}\t{:#x}", u8::from(far), m.t, m.instance, m.node));
                writeln!(
                    o,
                    "{k}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{class}\t{}\t{met}",
                    r.ray.thread,
                    r.ray.cascade,
                    r.ray.probe,
                    r.ray.origin[0],
                    r.ray.origin[1],
                    r.ray.origin[2],
                    r.ray.extent,
                    r.ray.far_extent,
                    verdict(r.triangles),
                    verdict(r.tree),
                    r.game.map_or("-".to_string(), |b| b.to_string()),
                    r.deepest
                )?;
            }
        }
        all.extend(rows);
    }
    let missing = memory.missing.borrow();
    if !missing.is_empty() {
        invalid.push(format!(
            "{} pages the page table names are not in the capture (first {})",
            missing.len(),
            missing.iter().take(8).map(|p| format!("{p:#x}")).collect::<Vec<_>>().join(", ")
        ));
    }
    println!("pages read with no entry in the page table (the program reads zeros): {}", memory.null.borrow().len());
    let sum = Summary::of(&all);
    println!("ALL DISPATCHES:\n{sum}");
    println!("top levels not sound by the check: {unsound}");
    let code = if !invalid.is_empty() {
        println!("RESULT: CAPTURE INVALID: {}", invalid.join("; "));
        2
    } else if sum.accepted() {
        println!("RESULT: PASS ({:.2} % of {} sure rays)", sum.agreement(), sum.sure);
        0
    } else {
        println!("RESULT: FAIL ({:.2} % of {} sure rays; needed 99 % of at least 1000)", sum.agreement(), sum.sure);
        1
    };
    std::process::exit(code);
}

/// An image as a 24-bit BMP, `width` x `height`: per pixel the ray from
/// the origin along `direction(x, y)` (x and y from 0 to 1, left to right
/// and top to bottom; any length), shaded as the head of this file says.
fn render(
    near: &Scene<'_, CaptureMemory>,
    far: &Scene<'_, CaptureMemory>,
    sun: [f64; 3],
    width: usize,
    height: usize,
    direction: &dyn Fn(f64, f64) -> [f64; 3],
) -> Vec<u8> {
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sun_len = len(sun).max(1e-12);
    let sun = sun.map(|x| x / sun_len);
    let row = (3 * width).next_multiple_of(4);
    let mut out = vec![0u8; 54 + row * height];
    out[0..2].copy_from_slice(b"BM");
    out[2..6].copy_from_slice(&((54 + row * height) as u32).to_le_bytes());
    out[10..14].copy_from_slice(&54u32.to_le_bytes());
    out[14..18].copy_from_slice(&40u32.to_le_bytes());
    out[18..22].copy_from_slice(&(width as u32).to_le_bytes());
    out[22..26].copy_from_slice(&(height as u32).to_le_bytes());
    out[26..28].copy_from_slice(&1u16.to_le_bytes());
    out[28..30].copy_from_slice(&24u16.to_le_bytes());
    out[34..38].copy_from_slice(&((row * height) as u32).to_le_bytes());
    for y in 0..height {
        for x in 0..width {
            let d = direction((x as f64 + 0.5) / width as f64, (y as f64 + 0.5) / height as f64);
            let dl = len(d).max(1e-30);
            let d = d.map(|v| v / dl);
            let towards = d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2];
            let rgb: [f64; 3] = match near.nearest([0.0; 3], d, 600.0) {
                None if towards > 0.9995 => [255.0, 255.0, 255.0],
                None => [110.0, 160.0, 250.0],
                Some((t, _, n)) => {
                    let l = len(n).max(1e-30);
                    // The normal on the viewer's side.
                    let facing = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                    let n = n.map(|v| if facing > 0.0 { -v / l } else { v / l });
                    let p = [0, 1, 2].map(|a| d[a] * t + n[a] * 0.02);
                    let lambert = n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2];
                    let depth = 0.55 + 0.45 / (1.0 + t / 6.0);
                    if lambert <= 0.0 {
                        [70.0 * depth, 80.0 * depth, 110.0 * depth]
                    } else {
                        let shadowed = near.brute(p, sun, 1000.0).0 == Verdict::Hit || far.brute(p, sun, 1000.0).0 == Verdict::Hit;
                        if shadowed {
                            [95.0 * depth, 95.0 * depth, 95.0 * depth]
                        } else {
                            let k = 0.45 + 0.55 * lambert;
                            [255.0 * k, 215.0 * k, 110.0 * k]
                        }
                    }
                }
            };
            let at = 54 + row * (height - 1 - y) + 3 * x;
            out[at] = rgb[2] as u8;
            out[at + 1] = rgb[1] as u8;
            out[at + 2] = rgb[0] as u8;
        }
    }
    out
}
