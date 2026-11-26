//! What a submission does for host copies of title buffers in the GPU's own
//! memory (`crate::shadow::cards`).
//!
//! Before the submission's lists, a list of ours copies what changed on the
//! copies' CPU sides to their GPU sides ([`Host::card_before`]). After the
//! submission, what its shaders wrote comes down the other way, into the
//! CPU sides, before it is compared with title memory ([`card_fetch`]): the
//! pages the shaders marked (`crate::shadow::copy_back_marks`) and the
//! ranges declared as written without marks ([`Host::card_declared`]).
//!
//! Our lists go to the queue each in a call of its own: buffers return to
//! the common state after each call, so none of them needs to know what
//! state the title's lists left a copy in.
//!
//! The two-sided copy follows fxceless/victory-fork (GPL-3.0), which finds
//! what shaders wrote by hashing on the GPU; here the shaders' own marks
//! say it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use super::{Host, HostList, Step, fail, host};
use crate::shadow::Shadow;

type OwnList = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// The host's objects for copies in the GPU's memory, used under the
/// submission lock (and [`WORK`]'s).
struct Work {
    /// Before a submission's lists: changed ranges go up.
    sync: OwnList,
    /// After it ran: what shaders wrote comes down.
    fetch: OwnList,
    /// Behind a submission's lists: the bitmaps of written pages and read
    /// slots come down.
    marks: OwnList,
    /// For the waits for fetches.
    fence: ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    sent: u64,
}

// SAFETY: D3D12 objects (free-threaded) and an event handle, used only
// under the lock.
unsafe impl Send for Work {}

static WORK: OnceLock<Option<Mutex<Work>>> = OnceLock::new();

/// Since the last frame-rate line: bytes copied to GPU sides and the ranges
/// they were; bytes fetched and their ranges; fetches and the microseconds
/// they were waited for.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Logs [`STATS`] per frame over `frames` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(super) fn log(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0)).collect();
    let (copies, bytes) = crate::shadow::card_copies();
    if copies > 0 {
        tracing::info!(
            "copies in GPU memory per frame: {:.1} MB sent up in {:.0} ranges; {:.1} MB fetched in {:.0} ranges by {:.1} fetches, \
             {:.1} ms waiting for them; {copies} such copies, {:.0} MB",
            s[0] / 1e6,
            s[1],
            s[2] / 1e6,
            s[3],
            s[4],
            s[5] / 1e3,
            bytes as f64 / 1e6
        );
    }
}

fn work() -> Option<std::sync::MutexGuard<'static, Work>> {
    WORK.get()?.as_ref().map(|w| w.lock().unwrap_or_else(|e| e.into_inner()))
}

impl Host {
    /// A buffer in the GPU's own memory that shaders read and write (the
    /// CPU cannot map it), zeroed, in the common state.
    pub(crate) fn card_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no buffer in GPU memory");
                None
            }
        }
    }

    /// Whether the host can keep copies in the GPU's memory: the lists
    /// that move their bytes exist (made at the first ask).
    pub(crate) fn card_ready(&self) -> bool {
        WORK.get_or_init(|| match self.make_work() {
            Ok(work) => Some(Mutex::new(work)),
            Err(e) => {
                tracing::warn!(error = %e, "host copies of title buffers stay in system memory");
                None
            }
        })
        .is_some()
    }

    /// The video memory budget DXGI gives the process in the host GPU's
    /// own memory, asked once; `None` where it cannot be asked. Copies in
    /// the GPU's memory take at most a quarter of it, an eighth with
    /// `VICTORY_CARD_BUFFERS_MB=auto`, any with a number there
    /// (`crate::shadow`), so that
    /// a card with less memory than the 8 GB they were measured on keeps
    /// room for the title's textures and targets.
    pub(crate) fn card_budget(&self) -> Option<u64> {
        static BUDGET: OnceLock<Option<u64>> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            use windows::Win32::Graphics::Dxgi::*;
            // SAFETY: plain DXGI queries; the out pointer is a live local.
            let asked = unsafe {
                (|| -> windows::core::Result<u64> {
                    let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
                    let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(self.device.GetAdapterLuid())?;
                    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                    adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)?;
                    Ok(info.Budget)
                })()
            };
            match asked {
                Ok(budget) if budget > 0 => {
                    tracing::info!(budget, "host GPU memory budget (copies of title buffers take at most a quarter)");
                    Some(budget)
                }
                Ok(_) => None,
                Err(e) => {
                    tracing::debug!(error = %e, "host GPU memory budget not asked");
                    None
                }
            }
        })
    }

    fn make_work(&self) -> windows::core::Result<Work> {
        let list = || -> windows::core::Result<OwnList> {
            // SAFETY: plain object creation; the list is closed until used.
            unsafe {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            }
        };
        // SAFETY: plain object creation.
        unsafe {
            Ok(Work {
                sync: list()?,
                fetch: list()?,
                marks: list()?,
                fence: self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                event: CreateEventW(None, false, false, None)?,
                sent: 0,
            })
        }
    }

    /// Before a submission's lists run: what changed on the CPU's side of
    /// copies in the GPU's memory since the last submission goes to their
    /// GPU side (a list of ours, sent now; the submission's lists follow it
    /// on the queue). The copies, to keep until the submission has run.
    pub(super) fn card_before(&self, marks: &[crate::shadow::CardMarksWork]) -> Vec<Arc<Shadow>> {
        if !crate::shadow::cards() {
            return Vec::new();
        }
        let dirty = crate::shadow::card_dirty();
        if dirty.is_empty() && marks.is_empty() {
            return Vec::new();
        }
        let Some(work) = work() else {
            return Vec::new();
        };
        let (allocator, list) = &work.sync;
        // SAFETY: the list's last run finished (every submission is waited
        // for under the submission lock, which the caller holds); live
        // buffers, and the ranges lie in them.
        let sent = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                for m in marks {
                    // The written pages in the GPU's memory start each
                    // submission clear (the last one's bits were fetched);
                    // the read slots are the CPU's where it cleared some.
                    if m.written > 0 {
                        list.CopyBufferRegion(&m.marks, 0, &m.zeros, 0, m.written);
                    }
                    if m.reads_up {
                        list.CopyBufferRegion(&m.marks, m.reads.0, &m.cpu, m.reads.0, m.reads.1);
                    }
                }
                for (shadow, ranges) in &dirty {
                    let Some(staging) = shadow.card_staging() else {
                        continue;
                    };
                    for &(a, b) in ranges {
                        list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                        STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                    }
                    STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                }
                list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                Ok(())
            })()
        };
        if let Err(e) = sent {
            fail(format_args!("host copies in GPU memory could not be brought up to date ({e})"));
        }
        dirty.into_iter().map(|(shadow, _)| shadow).collect()
    }

    /// With the bitmaps of written pages and read slots in the GPU's
    /// memory: the list that copies them down, to send behind the
    /// submission's lists (the wait for them is the wait for it;
    /// `crate::shadow::marks_fetched`, once it has run).
    pub(super) fn card_marks(&self, marks: &[crate::shadow::CardMarksWork]) -> Option<ID3D12CommandList> {
        if marks.is_empty() {
            return None;
        }
        let work = work()?;
        let (allocator, list) = &work.marks;
        // SAFETY: the list's last run finished (it ran behind a submission
        // that was waited for); live buffers of at least `bytes` bytes.
        let recorded = unsafe {
            (|| -> windows::core::Result<ID3D12CommandList> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                for m in marks {
                    if m.written > 0 {
                        list.CopyBufferRegion(&m.fetched, 0, &m.marks, 0, m.written);
                    }
                    list.CopyBufferRegion(&m.fetched, m.reads.0, &m.marks, m.reads.0, m.reads.1);
                }
                list.Close()?;
                list.cast()
            })()
        };
        match recorded {
            Ok(list) => Some(list),
            Err(e) => fail(format_args!("the views' written pages could not be fetched from GPU memory ({e})")),
        }
    }

    /// After the submission ran: the ranges its lists declared as written
    /// by shaders that do not mark their writes come down to the CPU's
    /// sides of copies in the GPU's memory, where their snapshots are
    /// compared (`Shadow::copy_back`).
    pub(super) fn card_declared(&self, runs: &[&HostList]) {
        let mut declared: crate::shadow::Ranges = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len, false, _) = step
                    && shadow.carded()
                {
                    let range = shadow.offsets(*address, *len);
                    match declared.iter_mut().find(|(s, _)| Arc::ptr_eq(s, shadow)) {
                        Some((_, ranges)) => ranges.push(range),
                        None => declared.push((shadow.clone(), vec![range])),
                    }
                }
            }
        }
        let ranges: Vec<(&Shadow, usize, usize)> = declared
            .iter_mut()
            .flat_map(|(shadow, ranges)| {
                let shadow: &Shadow = shadow;
                crate::shadow::joined(std::mem::take(ranges)).into_iter().map(move |(a, b)| (shadow, a, b))
            })
            .collect();
        card_fetch(&ranges);
    }
}

/// Copies the ranges (copy, offsets from and to) of copies in the GPU's
/// memory from their GPU sides to their CPU sides, now: one list and one
/// wait for all of them. Only while no submission is on the GPU.
pub(crate) fn card_fetch(ranges: &[(&Shadow, usize, usize)]) {
    let ranges: Vec<&(&Shadow, usize, usize)> = ranges.iter().filter(|r| r.1 < r.2 && r.0.carded()).collect();
    if ranges.is_empty() {
        return;
    }
    let (Some(host), Some(mut work)) = (host(), work()) else {
        return;
    };
    let started = std::time::Instant::now();
    work.sent += 1;
    let (allocator, list) = &work.fetch;
    // SAFETY: the list's last run finished (it was waited for below); live
    // buffers, and the ranges lie in them; the fence and event are ours.
    let fetched = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            for &&(shadow, a, b) in &ranges {
                let Some(staging) = shadow.card_staging() else {
                    continue;
                };
                list.CopyBufferRegion(staging, a as u64, &shadow.buffer, a as u64, (b - a) as u64);
                STATS[2].fetch_add((b - a) as u64, Ordering::Relaxed);
            }
            list.Close()?;
            host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
            host.queue.Signal(&work.fence, work.sent)?;
            work.fence.SetEventOnCompletion(work.sent, work.event)?;
            WaitForSingleObject(work.event, INFINITE);
            Ok(())
        })()
    };
    if let Err(e) = fetched {
        fail(format_args!("what shaders wrote to host copies in GPU memory could not be fetched ({e})"));
    }
    // A title write while the fetch ran (tests).
    #[cfg(test)]
    crate::shadow::tests::during_fetch();
    STATS[3].fetch_add(ranges.len() as u64, Ordering::Relaxed);
    STATS[4].fetch_add(1, Ordering::Relaxed);
    STATS[5].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}
