//! `VICTORY_TITLE_PROBE=<rva>[,<rva>...]` (diagnostics, off by default):
//! where the title's own code selects, latches and clears the bottom levels
//! it unpacks (Fable, bvh-harness-design-fable.md Aufruf 14 P5-P7).
//!
//! Up to four hardware execution breakpoints (Dr0-Dr3) at the executable's
//! base plus each RVA (`dll:<rva>`: XboxBvhBuilder.dll's base instead). Not a
//! byte of title code changes. A vectored handler of our own takes the
//! single-step exception of a breakpoint we set (the address is ours and Dr6
//! names its register, or Dr6 says nothing), writes a fixed record into a
//! ring allocated beforehand (no lock, no allocation: the title thread may
//! hold the heap lock), sets RF so that the instruction runs once without
//! breaking again, and continues; everything else goes on to the other
//! handlers unchanged. A helper thread arms every thread of the process
//! every 2 s (suspend, set the debug registers, resume) and logs the
//! records: point, thread, submission, all 16 registers and, for every
//! register that points at readable memory, what a record E there holds
//! (+00, +18, +20, +28, +40) and the header the CPU reads through it:
//! B = [E+18], k = u64[B+20] - u64[B+28] (0x38 if that underflows),
//! D0 = u32[B+k]; all registers, plus the record at rax - 0x40 and the byte
//! at rbx + rdi + 3. Register use by point (senior 125, rt-rea-bvh-calls.md
//! section 4): 32619e1 (selected) rax = E, r9 = [E+20], the target;
//! 3261a80 rax = E + 0x40; 4ee17e6 rsi = E; the wrapper's check ea6d2d or
//! 2490aed rbx = B, rdi = k, rax = target, rbp = source (E in no register:
//! joined by the target), the packed bit is bit 0 of the byte at B + k + 3.
//! The run's value: `32619e1,3261a80,4ee17e6,ea6d2d` (or 2490aed last).
//!
//! With the switch set, the executable's imports of XboxBvhBuilder.dll that
//! resolve to RVA 0x219d0 (the batched unpacking, ordinal 8; the EXE
//! imports it by name, IAT cell 6793fa0) and 0x216a0 (the batched build),
//! matched by address whatever the import names, are bound to wrappers that log their arguments and the result and
//! pass both through unchanged.
//!
//! The submission is the host's count when the record was taken (the `sN`
//! of the `BVH GT sN rehydration` lines): commands the title records now go
//! to a later submission.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use victory_pe::ImportSymbol;
use windows_sys::Win32::Foundation::{
    CloseHandle, EXCEPTION_SINGLE_STEP, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, CONTEXT_DEBUG_REGISTERS_AMD64, EXCEPTION_POINTERS,
    EXCEPTION_RECORD, GetThreadContext, SetThreadContext,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleHandleExW, GetModuleHandleW,
};
use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread,
    THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME,
};

/// The switch.
const SWITCH: &str = "VICTORY_TITLE_PROBE";
/// The builder DLL, whose imports are wrapped and whose base `dll:` names.
const BUILDER: &str = "XboxBvhBuilder.dll";
/// The DLL's batched unpacking (ordinal 8) and batched build.
const REHYDRATE_RVA: usize = 0x219d0;
const BUILD_RVA: usize = 0x216a0;

/// Debug registers there are.
pub(crate) const POINTS: usize = 4;
/// Records the ring holds before the helper thread drains it.
const RING: usize = 4096;
/// The record fields read at E.
const FIELDS: [u64; 5] = [0x00, 0x18, 0x20, 0x28, 0x40];
/// `CONTEXT.EFlags` resume flag: the instruction at RIP runs without its
/// execution breakpoint firing again.
const RF: u32 = 1 << 16;
/// Dr6: breakpoint conditions 0-3, and single step (TF).
const DR6_HITS: u64 = 0xf;
const DR6_BS: u64 = 1 << 14;

/// The switch's value, if set and not empty.
fn switch() -> Option<&'static str> {
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            std::env::var(SWITCH)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        })
        .as_deref()
}

/// Whether `VICTORY_TITLE_PROBE` is on.
pub fn enabled() -> bool {
    switch().is_some()
}

/// Where a probe point lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Base {
    Exe,
    Builder,
}

/// The points a `VICTORY_TITLE_PROBE` value names, at most [`POINTS`]
/// (hex RVAs, `0x` optional, `dll:` for the builder DLL); what it cannot
/// read is returned apart, for the log.
pub(crate) fn parse(value: &str) -> (Vec<(Base, usize)>, Vec<String>) {
    let (mut points, mut refused) = (Vec::new(), Vec::new());
    for token in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (base, hex) = match token.strip_prefix("dll:") {
            Some(rest) => (Base::Builder, rest),
            None => (Base::Exe, token),
        };
        let hex = hex.trim_start_matches("0x");
        match usize::from_str_radix(hex, 16) {
            Ok(rva) if rva != 0 && points.len() < POINTS => points.push((base, rva)),
            _ => refused.push(token.to_owned()),
        }
    }
    (points, refused)
}

/// The addresses the debug registers hold (0: unused); the handler's only
/// configuration.
static ADDRESSES: [AtomicUsize; POINTS] = [const { AtomicUsize::new(0) }; POINTS];
/// The RVAs, for the log.
static RVAS: [AtomicUsize; POINTS] = [const { AtomicUsize::new(0) }; POINTS];
/// Records and calls, in the order they were taken (one count for both, so
/// that the log orders breakpoint hits against the DLL calls).
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
/// Where the submission count comes from (`fn() -> u64`, one atomic read).
static SUBMISSION: AtomicUsize = AtomicUsize::new(0);
/// Records the ring had no room for.
static LOST: AtomicU64 = AtomicU64::new(0);
/// Hits by point.
static HITS: [AtomicU64; POINTS] = [const { AtomicU64::new(0) }; POINTS];

fn submission() -> u64 {
    let f = SUBMISSION.load(Ordering::Relaxed);
    if f == 0 {
        return 0;
    }
    // SAFETY: only ever set from a `fn() -> u64`.
    let f: fn() -> u64 = unsafe { std::mem::transmute(f) };
    f()
}

fn next() -> u64 {
    SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

/// What a register pointing at a record E shows.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub(crate) struct Record {
    /// E+00, +18, +20, +28, +40.
    pub fields: [u64; 5],
    /// B = [E+18], k, D0 = u32[B+k].
    pub head: [u64; 3],
    /// Bit 0: the fields were read; bit 1: k was read; bit 2: D0 was read.
    pub read: u8,
}

/// A breakpoint hit, as the handler takes it.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Hit {
    pub sequence: u64,
    pub point: u8,
    pub thread: u32,
    pub submission: u64,
    /// rax rcx rdx rbx rsp rbp rsi rdi r8-r15.
    pub regs: [u64; 16],
    pub rip: u64,
    pub eflags: u32,
    /// By register, as `regs`.
    pub records: [Record; 16],
    /// The record at rax - 0x40 (at 3261a80 rax is E + 0x40: senior 125,
    /// rt-rea-bvh-calls.md section 4).
    pub below_rax: Record,
    /// The byte at rbx + rdi + 3, bit 0 the packed bit as the CPU reads it
    /// (at the wrapper's check ea6d2d/2490aed rbx is B, rdi k); 0xffff if it
    /// cannot be read.
    pub packed_byte: u16,
}

const REGISTERS: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// A ring slot: `state` is the record's sequence in the ring plus one once
/// it is written.
struct Slot {
    state: AtomicU64,
    hit: UnsafeCell<MaybeUninit<Hit>>,
}

// SAFETY: a slot's hit is written by the one thread that reserved it, and
// read only after `state` says it is complete (release/acquire).
unsafe impl Sync for Slot {}

/// The ring, allocated before any breakpoint is set.
static SLOTS: OnceLock<Box<[Slot]>> = OnceLock::new();
/// Reserved and drained records.
static HEAD: AtomicU64 = AtomicU64::new(0);
static TAIL: AtomicU64 = AtomicU64::new(0);

fn ring() -> &'static [Slot] {
    SLOTS.get_or_init(|| {
        (0..RING)
            .map(|_| Slot {
                state: AtomicU64::new(0),
                hit: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect()
    })
}

/// Puts `hit` into the ring; no lock, no allocation (once [`ring`] exists).
fn push(hit: &Hit) {
    let Some(slots) = SLOTS.get() else {
        LOST.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut at = HEAD.load(Ordering::Relaxed);
    loop {
        if at.wrapping_sub(TAIL.load(Ordering::Acquire)) >= RING as u64 {
            LOST.fetch_add(1, Ordering::Relaxed);
            return;
        }
        match HEAD.compare_exchange_weak(at, at + 1, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => break,
            Err(now) => at = now,
        }
    }
    let slot = &slots[at as usize % RING];
    // SAFETY: slot `at` is ours until `state` is set; the drain does not read
    // it before.
    unsafe { (*slot.hit.get()).write(*hit) };
    slot.state.store(at + 1, Ordering::Release);
}

/// The records written so far, in order (the helper thread's, and tests').
fn drain(mut each: impl FnMut(&Hit)) {
    let Some(slots) = SLOTS.get() else {
        return;
    };
    loop {
        let at = TAIL.load(Ordering::Relaxed);
        let slot = &slots[at as usize % RING];
        if slot.state.load(Ordering::Acquire) != at + 1 {
            return;
        }
        // SAFETY: complete (state), and not reused before TAIL moves on.
        let hit = unsafe { (*slot.hit.get()).assume_init() };
        TAIL.store(at + 1, Ordering::Release);
        each(&hit);
    }
}

/// Whether `[at, at + len)` lies in committed, readable memory (a page that
/// is guarded or not accessible is left alone). Regions already found
/// readable are kept in `known`, as (start, end).
fn readable(at: u64, len: u64, known: &mut [(u64, u64); 4]) -> bool {
    if !(0x1_0000..0x7fff_ffff_0000).contains(&at) {
        return false;
    }
    let end = at + len;
    if known.iter().any(|&(s, e)| s <= at && end <= e) {
        return true;
    }
    let mut start = at;
    while start < end {
        let mut info = MaybeUninit::<MEMORY_BASIC_INFORMATION>::zeroed();
        // SAFETY: a query; the buffer is as large as said.
        let n = unsafe {
            VirtualQuery(
                start as *const c_void,
                info.as_mut_ptr(),
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if n == 0 {
            return false;
        }
        // SAFETY: filled by the query.
        let info = unsafe { info.assume_init() };
        const COMMIT: u32 = 0x1000;
        const READ: u32 = 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80;
        const GUARD_OR_NONE: u32 = 0x100 | 0x01;
        if info.State != COMMIT || info.Protect & READ == 0 || info.Protect & GUARD_OR_NONE != 0 {
            return false;
        }
        let region = (
            info.BaseAddress as u64,
            info.BaseAddress as u64 + info.RegionSize as u64,
        );
        known.rotate_right(1);
        known[0] = region;
        start = region.1;
    }
    true
}

fn read_u64(at: u64, known: &mut [(u64, u64); 4]) -> Option<u64> {
    // SAFETY: readable just now; title memory is not unmapped under a
    // running title.
    readable(at, 8, known).then(|| unsafe { (at as *const u64).read_unaligned() })
}

fn read_u32(at: u64, known: &mut [(u64, u64); 4]) -> Option<u32> {
    // SAFETY: as `read_u64`.
    readable(at, 4, known).then(|| unsafe { (at as *const u32).read_unaligned() })
}

/// What a record at `e` holds, and the header the CPU reads through it.
pub(crate) fn record_at(e: u64, known: &mut [(u64, u64); 4]) -> Record {
    let mut r = Record::default();
    if e & 7 != 0 || !readable(e, 0x48, known) {
        return r;
    }
    for (field, &off) in r.fields.iter_mut().zip(&FIELDS) {
        // SAFETY: the whole record is readable (checked above).
        *field = unsafe { ((e + off) as *const u64).read_unaligned() };
    }
    r.read = 1;
    let b = r.fields[1];
    r.head[0] = b;
    if let (Some(a), Some(c)) = (
        read_u64(b.wrapping_add(0x20), known),
        read_u64(b.wrapping_add(0x28), known),
    ) {
        let k = a.checked_sub(c).unwrap_or(0x38);
        r.head[1] = k;
        r.read |= 2;
        if let Some(d0) = read_u32(b.wrapping_add(k), known) {
            r.head[2] = u64::from(d0);
            r.read |= 4;
        }
    }
    r
}

/// The handler's work: takes a single-step exception at one of our
/// addresses, records it, sets RF, clears Dr6 and says to continue; leaves
/// every other exception alone (false).
pub(crate) fn take(record: &EXCEPTION_RECORD, context: &mut CONTEXT) -> bool {
    if record.ExceptionCode != EXCEPTION_SINGLE_STEP {
        return false;
    }
    // A single step (TF) that names no breakpoint is someone else's.
    if context.Dr6 & DR6_BS != 0 && context.Dr6 & DR6_HITS == 0 {
        return false;
    }
    let rip = context.Rip as usize;
    let Some(point) = (0..POINTS).find(|&i| {
        let at = ADDRESSES[i].load(Ordering::Relaxed);
        at != 0 && at == rip && (context.Dr6 & DR6_HITS == 0 || context.Dr6 & (1 << i) != 0)
    }) else {
        return false;
    };
    let regs = [
        context.Rax,
        context.Rcx,
        context.Rdx,
        context.Rbx,
        context.Rsp,
        context.Rbp,
        context.Rsi,
        context.Rdi,
        context.R8,
        context.R9,
        context.R10,
        context.R11,
        context.R12,
        context.R13,
        context.R14,
        context.R15,
    ];
    let mut known = [(0, 0); 4];
    let mut hit = Hit {
        sequence: next(),
        point: point as u8,
        // SAFETY: a plain query.
        thread: unsafe { GetCurrentThreadId() },
        submission: submission(),
        regs,
        rip: context.Rip,
        eflags: context.EFlags,
        records: [Record::default(); 16],
        below_rax: Record::default(),
        packed_byte: 0xffff,
    };
    for (r, &value) in hit.records.iter_mut().zip(&regs) {
        *r = record_at(value, &mut known);
    }
    hit.below_rax = record_at(context.Rax.wrapping_sub(0x40), &mut known);
    let packed = context.Rbx.wrapping_add(context.Rdi).wrapping_add(3);
    if readable(packed, 1, &mut known) {
        // SAFETY: readable just now.
        hit.packed_byte = u16::from(unsafe { (packed as *const u8).read() });
    }
    HITS[point].fetch_add(1, Ordering::Relaxed);
    push(&hit);
    context.EFlags |= RF;
    context.Dr6 = 0;
    true
}

unsafe extern "system" fn on_exception(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes the exception being dispatched.
    let taken = unsafe { take(&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    if taken { -1 } else { 0 } // EXCEPTION_CONTINUE_EXECUTION / _SEARCH
}

/// Dr7 for the addresses set: local enable per register, execution
/// breakpoints of length 1 (condition and length bits zero).
pub(crate) fn dr7(addresses: &[usize; POINTS]) -> u64 {
    (0..POINTS)
        .filter(|&i| addresses[i] != 0)
        .map(|i| 1u64 << (2 * i))
        .fold(0, |a, b| a | b)
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Sets the debug registers of thread `id` (not the calling thread) to
/// `addresses`: suspended only for the context calls (nothing allocates in
/// between). True if it had to change them.
pub(crate) fn arm_thread(id: u32, addresses: &[usize; POINTS]) -> bool {
    // SAFETY: opens a thread of our own process.
    let thread: HANDLE = unsafe {
        OpenThread(
            THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT,
            0,
            id,
        )
    };
    if thread.is_null() {
        return false;
    }
    let want = dr7(addresses);
    let mut changed = false;
    // SAFETY: a thread of this process that is not this one; resumed below.
    if unsafe { SuspendThread(thread) } != u32::MAX {
        // SAFETY: zeroed is a valid CONTEXT; the flags say what to read.
        let mut c: Aligned = unsafe { std::mem::zeroed() };
        c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
        // SAFETY: suspended thread, aligned context.
        if unsafe { GetThreadContext(thread, &mut c.0) } != 0 {
            let now = [c.0.Dr0, c.0.Dr1, c.0.Dr2, c.0.Dr3];
            let same =
                (0..POINTS).all(|i| now[i] == addresses[i] as u64) && c.0.Dr7 & 0xffff_00ff == want;
            if !same {
                (c.0.Dr0, c.0.Dr1, c.0.Dr2, c.0.Dr3) = (
                    addresses[0] as u64,
                    addresses[1] as u64,
                    addresses[2] as u64,
                    addresses[3] as u64,
                );
                c.0.Dr7 = want;
                c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                // SAFETY: as above.
                changed = unsafe { SetThreadContext(thread, &c.0) } != 0;
            }
        }
        // SAFETY: suspended above.
        unsafe { ResumeThread(thread) };
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(thread) };
    changed
}

/// The other threads of this process.
fn other_threads(ids: &mut Vec<u32>) {
    ids.clear();
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return;
    }
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: open snapshot, sized entry.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
}

/// One log line per record.
fn log_hit(hit: &Hit) {
    let point = usize::from(hit.point);
    let regs: Vec<String> = REGISTERS
        .iter()
        .zip(&hit.regs)
        .map(|(n, v)| format!("{n}={v:#x}"))
        .collect();
    let records: Vec<String> = REGISTERS
        .iter()
        .chain(&["rax-40"])
        .zip(hit.records.iter().chain(std::iter::once(&hit.below_rax)))
        .filter(|(_, r)| r.read & 1 != 0)
        .map(|(n, r)| {
            let f = r.fields;
            let mut s = format!(
                "{n}:+00={:#x},+18={:#x},+20={:#x},+28={:#x},+40={:#x}",
                f[0], f[1], f[2], f[3], f[4]
            );
            if r.read & 2 != 0 {
                s.push_str(&format!(",B={:#x},k={:#x}", r.head[0], r.head[1]));
            }
            if r.read & 4 != 0 {
                s.push_str(&format!(",D0={:#x}", r.head[2]));
            }
            s
        })
        .collect();
    tracing::info!(
        "TITLE PROBE hit seq={} p=Dr{point} rva={:#x} s={} tid={} rip={:#x} eflags={:#x} regs=[{}] packed_byte(rbx+rdi+3)={} E=[{}]",
        hit.sequence,
        RVAS[point].load(Ordering::Relaxed),
        hit.submission,
        hit.thread,
        hit.rip,
        hit.eflags,
        regs.join(" "),
        if hit.packed_byte == 0xffff {
            "?".to_owned()
        } else {
            format!("{:#x}", hit.packed_byte)
        },
        records.join(" "),
    );
}

/// Base of the module `address` lies in, if any.
fn module_base(address: usize) -> Option<usize> {
    let mut module = std::ptr::null_mut();
    // SAFETY: looks up the module containing `address` without a reference.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address as *const u16,
            &mut module,
        )
    };
    (ok != 0).then_some(module as usize)
}

/// Starts the probe if `VICTORY_TITLE_PROBE` names points: the handler, the
/// ring and the helper thread. `exe` is the executable's base; `submission`
/// the host's submission count. Without the switch nothing happens (no
/// handler, no thread, no debug register set).
pub fn arm_title_probe(exe: usize, submission: fn() -> u64) {
    let Some(value) = switch() else {
        return;
    };
    SUBMISSION.store(submission as *const () as usize, Ordering::Relaxed);
    let (points, refused) = parse(value);
    if !refused.is_empty() {
        tracing::warn!(
            ?refused,
            "{SWITCH}: left out (hex RVAs, dll: for {BUILDER}, at most {POINTS})"
        );
    }
    let wide: Vec<u16> = BUILDER.encode_utf16().chain(Some(0)).collect();
    // SAFETY: NUL-terminated name; no reference is taken.
    let builder = unsafe { GetModuleHandleW(wide.as_ptr()) } as usize;
    let mut addresses = [0usize; POINTS];
    for (i, &(base, rva)) in points.iter().enumerate() {
        let at = match base {
            Base::Exe => exe,
            Base::Builder if builder != 0 => builder,
            Base::Builder => {
                tracing::warn!(
                    rva = format_args!("{rva:#x}"),
                    "{SWITCH}: {BUILDER} not mapped, point left out"
                );
                continue;
            }
        } + rva;
        let mut known = [(0, 0); 4];
        let bytes = read_u64(at as u64, &mut known);
        tracing::info!(
            register = i,
            base = ?base,
            rva = format_args!("{rva:#x}"),
            address = format_args!("{at:#x}"),
            bytes = format_args!("{:016x}", bytes.unwrap_or(0).swap_bytes()),
            "{SWITCH}: execution breakpoint"
        );
        addresses[i] = at;
        RVAS[i].store(rva, Ordering::Relaxed);
    }
    if addresses.iter().all(|&a| a == 0) {
        tracing::info!("{SWITCH}: no breakpoints, only the {BUILDER} import wrappers");
        return;
    }
    ring();
    for (slot, &a) in ADDRESSES.iter().zip(&addresses) {
        slot.store(a, Ordering::Relaxed);
    }
    // SAFETY: the handler reads the exception it is given and writes only
    // the ring and that context's RF and Dr6.
    unsafe { AddVectoredExceptionHandler(1, Some(on_exception)) };
    std::thread::Builder::new()
        .name("title probe".into())
        .spawn(move || {
            let mut ids = Vec::with_capacity(1024);
            let mut last_arm = None::<std::time::Instant>;
            let mut last_status = std::time::Instant::now();
            let mut armed_total = 0u64;
            loop {
                if last_arm.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(2)) {
                    other_threads(&mut ids);
                    let armed = ids.iter().filter(|&&id| arm_thread(id, &addresses)).count();
                    armed_total += armed as u64;
                    if armed > 0 {
                        tracing::info!(armed, threads = ids.len(), "{SWITCH}: threads armed");
                    }
                    last_arm = Some(std::time::Instant::now());
                }
                drain(log_hit);
                if last_status.elapsed() >= std::time::Duration::from_secs(60) {
                    let hits: Vec<u64> = HITS.iter().map(|h| h.load(Ordering::Relaxed)).collect();
                    tracing::info!(
                        ?hits,
                        lost = LOST.load(Ordering::Relaxed),
                        armed = armed_total,
                        "{SWITCH}: status"
                    );
                    last_status = std::time::Instant::now();
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        })
        .ok();
}

/// The DLL function a wrapper calls (0: none).
static REHYDRATE: AtomicUsize = AtomicUsize::new(0);
static BUILD: AtomicUsize = AtomicUsize::new(0);
/// Build calls, entries by type (0 top, 1 bottom, other), triangle geometries.
static BUILD_COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Eight arguments forwarded as they came: the DLL functions take five
/// (unpacking) and an unknown number (build); the wrappers pass rcx, rdx,
/// r8, r9 and the next four stack slots on unchanged, so a callee taking
/// up to eight sees what its caller passed.
pub(crate) type Forward = unsafe extern "system" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64;

/// The import of `symbol` from `dll` resolved to `address`: the address to
/// bind instead (a wrapper) with the probe on, or `address`.
pub(crate) fn wrap_import(dll: &str, symbol: &ImportSymbol, address: usize) -> usize {
    if !enabled() || !dll.eq_ignore_ascii_case(BUILDER) {
        return address;
    }
    let rva = module_base(address).map(|base| address - base);
    let wrapped = match rva {
        Some(REHYDRATE_RVA) => {
            REHYDRATE.store(address, Ordering::Relaxed);
            rehydrate as *const () as usize
        }
        Some(BUILD_RVA) => {
            BUILD.store(address, Ordering::Relaxed);
            build as *const () as usize
        }
        _ => address,
    };
    tracing::info!(
        %symbol,
        rva = format_args!("{:#x}", rva.unwrap_or(0)),
        wrapped = wrapped != address,
        "{SWITCH}: {BUILDER} import"
    );
    wrapped
}

/// One unpacking entry as the DLL gets it: destination (+00), source
/// (+20), header address and the D0 the CPU reads there.
fn rehydrate_entries(n: u64, desc: u64, headers: u64) -> Vec<String> {
    let mut known = [(0, 0); 4];
    (0..n.min(64))
        .map(|i| {
            let e = desc.wrapping_add(i * 0x30);
            let dst = read_u64(e, &mut known);
            let src = read_u64(e.wrapping_add(0x20), &mut known);
            let header = if headers == 0 {
                None
            } else {
                read_u64(headers.wrapping_add(i * 8), &mut known)
            };
            let d0 = header.and_then(|h| read_u32(h, &mut known));
            let hex = |v: Option<u64>| v.map_or("?".to_owned(), |v| format!("{v:#x}"));
            format!(
                "{}/{}/{}/{}",
                hex(dst),
                hex(src),
                hex(header),
                hex(d0.map(u64::from))
            )
        })
        .collect()
}

/// Ordinal 8, `RehydrateRaytracingAccelerationStructureBatched2X(this,
/// list, n, desc, headers)`: logged with its result, passed through.
unsafe extern "system" fn rehydrate(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
) -> u64 {
    let sequence = next();
    let n = c & 0xffff_ffff;
    let entries = rehydrate_entries(n, d, e);
    // SAFETY: set from the import's own address before the wrapper was bound.
    let original: Forward = unsafe { std::mem::transmute(REHYDRATE.load(Ordering::Relaxed)) };
    // SAFETY: the caller's arguments, unchanged.
    let result = unsafe { original(a, b, c, d, e, f, g, h) };
    tracing::info!(
        "TITLE PROBE dll rehydrate seq={sequence} s={} tid={} n={n} desc={d:#x} headers={e:#x} hr={:#010x} entries(dst/src/header/D0)=[{}]",
        submission(),
        // SAFETY: a plain query.
        unsafe { GetCurrentThreadId() },
        result as u32,
        entries.join(" ")
    );
    result
}

/// The batched build: counts types, triangle geometries, formats and
/// vertex/index addresses (Fable Aufruf 14 P14); the first 64 calls in full.
unsafe extern "system" fn build(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
) -> u64 {
    let calls = BUILD_COUNTS[0].fetch_add(1, Ordering::Relaxed);
    let n = c & 0xffff_ffff;
    let mut known = [(0, 0); 4];
    let mut detail = Vec::new();
    for i in 0..n.min(4096) {
        let entry = d.wrapping_add(i * 0x50);
        let (Some(kind), Some(counts), Some(geometry)) = (
            read_u32(entry + 0x08, &mut known),
            read_u64(entry + 0x10, &mut known),
            read_u64(entry + 0x18, &mut known),
        ) else {
            continue;
        };
        let (count, layout) = (counts & 0xffff_ffff, counts >> 32);
        BUILD_COUNTS[1 + (kind as usize).min(2)].fetch_add(1, Ordering::Relaxed);
        let mut first = String::new();
        if kind == 1 && count > 0 {
            let g0 = if layout == 1 {
                read_u64(geometry, &mut known)
            } else {
                Some(geometry)
            };
            for j in 0..count.min(256) {
                let at = if layout == 1 {
                    read_u64(geometry.wrapping_add(j * 8), &mut known)
                } else {
                    Some(geometry.wrapping_add(j * 0x38))
                };
                if at.and_then(|at| read_u32(at, &mut known)) == Some(0) {
                    BUILD_COUNTS[4].fetch_add(1, Ordering::Relaxed);
                }
            }
            if let Some(g) = g0
                && let (Some(t), Some(fmts), Some(counts), Some(ib), Some(vb)) = (
                    read_u32(g, &mut known),
                    read_u64(g + 0x10, &mut known),
                    read_u64(g + 0x18, &mut known),
                    read_u64(g + 0x20, &mut known),
                    read_u64(g + 0x28, &mut known),
                )
            {
                first = format!(
                    " g0:type={t},ifmt={},vfmt={},icount={},vcount={},ib={ib:#x},vb={vb:#x}",
                    fmts & 0xffff_ffff,
                    fmts >> 32,
                    counts & 0xffff_ffff,
                    counts >> 32
                );
            }
        }
        if calls < 64 {
            detail.push(format!(
                "[{i}:type={kind},n={count},layout={layout},ptr={geometry:#x}{first}]"
            ));
        }
    }
    // SAFETY: as `rehydrate`.
    let original: Forward = unsafe { std::mem::transmute(BUILD.load(Ordering::Relaxed)) };
    // SAFETY: as `rehydrate`.
    let result = unsafe { original(a, b, c, d, e, f, g, h) };
    if calls < 64 {
        tracing::info!(
            "TITLE PROBE dll build call={calls} s={} n={n} r={:#x} {}",
            submission(),
            result as u32,
            detail.join(" ")
        );
    }
    if calls % 500 == 499 {
        let c: Vec<u64> = BUILD_COUNTS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        tracing::info!(
            calls = c[0],
            top = c[1],
            bottom = c[2],
            other = c[3],
            triangles = c[4],
            "TITLE PROBE dll build counts"
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The probe's statics are the process's: one test at a time.
    static ONE: Mutex<()> = Mutex::new(());

    fn reset() {
        for a in &ADDRESSES {
            a.store(0, Ordering::Relaxed);
        }
        drain(|_| {});
    }

    #[test]
    fn parses_points() {
        let (points, refused) = parse("32619bc, 0x3261a80,dll:219d0,4ee17e6,ea6d2d,zz");
        assert_eq!(
            points,
            vec![
                (Base::Exe, 0x32619bc),
                (Base::Exe, 0x3261a80),
                (Base::Builder, 0x219d0),
                (Base::Exe, 0x4ee17e6)
            ]
        );
        assert_eq!(refused, vec!["ea6d2d", "zz"], "at most four, and hex only");
        assert_eq!(dr7(&[1, 0, 1, 1]), 0b01_01_00_01);
    }

    /// Red proof (Aufruf 14 P13): a synthetic breakpoint hit at one of our
    /// addresses is recorded with its register and the record a register
    /// points at, RF is set and Dr6 cleared; a hit elsewhere, a plain single
    /// step and other exceptions are left alone.
    #[test]
    fn handler_takes_a_synthetic_hit() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        ring();
        // B: u64[B+20] - u64[B+28] = 0x150 - 0x100 = k = 0x50, the header
        // D0 at B + 0x50.
        let mut b = [0u64; 12];
        let k = 0x50u64;
        (b[4], b[5], b[10]) = (0x150, 0x100, 0x0100_8102);
        let mut e_record: [u64; 9] = [1 << 9, 0, 0x1234_5600, 0, 0, 0, 0, 0, 1 << 14];
        e_record[3] = b.as_ptr() as u64;
        ADDRESSES[2].store(0x7ff6_0000_1000, Ordering::Relaxed);
        // SAFETY: zeroed CONTEXT/EXCEPTION_RECORD are valid.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        let mut record: EXCEPTION_RECORD = unsafe { std::mem::zeroed() };
        record.ExceptionCode = EXCEPTION_SINGLE_STEP;
        context.Rip = 0x7ff6_0000_1000;
        context.Dr6 = 1 << 2;
        context.R9 = e_record.as_ptr() as u64; // E in r9
        context.Rbx = 0x55;
        assert!(
            take(&record, &mut context),
            "a hit at our Dr2 address is ours"
        );
        assert_ne!(context.EFlags & RF, 0, "RF set");
        assert_eq!(context.Dr6, 0);
        let mut hits = Vec::new();
        drain(|h| hits.push(*h));
        assert_eq!(hits.len(), 1);
        let h = hits[0];
        assert_eq!((h.point, h.rip, h.regs[3]), (2, 0x7ff6_0000_1000, 0x55));
        let r9 = h.records[9];
        assert_eq!(r9.read, 7, "fields, k and D0 read");
        assert_eq!(r9.fields, [1 << 9, b.as_ptr() as u64, 0, 0, 1 << 14]);
        assert_eq!(
            (r9.head[0], r9.head[1], r9.head[2]),
            (b.as_ptr() as u64, k, 0x0100_8102)
        );
        assert_eq!(h.records[3].read, 0, "0x55 is no readable record");
        assert_eq!(h.packed_byte, 0xffff, "rbx + rdi + 3 = 0x58 cannot be read");
        // At 3261a80 rax = E + 0x40; at the wrapper rbx = B, rdi = k.
        context.Rip = 0x7ff6_0000_1000;
        context.Dr6 = 1 << 2;
        context.Rax = e_record.as_ptr() as u64 + 0x40;
        context.Rbx = b.as_ptr() as u64;
        context.Rdi = k;
        assert!(take(&record, &mut context));
        let mut again = Vec::new();
        drain(|h| again.push(*h));
        assert_eq!(again.len(), 1);
        assert_eq!(
            (again[0].below_rax.read, again[0].below_rax.head[2]),
            (7, 0x0100_8102)
        );
        assert_eq!(again[0].packed_byte, 0x01, "byte 3 of D0 0x01008102");
        context.Rbx = 0x55;
        context.Rdi = 0;
        // Not ours: another address, a plain single step, another exception.
        context.Rip = 0x7ff6_0000_2000;
        context.Dr6 = 1 << 2;
        assert!(!take(&record, &mut context));
        context.Rip = 0x7ff6_0000_1000;
        context.Dr6 = DR6_BS;
        assert!(!take(&record, &mut context));
        record.ExceptionCode = windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        context.Dr6 = 1 << 2;
        assert!(!take(&record, &mut context));
        drain(|_| panic!("nothing else recorded"));
        reset();
    }

    /// The header offset k is 0x38 when u64[B+20] - u64[B+28] underflows.
    #[test]
    fn header_offset_underflows_to_0x38() {
        let b: [u64; 8] = [0, 0, 0, 0, 0x10, 0x20, 0, 0x0000_0000_0100_8002];
        let e: [u64; 9] = [0, 0, 0, b.as_ptr() as u64, 0, 0, 0, 0, 0];
        let mut known = [(0, 0); 4];
        let r = record_at(e.as_ptr() as u64, &mut known);
        assert_eq!((r.read, r.head[1], r.head[2]), (7, 0x38, 0x0100_8002));
    }

    #[inline(never)]
    fn target(x: u64) -> u64 {
        std::hint::black_box(x.wrapping_mul(3).wrapping_add(1))
    }

    /// Runs `target` three times on a thread of its own, armed (or not)
    /// with Dr0 at `target`, and returns the hits recorded at Dr0.
    fn run_armed(arm: bool) -> usize {
        reset();
        ring();
        let at = target as *const () as usize;
        let addresses = [at, 0, 0, 0];
        if arm {
            ADDRESSES[0].store(at, Ordering::Relaxed);
        }
        static HANDLER: OnceLock<()> = OnceLock::new();
        // SAFETY: as in `arm_title_probe`.
        HANDLER.get_or_init(|| unsafe {
            AddVectoredExceptionHandler(1, Some(on_exception));
        });
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            // SAFETY: a plain query.
            ready_tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            go_rx.recv().unwrap();
            let s: u64 = (0..3).map(target).sum();
            done_tx.send(s).unwrap();
        });
        let id = ready_rx.recv().unwrap();
        if arm {
            assert!(arm_thread(id, &addresses), "debug registers set");
        }
        go_tx.send(()).unwrap();
        // Without RF the thread breaks at `target` for ever: a timeout, not a hang.
        let sum = done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("target ran to its end (RF)");
        assert_eq!(sum, 1 + 4 + 7);
        worker.join().unwrap();
        let mut n = 0;
        drain(|h| {
            if h.point == 0 && h.rip as usize == at {
                n += 1;
            }
        });
        ADDRESSES[0].store(0, Ordering::Relaxed);
        n
    }

    /// Red proof (Aufruf 14 P13): a hardware breakpoint on a test function
    /// counts every call and lets it run on (RF); unarmed, nothing counts.
    #[test]
    fn hardware_breakpoint_counts_calls() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(run_armed(false), 0, "no debug register set, no hit");
        assert_eq!(run_armed(true), 3, "one hit per call");
    }

    /// Without the switch nothing is armed: no handler, no thread, no debug
    /// register (the test process has no `VICTORY_TITLE_PROBE`).
    #[test]
    fn off_without_switch() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            std::env::var_os(SWITCH).is_none(),
            "tests run without the switch"
        );
        reset();
        let (tx, rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            // SAFETY: a plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            go_rx.recv().unwrap();
        });
        let id = rx.recv().unwrap();
        arm_title_probe(0x1_4000_0000, || 0);
        assert!(!enabled());
        assert!(
            ADDRESSES.iter().all(|a| a.load(Ordering::Relaxed) == 0),
            "no address configured"
        );
        // SAFETY: as in `arm_thread`.
        let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, id) };
        assert!(!thread.is_null());
        // SAFETY: zeroed is a valid CONTEXT.
        let mut c: Aligned = unsafe { std::mem::zeroed() };
        c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
        // SAFETY: suspended for the read, resumed after.
        let dr7 = unsafe {
            SuspendThread(thread);
            let ok = GetThreadContext(thread, &mut c.0);
            ResumeThread(thread);
            CloseHandle(thread);
            assert_ne!(ok, 0);
            c.0.Dr7
        };
        assert_eq!(dr7 & 0xff, 0, "no debug register enabled");
        assert_eq!(
            wrap_import(BUILDER, &ImportSymbol::Ordinal(8), 0x1234),
            0x1234,
            "imports bound as resolved"
        );
        go_tx.send(()).unwrap();
        worker.join().unwrap();
    }

    extern "system" fn echo(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64) -> u64 {
        a ^ b.rotate_left(8)
            ^ c.rotate_left(16)
            ^ d.rotate_left(24)
            ^ e.rotate_left(32)
            ^ f.rotate_left(40)
            ^ g.rotate_left(48)
            ^ h.rotate_left(56)
    }

    /// The wrappers pass arguments and result through unchanged.
    #[test]
    fn wrappers_pass_through() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        REHYDRATE.store(echo as *const () as usize, Ordering::Relaxed);
        BUILD.store(echo as *const () as usize, Ordering::Relaxed);
        let args = [0x11u64, 0x22, 0, 0, 0x55, 0x66, 0x77, 0x88];
        let want = echo(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        );
        // SAFETY: `echo` takes the eight arguments; n = 0 reads nothing.
        let (r, b) = unsafe {
            (
                rehydrate(
                    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                ),
                build(
                    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                ),
            )
        };
        assert_eq!((r, b), (want, want));
        REHYDRATE.store(0, Ordering::Relaxed);
        BUILD.store(0, Ordering::Relaxed);
    }
}
