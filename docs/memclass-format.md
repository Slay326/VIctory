# Memclass: stage 1 classification log and event stream (format v1)

Status: stable for the replayer (night 2). Changes only with a new version
number in the header; a reader must refuse versions it does not know.

Source: `crates/victory-gpu/src/memclass.rs` (writer),
`memclass_stream.py` (reader, `--check` against counters) and
`memclass_table.py` (classification table from the log): local tools of the night
agents (folder `scripts/memclass/` of the agent workspace), not in the repo
(the asset policy allows no `.py`). Everything they read is described here.

## Switches

**Runs with `VICTORY_MEMCLASS` or `VICTORY_MEMCLASS_STREAM` give no valid
performance numbers** (frame times, fps, memory): they are for classifying
and capturing only. The log's `memclass on` line says so too. A/B/A and
fps measurements run without both switches.

The switches are read once when the host starts (`memclass::init`, at the
start of `victory run`); hooks never read them.

| Variable | Values | Meaning |
|---|---|---|
| `VICTORY_MEMCLASS` | `1` / anything else | Classification on. Off: every hook is one load of a flag; no memory, no atomics written, no stream. |
| `VICTORY_MEMCLASS_GRAN` | `64k` (default), `4k` | Unit the per-frame flags and the table count in. GPU write marks are 64 KiB whatever the unit (a 4k unit gets the 16 units of the marked page). |
| `VICTORY_MEMCLASS_STREAM` | file path | Also write the event stream (needs `VICTORY_MEMCLASS=1`). |
| `VICTORY_MEMCLASS_STREAM_FRAMES` | n, default 0 | End the stream after n frames (0: whole run). |
| `VICTORY_MEMCLASS_INTERVAL_S` | s, default 5 | Seconds between `memclass interval` log lines. |
| `VICTORY_MEMCLASS_TABLE_S` | s, default 60 | Seconds between full tables in the log (the last complete one counts). |

## Access kinds (flag bits)

The same bits are used in the log (`ever=0x....`) and in the stream (`flag`).

| Bit | Name | Where it is seen | Notes |
|---|---|---|---|
| 0 | `cpu_w_title` | guard handler (`watch.rs`, `on_write`), fault address, length 1 | only the first write after a copy took the page faults; per-frame counts are low bounds |
| 1 | `cpu_w_runtime` | `kernel_will_write` (file reads, DirectStorage, page clearing) | kept apart from bit 0 (F1 question 4) |
| 2 | `cpu_r_after_gpu` | reserved, hook `memclass::cpu_read` not called | F1 question 3 open; never set in v1 |
| 3 | `gpu_w_mark` | `copy_back_marks`: pages shaders marked (also marked pages withheld) | 64 KiB |
| 4 | `gpu_w_decl` | `Shadow::writing`: range a submission declares shaders write | may over-cover |
| 5 | `gpu_w_back` | `write_runs_by`: spans a copy-back writes into title memory | only if every span landed (a copy-back that did not reach title memory counts neither as GPU write nor as traffic); one ACCESS and one SYNC back per span, the time shared by bytes |
| 6 | `gpu_w_copy` | copy steps into title memory, texture read-backs (`list.rs`) | |
| 7 | `gpu_r_refresh` | `Step::Refresh`: range a list asked to be brought up to date | one record per step = copystat `Refreshes` |
| 8 | `gpu_r_heap` | `Step::Heap`: ranges a heap's views reach, at submission | |
| 9 | `gpu_r_table` | `memory::prepare`: table/log: every copy the memory page table names (written or not); stream: only the copies the table hands to the refresh (not clean ones: ~4,700 copies per list would dominate the stream) | per list using the table |
| 10 | `gpu_w_kept` | `memory::drop_withheld`: GPU results dropped from the copies | |
| 11 | `remap` | `title_memory_remapped` | |
| 12 | `copied_up` | a refresh copied the unit (SYNC up); units only, no ACCESS record | |
| 13 | `cpu_w_dirty` | derived at the frame end, units only: `copied_up` in this frame, the unit was copied before, and none of `gpu_w_mark`, `gpu_w_back`, `gpu_w_copy`, `gpu_w_kept`, `cpu_w_runtime`, `remap` in this frame or the one before | the title's CPU wrote it although no fault was seen (a page left open after its first write faults no more; Fable P2 A3). A replayer derives it the same way from SYNC up records |

GPU-read proxies are bits 7-9; the reader picks which count (F1 question 2).

## Log lines (target `victory_gpu::memclass`)

All `key=value`, space separated.

- `memclass interval frame=F frames=N secs=S unit=U sync_up_mb sync_up_ms sync_back_mb sync_back_ms <flag>_mb... gpu_only_mb cpu_only_mb shared_mb unknown_mb lost=L [stream_mb stream_mb_per_min stream_on]`
  - per frame over the interval: sync MB/ms, MB of units with each flag; classes so far under the proposed rule (Q2); `lost` = tags dropped (no chunk slot left, should be 0); stream size so far and **MB per minute** (for sizing an 8-min capture).
- `memclass table begin frame=F unit=U regions=R` ... `memclass table end frame=F`
  - `memclass region id=I kind=K at=A len=B f_<flag>=frames ... sync_up_bytes sync_up_us sync_back_bytes sync_back_us`
    - kind: `virtual` (write-watched `XMemVirtualAlloc`), `virtual_unwatched`, `section` (physical pages; `at` = section handle; all views of it count here at the section offset), `unnamed` (per 4 GiB window, nothing named it).
    - `f_<flag>` = frames in which any unit of the region had the flag.
    - sync bytes/us: a range over several regions is cut at region borders, each piece's time by its bytes.
  - `memclass units region=I ever=0xHHHH units=N bytes=B`: units whose flags over the run are exactly `ever`.

Class rule (applied by the reader; default = proposal in night-queue Q2):
CPU = `cpu_w_title|cpu_w_dirty|cpu_r_after_gpu` (optionally `cpu_w_runtime`), GPU = all `gpu_*` (or a subset).
CPU and GPU = shared; CPU only = CPU-only; GPU only = **GPU-only candidate** (no CPU write seen; CPU read after GPU write is not measured in v1, so not proven GPU-only; `gpu_r_table` is an upper bound of GPU reads) **if** the region kind is `virtual` or `section`, else unknown; nothing = unknown.

## Event stream (binary)

Little-endian. `varint` = unsigned LEB128 (7 bits per byte, low first, high
bit = more). `zigzag(d)` = `(d << 1) ^ (d >> 63)` on i64.

### Header (8 bytes)

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `VMCS` |
| 4 | 2 | version, u16 = 1 |
| 6 | 1 | unit shift (12 or 16; the run's `VICTORY_MEMCLASS_GRAN`) |
| 7 | 1 | reserved, 0 |

### Records

Each record: tag (varint, all v1 tags < 0x80 so one byte), then fields in
this order. Single-byte fields (`kind`) are values < 0x80 and may also be
read as varints.

| Tag | Name | Fields |
|---|---|---|
| 1 | ACCESS | `flag` varint (one bit of the table above), `tid` varint (Windows thread id), `seq` varint (number of the latest submission begun, 0 before the first), `at` varint = zigzag(address - previous ACCESS address; first: address - 0), `len` varint (bytes) |
| 2 | SUBMIT | `seq` varint (1, 2, ...), `lists` varint |
| 3 | POINT | `kind` u8 (0 executed: a group of lists goes to the GPU; 1 gpu_done: the GPU finished, copy-backs follow; 2 end: copy-backs done), `seq` varint |
| 4 | SYNC | `kind` u8 (0 up: title memory -> host copy; 1 back: GPU bytes -> title memory), `at` varint (absolute), `len` varint (bytes), `us` varint |
| 5 | FRAME | `frame` varint (1, 2, ...; a presented frame ends here) |
| 6 | REGION | `kind` u8 (0 virtual unwatched, 1 virtual watched, 2 section, 3 view), 4 varints: virtual `at, len, 0, 0`; section `handle, bytes, 0, 0`; view `at, len, section handle, offset` |
| 7 | END | `frame` varint; the last record |
| 8 | COUNTERS | follows every FRAME: 4 varints, totals since the stream began: `refreshes` (copystat `Refreshes`), `faults` (copystat `GuardFaults`), `copied` (bytes the submission refreshes copied, the part of the host line's "copies MB" that is not copy steps), `copy_steps` (copystat `CopyBytes`: copy/fill steps). copystat values are 0 in builds without `diagnostics`. |

### Ordering and meaning

- File order is the order records were made, across threads (one lock).
  Title threads' ACCESS records (`cpu_w_title`, `cpu_w_runtime`) interleave
  with the submitting thread's records as they happened.
- A submission is `SUBMIT seq` ... `POINT end seq`. Inside it, in order:
  the preparation (ACCESS `gpu_r_*`, `gpu_w_decl`, `gpu_w_copy`; SYNC up for
  refresh copies), then per group of lists `POINT executed`, then
  `POINT gpu_done`, then copy-backs (ACCESS `gpu_w_mark`, `gpu_w_back`,
  `gpu_w_kept`; SYNC back), then `POINT end`. One submission at a time
  (the runtime holds a lock from preparation to copy-back).
- `seq` in ACCESS: the submission the access happened during or after;
  a title-thread ACCESS with `seq = s` that comes after `POINT end s` is
  between submissions s and s+1.
- Records come per hook call, not per unit: one ACCESS may cover many
  units (`len`), and the same range may repeat (every list asks for its
  refreshes). A replayer expands ranges to units itself (header shift).
- `REGION` records come when the runtime allocates or maps; a view
  replaces whatever named its addresses before; an unmap (ACCESS `remap`
  without a following view) leaves the addresses unnamed.
- A stream without END was cut off (crash, kill): read the whole records
  up to the cut; the last frame may be incomplete.

### Check against counters (gate item 2)

`memclass_stream.py <file> --check`: per frame, ACCESS `gpu_r_refresh`
records = delta `refreshes`, ACCESS `cpu_w_title` records = delta `faults`,
SYNC up bytes = delta `copied`. Totals must be equal; a single event on
the other side of a FRAME record (two statements apart on another thread)
is listed and allowed if the running difference is back to 0 within
`--tolerance-frames` (default 1).

### Size

The interval line logs `stream_mb_per_min`. Estimate before any run
(computed by a script from play baseline 192359 per-frame counters, PC 1,
with an assumed ~12 bytes per record and 3,000-30,000 records per frame
not covered by copystat; NOT measured): 57-113 MB per minute, 0.45-0.91 GB
for 8 minutes. Measure it with the first 180 s run before a full capture.

## Shortcuts (v1)

- `cpu_r_after_gpu` is never set (F1 question 3).
- Memory freed and allocated again keeps its old region until a new
  allocation or view names it; `XMemVirtualAlloc` placeholders are not a
  region (their views are, via the section).
- Card copies (VRAM) and snapshots are not SYNC records (copystat has them).
- The guards see the title's first write after a take only.
