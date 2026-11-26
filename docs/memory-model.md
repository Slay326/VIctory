# Memory model (stage 2 draft)

Status: DRAFT for review. Decisions still open are marked
**DECISION (Fable):**; nothing marked so is settled. Number marks:
[M] measured, [C] computed by a script from measured data, [E] estimated.

## 1. Target model

Today every title buffer the GPU uses has a host copy, refreshed before and
copied back after submissions, as if memory were unified. Target:

| Class | Where the bytes live | Sync |
|---|---|---|
| GPU-only | on the card (copy only) | none: no refresh, no copy-back, no title mirror |
| CPU-only | title memory only | never goes to the card |
| shared | title memory + copy | only at explicit sync points (before a GPU read, at the CPU's next touch) |
| unknown | as today | as today |

Rule: a wrong class may cost speed, never correctness. A page leaves
GPU-only the moment the title touches it (section 4) and never returns.

Bookkeeping moves from per copy to per submission (pages read/written,
sync points needed): the base for flights. Today `HostDevice::execute`
waits for each group's fence under the `submitted` lock and copies back
synchronously after it.

**DECISION (Fable):** record format and owner of the per-submission
bookkeeping (memclass stream records, PageNet, or a new table), and whether
stage 4 starts with it or with the next block.

## 2. Stage 1 findings (classification)

Source: memclass streams of one 8-min run per PC (instrumented, so ms are
upper bounds and not performance numbers). Capture runs count for content
per submission only, never for rates or traffic.

Classes by size, last full table [C from M]:

| | PC 1 (3070) | PC 2 (4070) |
|---|---|---|
| GPU-only candidates | 4,891 MB (26.6 %) | 4,618 MB (26.3 %) |
| CPU-only | 51 MB (0.3 %) | 44 MB (0.2 %) |
| shared | 518 MB (2.8 %) | 508 MB (2.9 %) |
| unknown | 12,898 MB (70.3 %) | 12,404 MB (70.6 %) |

"Candidate" = GPU writes, no CPU write seen, CPU read not measured in
stage 1. Largest blocks, same order on both PCs: a 6.5 GB section (about
3.3 GB candidates, 390 MB shared), a pool of ~0.8-1.0 GB, and a 744 MB
section with no CPU bit at all (392 MB touched).

Sync traffic per frame, median [C]: PC 1 up 137 MB / 15.6 ms, back
51 MB / 43.6 ms; PC 2 up 134 MB / 29.5 ms, back 50 MB / 62.9 ms.

Unknown share of sync traffic [C]: PC 1 0.1 % of MB / 0.4 % of ms, PC 2
(world) 0.1 % / 0.6 %: the 70 % unknown by size (unnamed 4 GiB windows,
untouched section units) is almost traffic-free. GPU-only candidates carry
81-83 % of sync ms, so the open point is proving them (PageNet measure).

Gate 2 check [M]: stream counters match copystat on both PCs.

## 3. PageNet

One module, two modes: `measure` (stage 1 read measurement) and `net`
(stage 3 safety net). Code: `crates/victory-gpu/src/watch/net.rs` on branch
`block1/pagenet` (not in `play`), `VICTORY_PAGENET=measure`, default off;
`net` is not accepted yet.

- Unit: state per 64 KiB title page; faults counted per 4 KiB (read/write)
  to expose false sharing.
- Mechanism: `PAGE_NOACCESS` and the existing vectored handler (called at
  the start of `watch.rs` `on_write`, so the order is fixed and read
  faults are seen); not `PAGE_GUARD` (one-shot, races between threads).
- Only pages under the write guard (read-only, in the guard set) and all
  their aliases are armed; opening restores read-only, under the guard
  lock. Engine reads through `host::read` open armed pages uncounted
  (host touch), so measure leaves sync results unchanged.
- Interface: `PageNet::new(mode, backend: impl Protect)` (OS-free tests),
  `arm`, `on_fault -> Handled | NotOurs`, `is_gpu_only`, `disarm_for_free`,
  `stats`, `drain_demotions`.
- States: measure: Shared <-> ArmedMeasure. net: Shared -> GpuOnlyArmed ->
  Syncing -> Demoted (final).

Invariants (tests t1-t10 against a mock, each shown red by mutation):
- I1 an armed page has no CPU write since its take.
- I2 on a fault: fence -> copy -> Demoted -> open -> continue (all GPU
  writes submitted to the page are finished and copied first).
- I3 Demoted is never armed or GPU-only again (arm = no-op + counter).
- I4 the first CPU write after demotion is dirty, uploaded before the next
  GPU read.
- I5 two threads faulting on one page: one sync, both continue after it.
- I6 foreign address -> NotOurs, no state change. I7 measure changes no
  byte and no sync decision. I8 host accesses never count as title reads.

Gaps of measure: only copy_back_page (computed pages) arms, declared and
marked copy-back do not; title syscalls reading an armed page fail
(`ERROR_NOACCESS`); faults count as title accesses whoever's code ran
(errs toward shared).

**DECISION (Fable):** arm also after declared/marked copy-back if the two
target sections show few armed pages in the measure run.

### copy_back_page_forced

The net must not rely on the "written since take" path of copy_back_page.
`copy_back_page_forced(page)` never discards (correct by I1) and writes the
GPU bytes through the second view (`Own::beside`), so the title view stays
no-access and no CPU write can be lost in between. Only section memory has
a second view (with `VICTORY_BACK_OWN_VIEW=1`); private pools
(`XMemVirtualAlloc`, write-watched) have none.

**DECISION (Fable):** net path lock/fence order. Today the fault path holds
the guard lock across `on_fault`, and the submit thread holds `submitted`
during execute and copy-back; host writes 4 and 5 (section 5) fault under
`submitted`. Order of guard lock, `submitted` and fence wait to be fixed.

### H-02 (copy_back_page drops GPU bytes)

Finding: if the title wrote anywhere in a 64 KiB page during a submission
and the page had no saved snapshot, copy_back_page returns 0 and keeps the
title's bytes for the whole page; a unit test (GPU at offset A, title at B)
is red today [M]. The red stream replay was a replayer artifact (0
violations after the fix [M]; a replay of a stream with a clean end is
pending). In game logs: once, in a BVH run on PC 2 [M].

Fix path: arm computed pages before the submission (the saved snapshot
exists, put_back diffs bytes), no 4 KiB tracker. Wrong only when title and
GPU write the same byte, where order is undefined anyway. Behind
`VICTORY_BACK_PREARM=1`, default off, own branch. Accept: the copy-back
test green with the switch and red without; cargo test --workspace green;
A-B on PC 1 with the prearm cost in ms/frame (script). Not a precondition
for stage 3 (demoted pages fall back to today's behaviour); needed for the
write-fault case and before default on.

**DECISION (Fable):** review of the prearm diff; whether it is pulled
forward (only if the PC 2 intersection run ties H-02 to BVH zeros).

## 4. First block: the 744 MB section

Choice (block 1): the 744 MB physical-pages section, no CPU bit. Sections
have the second view (needed by the forced path); with no CPU bit every
demotion points at a bug in the net or the classification. The mixed
6.5 GB section is stage 4.

Facts [M]: 744,488,960 bytes; 54,051 views of 64 KiB in one title range of
about 3.5 GB (about 4.8 aliases per page), identical on both PCs. The second
view exists for it in the runs (own view on, no "no view of our own"
warning), but today only copy_back_marked uses it; copy_back_page and
copy_back write through the title view. No runtime (DirectStorage) writes
into it in the runs.

Traffic per world frame, median [C, instrumented run, upper bound]:

| | up | back | share of all back (bytes / time) |
|---|---|---|---|
| PC 1 | 39.2 MB / 4.05 ms | 25.5 MB / 14.1 ms | 47 % / 33 % |
| PC 2 | 39.6 MB / 6.7 ms | 21.9 MB / 20.3 ms | 42 % / 31 % |

Gain estimate, PC 1, cost model on the play baseline [C]: preparation
-3.1 ms (calibrated, +-10 %), copy-back -11.6 ms (NOT calibrated, two-point
check failed), together about -15 ms/frame (344.9 -> 329.6 ms, 2.90 -> 3.03
fps). Upper bound straight from the stream: 18.2 ms. Caveat: the upload
saving holds only if the data stays on the card (the GPU reads the section
again); GPU time and VRAM pressure are not predicted. The number is the
stage 3 prediction (gate point 7), not a result.

**DECISION (Fable):** "this section is GPU-only" (needs zero read faults
over the run on both PCs in measure mode), and the alias handling (arm all
~4.8 title addresses of a page together).

## 5. Host write sites (c5)

Copy steps write title memory today and the copy catches up at the next
refresh. A GPU-only page has no refresh (title view closed, never
"changed"), so for armed pages the host must write the copy directly and
the title not at all (or only at sync, through the second view).

| # | Site | Writes | Lock | Needs for armed pages |
|---|---|---|---|---|
| 1 | Step::Copy -> copy_mapped (buffer copies) | title; copy by refresh | submitted | write copy directly |
| 4 | copy_memory (CopyMemoryToMemory) raw copy | title; copy by refresh | submitted, no `writing` | write copy directly; faults under submitted |
| 5 | Written::write (Write32/64, timestamps, query resolve) | title; copy by refresh | submitted, after copy-backs | as 4 |
| 10 | copy_texture_region readback | title; copy by refresh | submitted (after) | write copy directly |
| 6/7/8 | copy-back (declared / marked / computed) | title + copy | submitted | 8 must use the second view (forced path) |
| 12 | descriptor copies, raw | title | title thread | fault counts as title write (correct) |
| 13/14/15 | ReadFile/fread, DirectStorage (address and page targets) | title (kernel or second view) | title / DS worker | fail on no-access unless the net opens the page; not seen in the section |
| 2, 3, 9, 11 | reads of title, refresh into the copy | copy or nothing | submitted | read via host touch |

**DECISION (Fable):** for 1/4/5/10, write the copy directly and skip the
title, or demote on any host write (simpler, costs speed).

## 6. Migration stages

- Stage 3, `VICTORY_MEMMODEL=1`, default OFF: stop mirroring and copying
  back the 744 MB section; net mode guards it. Commits: c3 forced copy-back
  (test t3), c4 net fault path with fence (t1, t4), c5 wiring for the
  block. Fable writes c3-c5.
- Stage 4, only after stage 3 is proven: the GPU-only part of the 6.5 GB
  section (needs the read-fault table and a per-unit split, since 390 MB
  are shared), or per-submission bookkeeping as the first piece of flights.
  **DECISION (Fable):** which first.

## 7. Risks

- The block turns out CPU-read: stage 3 then proves correctness only; the
  largest candidate with zero read faults on both PCs is taken instead.
- Lock order guard / `submitted` / fence (deadlock if the submit thread
  faults on an armed page); raw host writes 4/5; title syscalls on
  no-access pages fail instead of faulting; false sharing (H-02).
- Measure and memclass runs are slowed: never performance numbers.
- Crashes (about 25 % of runs [E]): plan 8 slots for 5 valid runs.

## 8. Proof per stage

Gate before the first stage 3 game run (all required):
1. Stage 1 table validated on both PCs over the 8-min run; block chosen;
   unknown share stated.
2. Event capture over the whole run including the cell scene; matches
   copystat.
3. Replayer green on that capture with stage 3 logic; demotions per region
   (the stream needs CPU read events first, else green proves nothing).
4. Safety net with tests red/green; interleaved submissions.
5. Poison on the CPU side for the range no longer mirrored (diag build).
6. Run script: `VICTORY_MEMMODEL` in the env-leak test, demotion log
   parsed, switch default off.
7. Prediction (ms, MB) written down before the run; fixed capture frames
   for the PC 2 image check.

After the gate: cargo test --workspace green; PC 1 A-B-A(-B-A) against
the `play` baseline (frame breakdown, RAM, VRAM, fps with spread from clean
runs); PC 2 same capture frames, image difference within run-to-run noise,
full run without crash or device removed, BVH on and off; demotion log
(count, reason). Many demotions = wrong classification, back to stage 1.
Only then may the switch become default on `play`.

## 9. Reusing the RT arena

The RT branch (`rt/play`, `arena.rs`, `VICTORY_BVH_ARENA=1`, not in
`play`) already keeps GPU-only data on the card: a 64 KiB page directory,
resident 64 MiB chunks, view-heap page tables naming arena pages, no
write-back to title memory, late title writes uploaded per 4 KiB (write
watch with reset). Reusable: the directory and page-table indirection, the
residency budget, the 4 KiB late upload. Not as is: no net (write watch
only, which section views lack), stale bytes for unwatched memory.

**DECISION (Fable):** whether the stage 3 block lives in arena pages or in
its existing copy with refresh/copy-back switched off.
