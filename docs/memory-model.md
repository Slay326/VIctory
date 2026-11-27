# Memory model (stage 2 draft)

Status: DRAFT for review. Decisions are marked **DECISION (Fable):** and
answered below each (block 2, block 3); after block 3 none is open except
where an answer says what it waits for. Number marks:
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

**ANSWER (Fable, block 2):** owner: a new table in `victory-gpu`, keyed by
submission number, fed by the hooks memclass already has. Not the memclass
stream (diagnostics, may be off) and not PageNet (page state, no notion of
a submission). Record format: open until stage 3 has run; stage 3 needs
none (see the lock order answer: its sync point is the `submitted` lock).
Order: see section 6.

**ANSWER (Fable, block 3):** closed for stage 3. The net as built (c4,
50140bb) needs no per-submission record: every sync runs with `submitted`
held, and the lock names its holder. The record format is left open on
purpose until the stage 3 result is in (see section 6); it does not block
this document.

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

**ANSWER (Fable, block 2):** yes, done: `block1/pagenet` b777960. The run
showed zero armed pages [M], computed pages never go back in the world.
All three ways arm now; a marked page written through our own view arms the
title's view and its aliases. A copy-back with nothing to put back closes
its page again if it was closed (its own read opens it; without this every
such copy-back left earlier GPU bytes open to uncounted reads). "Gaps of
measure" above is outdated in its first clause. Left: reads in the
microseconds between a copy-back's read and the re-arming are not counted,
so zero reads means "very likely GPU-only"; the proof stays the net.

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

**ANSWER (Fable, block 2):** order `submitted` -> guard -> PageNet, and no
fence wait of its own in stage 3: no GPU work is in flight outside
`submitted` today, so "holding `submitted`" is I2.
1. Fault, under guard + PageNet: GpuOnlyArmed -> Syncing (this thread owns
   the sync), then both locks are released. The guard is never held while
   waiting for `submitted` (the submit thread takes the guard inside it).
2. Other threads faulting on a Syncing page wait for its owner without any
   lock (I5).
3. The owner takes `submitted`, brings the page down (card fetch where the
   copy is in GPU memory) and writes all of it through our own view
   (`copy_back_page_forced`; the title's view stays closed, I1 makes the
   whole page correct), then under the guard: Demoted, open, wake.
4. The thread that holds `submitted` faulting itself (host writes 4/5, raw):
   recognised by a named holder, as the guard lock's; it syncs without
   taking the lock. Correct at any point of a submission: before execute
   nothing is in flight, after it the fence was waited for.
5. Kernel copies (site 1, reads 2/3/9/11) do not fault: they ask the net
   first, as `host_touch` does in measure; in net mode that is a demotion
   with sync, counted as a host access.
Open, to be checked in code before c4: that no code under `submitted`
waits for a title thread (then a title thread faulting while holding a
title lock could deadlock). Assumed not; not verified.

**ANSWER (Fable, block 3):** built as answered, with three changes (c4,
50140bb), all read in code and covered by tests:
- Full order: the runtime's page lock (it reports unmaps and new mappings
  holding it) -> `submitted` -> one sync at a time -> `OWN` (the forced
  copy-back) -> guard -> PageNet. `victory-gpu` never calls into the
  runtime (its one callback is the failure report), so nothing holds
  `submitted` and waits for the page lock [read in code].
- The fault looks at the page's state with no lock, takes `submitted`
  first, and the guard only for the last step (demote and open). Threads
  faulting on a page being synced wait on `submitted`, not on a state.
- No sync ever starts under the guard lock: whatever takes the guard lock
  for a write it announces syncs before taking it, and the guard never
  opens a closed page (it stays closed; the next access syncs).
- The copy-back's worker threads count as the holder of `submitted`.
Under `submitted` the code waits for the GPU's fences and for its own
worker threads only [read in code: `execute`, `card_fetch`,
`copy_back_marks`]. Not verified, and on the gate sheet: that no title or
runtime thread touches title memory by raw pointer while it holds a lock
the submitting thread takes inside `submitted` (descriptor copies, site
12, are the candidate). Such a case hangs; it cannot corrupt.

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

**ANSWER (Fable, block 2):** no diff to review yet. Not pulled forward: the
PC 1 stream shows no H-02 case with the fixed replayer [M]; it stays tied
to the result of the PC 2 intersection run.

**ANSWER (Fable, block 3):** closed for this document: prearm is not part
of stage 3 and no diff exists. Stage 3 does not depend on it: a page is
closed only after a copy-back that left title memory and the copy equal,
and a page the title wrote meanwhile is refused (counted as
`refused_changed`). If a prearm diff is written later it gets its own
review; H-02 stays a separate item.

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

**ANSWER (Fable, block 2):** not decided; it needs the measure run of
b777960 and one more fact (below).
- Criterion: per page, not per section. The section is the first block if,
  on both PCs, the pages with a title read or write are a small part of its
  armed pages and the faults per world frame fall to about zero after the
  first seconds. "Zero read faults in the whole section" is not required:
  the net demotes single pages. The threshold is set from the run's table
  (script), not in advance.
- Aliases: all title addresses of a page are armed together or the page is
  not GPU-only (as measure does); one state per physical page; a fault at
  any address demotes all of them.
- NEW, a precondition the net does not cover (hypothesis, not verified):
  the section has no CPU write but 39 MB per frame go up. If that is other
  host copies of the same pages (the same address, or one of the ~4.8 alias
  addresses) being refreshed from title memory after our copy-back marked
  them changed, then title memory is how the GPU's bytes get from one copy
  to another. Stop the copy-back and those copies go stale with no CPU
  access involved: no fault, no demotion, wrong image. Rule for stage 3
  either way: a page is GPU-only only while exactly one copy holds its
  physical page (no other holder at its address or at an alias); else it
  stays shared. Data needed first (Opus, script from the stream or a
  counter): of the section's pages, how many are held by more than one
  copy, and what share of its "up" bytes follows a copy-back's mark rather
  than a title write.

**ANSWER (Fable, block 3):** GO for the 744 MB section, with the alias
condition.
- Evidence: measure runs of b777960 on both PCs, about 640 and 650 world
  frames: 949,594 and 796,882 arms of the section's pages, 0 read and 0
  write faults [M]. That is "very likely GPU-only"; the proof is the net.
- The hypothesis above is confirmed [C, alias_copies.py]: 93.6 % (PC 1)
  and 99.9 % (PC 2) of the section's "up" bytes follow a GPU write at
  another address of the same physical page, 0 % a title write; 74 % and
  49 % of the physical pages with "up" bytes are held by more than one
  copy.
- So, as built (c4/c5): a page is closed only while exactly one tracker,
  the closing copy's, holds its physical page over all its addresses, and
  all of them are under the write guard. Pages with more holders stay on
  today's path, counted as `refused_holders`, not as demotions. A holder
  that comes later reads title memory first, which syncs and demotes
  (`host`). All addresses of a page close together and leave together by
  one sync; a new mapping is closed with them; an unmap syncs first
  (`remap`); withheld shader writes take the page out (`withheld`).
- Consequence for the prediction: stage 3 takes only the single-holder
  pages out of mirroring and copy-back. Most of the section's "up" traffic
  is alias catch-up and stays. Stage 3 mainly proves the net; the gain of
  this section needs "one copy per physical page" (section 6).
- The block is named at run time by the section's size in bytes
  (`VICTORY_MEMMODEL_SECTION`, from the stream's REGION record; handles
  change from run to run).

### Withheld writes (block 5)

**DECISION (Fable):** what a GPU-only page does when shader writes to it
are withheld (`crate::memory::withholding`).

**ANSWER (Fable, block 5):** pages that programs using title memory write
are never GPU-only.
- The gap, as found: when a list using title memory reads memory with no
  copy, what such programs wrote is dropped from the host copies, which
  take title memory's bytes again. That is known only after the list ran.
  A closed page has no bytes from before in title memory (its copy-backs
  were skipped), so the net could only publish what should have been
  dropped and demote (`withheld`). The first stage 3 run on PC 2 with the
  BVH on did that for 225 pages [M] and is invalid; on PC 1, 744 of the
  single-holder pages also carry kept writes in the stage 1 stream [C].
- Not a before-image: saving one for closed pages is the copy-back stage 3
  saves, and before the lists run it is not known which pages they write.
- Instead, by who writes. Only programs using title memory are withheld
  (their own mark bitmap, the flag on their declared ranges, the pages
  they write at computed addresses), and ranges of other programs that
  overlap a declared one of theirs in the same copy. So
  (`watch::net::program_writes`): before a submission's lists run, every
  range it may withhold is told (`host::may_be_withheld`); after it, when
  its writes go back, the pages such programs marked or wrote at computed
  addresses. A told page is never closed again, at any of its addresses
  (`refused_program`); one that is closed gets the GPU's bytes in title
  memory and leaves the class (`program`, a demotion that is no
  classification error). Withheld ranges are remembered the same way.
- What is left: a page such a program writes for the first time in the
  very submission that is withheld, through a view no list declared or at
  a computed address, while the page is closed. Counted as before
  (`demoted_withheld`); a run with any is invalid. Closing that too needs
  the set of pages such a program can reach before it runs, which the
  page table does not give (it names every copy).
- Cost: with the BVH on, the builder's pages leave stage 3. How many of the
  section's single-holder pages remain is measured by the run
  (`closed_*` against `refused_program`), not predicted.
- `VICTORY_MEMMODEL` does not become the default while `demoted_withheld`
  can be reached; whether the remaining case is accepted or closed is
  decided from the full runs' counts.

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

**ANSWER (Fable, block 2):** stage 3: demote on any host write to a
GPU-only page (sync first, then the write goes as today), counted per site
in the demotion log. Writing the copy directly comes only for a site the
log shows in steady traffic. Sites 4/5 fault on the thread holding
`submitted`, site 1 asks the net before its kernel copy (lock order
answer, points 4 and 5); site 10 as site 1.

**ANSWER (Fable, block 3):** built so (c4/c5). All host writes and reads
of a closed page demote with reason `host`; the log does not tell the
sites apart yet (one counter). Added beyond the table: site 15 (the
runtime's writes through its own view) is told before the write with the
model on (`title_memory_will_change`), since its report comes only after
the bytes landed and a sync then would have overwritten them; the pages
count as being written until the report, so none is closed meanwhile.

## 6. Migration stages

- Stage 3, `VICTORY_MEMMODEL=1`, default OFF: stop mirroring and copying
  back the 744 MB section; net mode guards it. Commits: c3 forced copy-back
  (test t3), c4 net fault path with fence (t1, t4), c5 wiring for the
  block. Fable writes c3-c5.
- Stage 4, only after stage 3 is proven: the GPU-only part of the 6.5 GB
  section (needs the read-fault table and a per-unit split, since 390 MB
  are shared), or per-submission bookkeeping as the first piece of flights.
  **DECISION (Fable):** which first.

  **ANSWER (Fable, block 2):** provisional: the next block first (same
  machinery and proof as stage 3), the bookkeeping as a design section here
  in parallel, no code. Fixed only with the stage 3 result: if its gain on
  PC 1 is well under the prediction, the time is in the synchronous wait
  and bookkeeping goes first.

  **ANSWER (Fable, block 3):** changed by the alias result. Candidate
  order for stage 4, fixed with the stage 3 numbers: (1) one copy per
  physical page for the 744 MB section (the alias catch-up is its "up"
  traffic and the reason most of its pages cannot be GPU-only), (2) the
  GPU-only part of the 6.5 GB section, (3) per-submission bookkeeping.
  (1) goes first if `refused_holders` is the large number in the stage 3
  log on both PCs; if instead few pages are refused and the gain is still
  small, the time is in the synchronous wait and (3) goes first.

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

Added with the code (block 3, `block1/pagenet` aad9013):
- Point 4 is met in the unit tests: 24 mutations of the net, all red
  (`pagenet-mutate4.py`).
- Point 6 covers three variables: `VICTORY_MEMMODEL`,
  `VICTORY_MEMMODEL_SECTION` (the section's size in bytes; without it
  nothing is closed) and `VICTORY_MEMMODEL_OUT` (the log file).
- In every stage 3 run the summary line must show `forced_failed=0` and
  `cannot_sync=0`; anything else ends the run's validity (a sync that had
  nothing to copy from, or a closed page met where no sync could run).
- Before the first run: no `gpu_w_kept` record (withheld shader writes) on
  the section in either stream; the section's exact size from both
  streams; the count of single-holder pages (what stage 3 can close).
- Known gap, compared run against run: a system call of the title's that
  reads from a closed page fails instead of faulting.

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

**ANSWER (Fable, block 2):** the existing copy, with refresh and copy-back
switched off for GPU-only pages. A demoted page falls back to today's
behaviour and needs that copy anyway; nothing moves, so a stage 3 result
measures the sync alone. Consequence to write into the prediction: stage 3
saves time, not RAM or VRAM (the CPU side of a copy in GPU memory stays
allocated). Arena pages are a later stage with its own proof.
