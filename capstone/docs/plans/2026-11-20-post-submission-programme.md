# After the submission: build what the paper describes and the silicon lacks, then optimised ports

Status: approved by the project lead on 2026-11-20; in progress. This is the programme for every lane. Each lane's
brief is in section 7. The board queue of the first day is train PREREG Addendum 196 (spec-silicon 087e8aba0e11).

## Context

The paper is submitted (paper `main` = ebc1700a, the state of 18 Nov). The lead's request of 20 Nov: study
everything of these weeks, find what is not implemented, implement it (instructions, caches, new bitstreams,
everything), thoroughly now that the deadline is gone; after that, optimised ports of gcc, perl and every other
benchmark.

The measured prototype P1 (capstone-ariane d83d3a633) is not the design Section 5 specifies. It has the paged
lifetime table and a 16-entry record TLB. It lacks the capability format, the record, the record cache, three
instructions and the trap route that the paper describes. A security defect found on 17 Nov is open on it.
Several performance levers were sized and never built or never run. The compiler configuration the paper measured
is not on `dev`. The ports are older two-operand revisions. All three boards have been idle since 19 Nov 07:01.

Sources: three inventories of 20 Nov (hardware; software and ports; ten session transcripts with their screenshots
and uploads). Rows marked (v) were re-read at the primary source; every other row is re-read before it is acted on.
The shared `dev` checkout is 918 commits behind `origin/dev`; every doc is read from its git ref.

## 1. What is not implemented

### 1.1 Section 5 and 6 of the paper against P1

| Paper | P1 today | Missing |
|---|---|---|
| 63-bit lifetime id: slot 31, generation 32; address 38 bits | 55-bit id (slot 25, generation 30), 39-bit cursor, type and permission fields | the format: about 70 width and compare sites |
| 16-byte record: generation, previous, next, depth; generation advances at release | 137-bit record with parent/valid/free/linear; generation advances at allocation | layout and both semantic changes |
| 2^31 slots, growth at zero free slots, CDERIVE fails without a fault when it cannot grow | 2^25 slots; cause 30 fires with 256 identities left; no NOGROW; unreachable record fails closed without cause 29 | growth and failure rules |
| CREVOKE with one operand, releasing the operand too | two operands (v) | the instruction |
| CRENEW (appendix) | absent; funct7 0x54 decodes as CSNODE (v) | the instruction |
| CSMINT in hardware | illegal at decode, the monitor emulates it (v) | the instruction |
| record cache: 2-way, 8 KiB, 64-byte lines, LRU, used by checks, allocation and release | none on P1. B5a (6e82d7bd2) has a write-through one behind the verdict cache; measured slower (gcc +0.41 %); mcf never measured | a cache that wins |
| trap route: protected U when `srevroot` is set, S accesses checked through references, tagged `sscratch`, causes 24-30 to S | CSRUNV: a context runs at M with U permissions | all of it |
| raw tag moves CLDRAW / CSTRAW and per-page CPTAG | none | all three, and a per-page "may hold tags" flag |
| no types, rights, SEAL/DELIN, CSRUNV, supervisor | all present | the deletions |

No RTL for any of this is written (scope note `2026-11-17-sublet-b5b-scope.md` on `origin/land/rtl-sublet-p2`).
The reference model exists in capstone-qemu `origin/address-space-root`: c1985bb529 has the new format, record,
one-operand CREVOKE and CRENEW with CSRUNV still present (v); 9289114e06 is the final trap-route model (v).
Specification: `capstone/docs/design/sublet-reference-model.md` at 32dde56a6b42.

### 1.2 Open defects on P1

| Id | What | State |
|---|---|---|
| R-76 | outside supervision, S/U `ccsrrw` reads and replaces the monitor's capabilities (security) | 7-line fix in `csr_regfile.sv` (7180a83b6, v); sealed only on top of the record cache; never flashed |
| R-75 | a one-cycle race in the RTL's PC liveness check (which the reference model does not have): the caller's code capability is probed during the switcher's M transit while the context's table is selected (ids 0 and 1 are header slots there, so DEAD); at the namespace change the registered flush clears the LSU's record one edge after the tracker consumed the stale DEAD, and the M handler then takes cause 25 forever. Settled by a one-variable pair (va-rc-trap-p vs -zp, land/rtl-sublet-p2 6f6cf0cb4262). Cannot occur on B5b, which deletes CSRUNV. Earlier readings ("below M", "declared illegal", "any id change while srevroot holds a table") are withdrawn | sim only; fix options for the lead: (a) re-resolve on a namespace change (prototype passes), (c) ids carry a namespace, (d) clear the records in the cycle the table term changes |
| M-16 | protected tshark did not return in 1 of 3 boots | no mechanism; Wireshark has no cost row |
| R-73, R-74, R-44, R-28, S-21, S-22 | debug-module writes around tags; tag-cache fill race; CPMP tracker; audit items | open |
| R-72, R-68, R-60, R-32 | registry says open, the fix commit is in P1 | silicon check and registry entry missing |
| CREVOKE | not interruptible: 32-39 cycles a record, 39 ms on PostgreSQL's largest reset | never planned |
| RTL assertions | confirmed: none had ever run (no --assert on the Verilator rule). With a side Verilator 5.038 and --assert, 101 sites compiled in, 0 fired on P1's 95-test set (ISSUES R-78, 20 Nov) | done for P1; 5.008 cannot build them |
| timing and area | WNS -13.173 ns; 91.9 % LUT, 99.7 % slices (v) | every addition needs room |

### 1.3 Hardware levers sized and not built, or built and not run

| Lever | State |
|---|---|
| verdict-cache miss (15.5 cycles; 7-8 % of nested in-context cycles); 256x4 verdict cache | not built |
| store-buffer forwarding; `pcc_hold` from flops; full UNINIT-STC lever | never built |
| L1D 64 KiB (a7f95f371), I-cache 64-byte lines (105c464bc); L1D 48 KiB 12-way (b9ab75c41) | two sealed and never run; one unbuilt |
| spatial-only core (asked for on 15 and 17 Nov, and by the review) | never synthesised; the paper prints an estimate |
| compressed capability instructions (stage 2) | decoder sketch only; the lead said "try, after the rest" |
| PC-relative capability load; liveness-test instruction | proposals |
| Sstc timer; D-cache prefetch; generation-only cache | ideas, not built |

### 1.4 Software

| Item | State |
|---|---|
| the measured configuration (arms FRCDKMSa3 for Full and FRCDKMSspa3 for +Spatial, compiler build c9) | not buildable from `origin/dev`. In the measured images by content (v, corrected twice on 20 Nov): fold-global-lo, cap-base-rvc, memcpy alignment dispatch, drop-redundant-shrink, direct calls, frame setup (compiler c9, built at 29027126f39b); memset null-capability (in the SDK source, branch alloc-lean 9950c04d384f, as 7166a346d516); node-table pool code in the kernel. The SDK (`perf/alean/sdkm13b`, +Spatial `sdksp13b`) was built by llvm-c4 from alloc-lean, 128 commits that `origin/dev` lacks, the native route included. NOT in the measured images: the C-69 fix, kfast, the heap-excess patch |
| compiler C2 (hoist `cincoffset`) and C3 (capabilities in registers) | not done: `canTrap` marks every `cincoffset` trapping; no rematerialisation, no hints |
| compiler C4 (move round trips), C5 (redundant bounds) | C4 not counted per benchmark; C5 built, default off, not in the link driver |
| open compiler issues | C-32, C-46, C-53, C-56, C-57, C-59, C-60, C-63 (C++ exceptions), C-68, C-70, C-74 |
| native route | refuses sockets, epoll, eventfd, timerfd, threads, fork; so memcached never ran as a server |
| virtual-profile threads | need a kit with matched supervisor slots above 1 (Q-32); on a 1/1 kit the module refuses the second context with EBUSY (capstone_vm.c:3271), which musl's pthread_create returns as EAGAIN. Measured 22 Nov by the compiler lane: with 32 slots on both sides, pthread_create, tls_local_exec, pthread_tsd and pthread_cond pass (lane compiler/virtual-thread-slots) |
| virtual-launcher thread gaps | at 32 slots two libc-tests still fail past pthread_create (compiler lane, dev 891610c248c1; read from source, not traced): PRIO_INHERIT mutexes get ENOSYS, because the virtual futex filters admit only WAIT/WAKE(_BITSET) (runtime/virtual/pthread.c:175, exec.c:1218-1220), while the delegated runtime has PI; pthread_cancel gets ESRCH, because the virtual launcher sets no host->tkill (delegate-service.c:701-709). Also: run-libc-test-delegated.py should skip pthread tests with a reason on a 1-slot kit instead of reporting faults |
| kernel and module | kfast and NO_HZ never run on a board; huge pages refused (mcf's only lever); Plain's ticks are corrected by an estimate |
| record table | default 262,144 records too small for Perl and gcc; a subtree above 65,532 records cannot be built, cause unknown (resolved 24 Nov: not a limit of the native route, see 7.5 item 4) |
| QEMU twin | models neither P1's paged table nor code placement; Zcb off; runtime cases 29-31 fail |
| tooling | `spec-classify.py` misreads native transcripts (fix only in `kitfix-next`); kits cap firmware at 26 MB; the board watchdog reminds and queues nothing |
| measurements | SPEC train and ref never run on the final configuration; rows have 1-2 runs, no intervals; SQLite and PHP bug corpora never rerun on the board |
| records | x264 second runs and the 250 Hz run are unrecorded; ISSUES header and `current-state.md` are stale |

### 1.5 Ports

| Program | State |
|---|---|
| perlbench | only scalar heads protected (+5.3 %); bodies, hash entries, op slabs not; patch 0002 applied by no build; +788.8 M in-context cycles of Perl 5.36 on suns.pl unattributed (cycles, not perlbench instructions: corrected 20 Nov); on SPEC's test.pl the protected image costs +15,591 M cycles (+5.84 %); timer share 11 % unexplained |
| gcc | pools +5.1 %; all three allocators +15.4 % with a 2^22-record table, one boot. SPEC's `ggc_collect` returns at once (v), so "release at collection" cannot help; typed free lists, objects of 4 KiB and more, growing obstack objects unported; the collaborator's port needs one-operand CREVOKE and CRENEW |
| mcf, lbm, x264, imagick, nab, xz | no allocator of their own (wrappers over `malloc`, or compiled out); the work here is code generation and memory: mcf's 112-byte arc and page-table walks, lbm's seven spills an iteration, imagick's filter loop |
| applications | every P1 row but SQLite uses a two-operand patch, not Fig. 10's; memcached in-process; PHP one binary in two modes; SQLite `--size 100` deferred; most costs unattributed |

## 2. Order of work

Principle: each piece is built once, in the stack where it survives. Anything that touches the record or the walker
(record cache, interruptible CREVOKE, growth rules, the three instructions) is built on the new format, not on P1's.
Anything independent of the format is built on P1 now and rebased.

### Step 0, the first hours (no new hardware)

1. Record what is measured and unrecorded: x264 second runs (30,671 s and 30,673 s, both in band, outputs equal),
   Plain x264 at 250 Hz (timer share 7.93 %), as PREREG Addendum 195, RESULTS and a memory note. Paper unchanged.
2. Commit this programme as `capstone/docs/plans/2026-11-20-post-submission-programme.md` with one brief per lane.
3. Put the boards to work (section 4) and start two guarded syntheses (bitstream 1 and the spatial-only core).
4. Repair tooling that would void the new runs: classifier fix into the kits, firmware cap, watchdog that queues.

### Hardware line A: format-independent, on P1

| Bitstream (by function) | Content | Written prediction before synthesis |
|---|---|---|
| 1. P1 + `ccsrrw` privilege fix | cherry-pick of 7180a83b6's `csr_regfile.sv` change onto d83d3a633 | `u-ccsrrw` refused on silicon; gcc, mcf, perl within 0.2 % of P1; top LUT within 2,000 |
| 2. spatial-only core | P1 with the lifetime table, walker and verdict cache removed by a parameter | area inside the paper's estimate range; gives the measured row and a real +Spatial hardware arm |
| 3. performance batch | larger verdict cache and cheaper miss path; store-buffer forwarding; the L1D and I-cache variant that wins on silicon; R-73, R-74, R-75 fixes; LRU in the record TLB; assertions on in simulation | per lever, from a cycle model on gcc, mcf and perl traces; a lever whose model shows under 0.2 % is dropped before synthesis |

Each lever gets a synthesis-only gate with per-module LUT before the batch build. The batch is one bitstream.

### Hardware line B: Section 5 as written

| Bitstream | Content | Reference |
|---|---|---|
| 4. Sublet format and instructions (B5b-1) | in three gated steps: (a) the codec alone; (b) 128-bit record, walker, release-time generations, growth and failure rules, interruptible CREVOKE; (c) one-operand CREVOKE, hardware CSMINT, CRENEW. Then the record cache of 5.2 on the 16-byte record (4 records a line, write-back, shared by checks, allocation and release). CSRUNV kept. Monitor ported in step (a) | QEMU c1985bb529, tests of section 8 of the reference model |
| 5. trap route (B5b-2) | `srevroot` as the protected-U switch, S accesses checked through references, tagged `sscratch`, causes 24-30 to S; CLDRAW, CSTRAW, CPTAG; then the deletions (types, rights, SEAL/DELIN, CSRUNV, supervisor), which return area | QEMU 9289114e06; the collaborator's kernel (31 patches) and firmware |
| 6. instruction-set stage 2 | compressed capability instructions (the sketch's recommended set), PC-relative capability load, liveness test; each first in QEMU and the compiler, measured, then RTL | `2026-11-14-capability-rvc-stage2-decoder-sketch.md` |

The record cache is decided by a model first: record-access traces from QEMU for gcc, mcf and Perl, replayed
through the 5.2 cache, a generation-only cache and no cache. B5a's mcf reading and its events 34-37 calibrate the
model. The variant goes to RTL only if the model shows a gain on at least one of the three and no loss on any.

### Software line (parallel to both hardware lines)

1. Land the measured configuration on `dev`: the compiler changes that are in the measured images, each as one
   squashed change after lit and the QEMU suites, so that `dev` builds the paper's images. The C-69 fix, kfast
   and the heap-excess patch are separate changes with their own measurements.
2. Compiler: C2 (non-trapping `cincoffset` where bounds are proven, then hoisting), C3 (rematerialisation and
   allocation hints), C4 (count, then fix), C5 into the link driver, bounded-global hoisting, the open issues with
   C-63 and C-70 first.
3. Kernel, module, runtime: board readings for kfast and NO_HZ; huge pages in the module; record table grown on
   demand instead of a fixed ceiling; the 65,532 cap root-caused; node-table pool default on.
4. Native route: sockets, epoll, threads. The collaborator's branch has threads and the trap route; this is
   coordinated with the collaborator before any change to `capstone_vm`.
5. QEMU twin: P1's paged table, Zcb, runtime cases 29-31.

### Ports and measurements (after bitstream 4; QEMU work starts now)

1. gcc: all three allocators on the one-operand instructions with CRENEW for pools (the collaborator's design),
   typed free lists, large and growing objects; cost split per allocator; table grown on demand.
2. perlbench: bodies, hash entries and op slabs; patch 0002 landed; CRENEW and liveness-test variant; the
   unattributed instructions attributed.
3. mcf, lbm, x264, imagick, nab, xz: per-benchmark code generation and memory work from the recorded hot spots.
4. Applications: Fig. 10's one-operand revisions on silicon; memcached as a server; PHP with a pristine second
   binary; Wireshark's non-return root-caused; SQLite `--size 100`; costs attributed.
5. Measurements for the camera-ready: SPEC train for all eight (ref where memory allows), three runs per arm with
   intervals; SQLite and PHP bug corpora on the board; the spatial-only row.

## 3. Lanes

| Lane | Owns |
|---|---|
| rtl | lines A and B RTL, directed tests, lint, the cycle and cache models |
| board | synthesis under the guard (two jobs), seals, flashes, Tier 1 boots, the queue |
| compiler | landing the configuration, C2-C5, compiler issues, stage 2 in the compiler |
| nested | ports of gcc, perlbench and the applications; record-access traces |
| runtime (this session) | coordination, kits and tooling, SPEC and train runs, kernel and module levers, records |

Each lane gets a self-contained brief in `docs/plans/`. Their reports are claims and are checked at source.

## 4. Board queue from the first hour

| Board | Now | Then |
|---|---|---|
| 1 (P1) | third runs of the Full rows, then of the +Spatial rows (Addendum 196), until the train stages exist; then SPEC train, Full build, problem rows first: gcc, perlbench, mcf, imagick | +Spatial train; kfast and NO_HZ pairs |
| 2 | flash sealed record-cache + `ccsrrw` fix bitstream (7180a83b6): Tier 1 boot, `u-ccsrrw` on silicon, mcf pair with events 34-37 | L1D 64 KiB and I-cache 64-byte bitstreams against their bases; bitstream 1 when sealed |
| 3 (Plain) | third runs of the Plain rows and a second Plain x264 run (Addendum 196); then SPEC train, Plain, same order | SQLite and PHP bug corpora need the Full bitstream: they go to board 1 or 2 |

Train inputs that do not fit the board's memory are found on the QEMU twin first and listed, not shrunk.

## 5. Gates and verification

- RTL: directed tests that create the condition; the 95-test sweep with the RVFI trace compare; `rtl-lint-gate.sh`;
  `claim-auditor` for synthesizability; a synthesis-only gate (40 GB guard, hierarchical utilisation) for each wide
  addition; the full build under `synth-guard.sh` through the synthesis store's `launch-job.sh`; route failure
  pre-registered as a branch. LUT bands never tighter than 2,500 a module, 2,000 at top.
- Line B: every step is compared against the QEMU reference revision on the reference model's own tests, and the
  monitor boots on it in simulation before any synthesis.
- Bitstreams: sealed, sha256 checked before the flash, name read back, fallback registered, own Tier 1 boot.
- Board runs: pre-registered and pushed before launch, stubbed dry run kept, control first, results cited by image
  hash; a fix for wrong values is checked against an independent reference.
- Compiler: lit and the QEMU suites (serialised) before anything lands; byte-identical output with a flag off.
- Every commit and push: `precommit-scan.sh` by absolute path; shared branches get squashed commits only.
- Results go to PREREG, RESULTS and `docs/ref/fpga-silicon-measurements-for-paper.md`. The paper changes only on
  the lead's word, through eval/* PRs; the abstract never.

## 6. Left with the lead (none of it blocks the work)

- Anonymous repository: needs the lead's repository or a token that can create one. One session quoted a rule that
  artifact repositories must not change after the deadline; that contradicts what was said on 18 Nov.
- The collaborator: kernel and firmware for the trap route, and the gcc port, are theirs; contact goes through the lead.
- The paper says "with both a QEMU and an FPGA implementation available"; the FPGA one matches Section 5 only after
  bitstream 5. Any wording change is the lead's.

**Rulings of the lead, 22 Nov** (asked by the runtime lane, answered in its session):
- R-75: option (d). The LSU's ALLOW/DEAD records give no PC verdict while a namespace change is pending (prototype
  capstone-ariane r75-earlyclr b7ea390a1). No ruling change. It becomes a candidate on bitstream 1's tree, through lint,
  the sweep with a firing count, a synthesis-only gate and then a full build.
- Line B (B5b): option (C), staged.
  - B5b-1 is the Sublet format, the record and the instructions, with CSRUNV kept. The format has no types, so
    keeping CSRUNV means a monitor redesign.
  - B5b-2 is the trap route, the raw tag moves and the deletions.
- Settled by the lanes the same day: the bitstream-1 default with P1 as the registered fallback (the lead, relayed by
  the board lane; board choices are delegated to the lanes); and the cone programme on P1 stops after cut 1.
  - The clock stays at 40 ns, so the slack cut 2 would buy is unused.
  - Its mcf cost bound is +0.25 %.

## 7. Lane briefs

Rules for every lane. A report from another lane or from a subagent is a claim: check it at its primary source before
acting on it. Every board run is pre-registered and pushed before it starts. Shared branches get one squashed commit
per logical change, after `precommit-scan.sh`. No personal names anywhere. QEMU suites are serialised. Board time is
requested from the runtime lane, which keeps the queue. Reports are short: what was done, the reading, the evidence
path, what is next. A retraction is reported at once.

Reference points used below:
- P1: capstone-ariane d83d3a633. Full build: 187,368 LUT placed (91.94 %), 76,543 FF, 50,800 slices (99.71 %),
  WNS -13.173 ns, routed clean.
- Record cache B5a: 6e82d7bd2. Its `ccsrrw` fix: 7180a83b6 (branch `r76-ccsrrw-priv`). That full build is sealed
  (job `b5ar76-7180a83b6-full-out` in the synthesis store): routed with 0 errors, 189,920 LUT (93.19 %), 77,368 FF,
  50,858 slices (99.82 %), WNS -13.615 ns. It has not been flashed.
- Reference model: capstone-qemu `origin/address-space-root`, c1985bb529 (new format, 16-byte record, one-operand
  CREVOKE, CRENEW, CSRUNV kept) and 9289114e06 (trap route). Specification:
  `capstone/docs/design/sublet-reference-model.md` at 32dde56a6b42.
- Scope note for line B: `origin/land/rtl-sublet-p2:capstone/docs/plans/2026-11-17-sublet-b5b-scope.md`.

### 7.1 rtl lane

In this order; items 1 to 3 are independent of each other and can overlap.

1. **P1 with the `ccsrrw` privilege fix (bitstream 1).** A branch from d83d3a633 that carries only the
   `csr_regfile.sv` term of 7180a83b6 and its three tests (`u-ccsrrw`, `u-ccsrrw-scr`, `u-ccsrrw-m`). Gates: the
   three arms read as recorded for the fix (0x2, 0x0, 0x0, A; and 0xb, 0x4, A, B for the M control); the lint gate;
   the 95-test sweep with the RVFI trace compare against P1. Add the arm the registry lists as not exercised: a
   supervised domain below M issuing `ccsrrw` on CSCRATCH and CEPC, which must stay legal. The term sits in a cone
   on the standing UNOPTFLAT list; the sealed full build of the same term on B5a is the synthesis evidence. Hand the
   hash with its lint numbers to the board lane for a full build. Written prediction: top LUT within 2,000 of P1;
   routed; `u-ccsrrw` refused on silicon; gcc, mcf and perlbench within 0.2 % of P1.
2. **Spatial-only core (bitstream 2).** P1 with the lifetime machinery removed by one parameter: the lifetime table
   and its walker, the record TLB, the verdict cache, and the lifetime check on loads and stores. Bounds checks and
   tags stay. The lifetime instructions become illegal. Before synthesis, write the predicted area (per module, from
   P1's hierarchical utilisation) and the list of tests that must still pass. The +Spatial images are the workload:
   check on the QEMU twin or in the sweep that they execute no lifetime instruction. This build gives the paper's
   estimated row a measured value.
3. **Line B, step (a): the codec alone.** The 63-bit id format (slot 31, generation 32; 38-bit address) as compress
   and decompress modules with their own directed tests against the reference model's encoder, and a synthesis-only
   gate with per-module LUT before anything else is connected. A compress-then-decompress round trip once took
   technology mapping from 5 GB to 94 GB; that is why the codec goes first and alone. The monitor port to the new
   format starts in the same step.
4. **Models before RTL.** (i) The record-cache model: replay record-access traces (the nested lane provides them
   for gcc, mcf and Perl) through the cache of Section 5.2, a generation-only cache and no cache. A variant goes to
   RTL only with a gain on at least one of the three programs and no loss on any. (ii) A cycle model per
   performance lever (larger verdict cache and cheaper miss path, store-buffer forwarding, LRU in the record TLB).
   A lever whose model shows under 0.2 % on gcc, mcf and perlbench is dropped before synthesis.
5. **Defects.** R-75 root cause and fix; R-73 and R-74; confirm whether any RTL assertion has ever run in
   simulation and switch them on; silicon check lists for R-72, R-68, R-60 and R-32 so the registry can be closed.
6. Then line B steps (b) and (c), the record cache on the 16-byte record, the trap route, the deletions, and
   stage 2 of the instruction set, as in section 2.

### 7.2 board lane

1. Keep two guarded synthesis jobs running whenever two hashes are ready (`launch-job.sh`, `mem-headroom.sh`, the
   16 GB floor). Every job runs under `synth-guard.sh` with its ceiling, and the watchdog log is checked after
   launch and at the end. Read counts from the `.vds`, never from `runme.log`.
2. Grade the sealed build of 7180a83b6 with `grade_full.py` against P1's seal and fill the empty "full grade"
   line of its `SEAL.txt`.
3. First jobs: the full build of bitstream 1 when the rtl lane hands over its hash, and the synthesis-only gate of
   the codec. Then the spatial-only core. A build near 178,000 to 190,000 LUT may fail to route: pre-register that
   branch for each build.
4. The two cache-sizing builds (L1D 64 KiB a7f95f371, I-cache with 64-byte lines 105c464bc) are on an older base
   (merge-base 5f41c486e) and the second has no bitstream file in the store. Rebuild both on P1 when the queue has
   room, so each differs from P1 in one thing.
5. Name every bitstream by its combined ideas. Seal with sha256 and hand the seal to the runtime lane, which
   flashes and runs.

### 7.3 compiler lane

1. **Put the measured configuration on `dev`.** Corrected on 20 Nov after the compiler lane's reading, re-checked
   at the measured arm's `sdk.json`, the compiler's build file and the images. (Corrected again the same day: the first
   re-check read arm FRCDKMS, whose SDK is `perf/memset` at beb1bbabe26a. The paper's Full rows come from arm FRCDKMSa3,
   whose SDK is `perf/alean/sdkm13b`, from branch alloc-lean 9950c04d384f. The conclusions below hold on FRCDKMSa3,
   re-checked there.) In the paper's images: 2f3ff76eba7b,
   74080cce29e5, 29027126f39b, 331d73414667, 5069028a65b0 (ancestors of the compiler source 29027126f39b), the
   frame setup e241065990d9 (there as f9d76538e477), the memset change b68bbde23c79 (in the SDK source,
   alloc-lean 9950c04d384f, as 7166a346d516) and the node-table pool 7719bccb8ca4 (its parameter is in the measured kernel
   image). NOT in the paper's images: the C-69 fix c77dd6e705ad (the measured compiler's build file has no
   `CapstoneMisalignedCapAccess.cpp`), kfast 3f7c8e19fecc (no `kfast` string in the measured kernel image) and the
   heap-excess patch 76455aff7e7a (`__malloc_calloc_alloc` occurs 0 times in FRCDKMSa3's libc and perlbench
   image, 3 times in the control library). An earlier version of this section listed all eleven as measured; that
   was wrong for those three. Land the measured ones each as one squashed change after lit and the QEMU suites.
   Acceptance: a compiler built from `dev` produces the paper's SPEC images byte for byte, or the differences are
   listed and explained. The three others land as changes of their own. The measured runtime is alloc-lean 9950c04d384f
   (128 commits that `dev` lacks, the native route among them, plus alloc-lean's own mallocng series); who lands it is the lead's
   decision. A compiler from `dev` with the measured runtime reproduces 10 of 11 FRCDKMSa3 images byte for byte, and gcc
   differs only in clang's embedded version string (the compiler lane's reading of 20 Nov, not re-run here).
2. **C2:** `cincoffset` is marked trapping everywhere (`CapstoneInstrInfo.cpp`, `canTrap`), so it is never hoisted.
   Make it non-trapping where the bounds are proven, then hoist. **C3:** rematerialisation and allocation hints, so
   capabilities stay in registers. **C4:** count the move round trips per benchmark, then fix. **C5:** the
   redundant-bounds flags into the link driver, default on once lit and the suites pass.
3. Open issues, C-63 (C++ exceptions) and C-70 first, then C-32, C-46, C-53, C-56, C-57, C-59, C-60, C-68, C-74.
4. Per benchmark, from the recorded hot spots: mcf's 112-byte arc and its page-table walks, lbm's seven spills an
   iteration, imagick's filter loop. Report each change as a share of that row's gap to Plain.
5. Stage 2 of the instruction set (compressed capability instructions, PC-relative capability load, liveness
   test) first in QEMU and the compiler, with an instruction and size census per benchmark, before any RTL.
6. Every measurement prints the compiler identity beside the numbers. Flags off must give byte-identical output.

### 7.4 nested lane

1. **Record-access traces** from QEMU for gcc (train01), mcf and Perl, for the rtl lane's record-cache model: one
   line per record read or write with the slot and the kind (check, allocation, release).
2. **gcc, all three allocators** on the one-operand instructions. Corrected on 20 Nov: no revision of this
   repository pins QEMU c1985bb529 and no kit exists for it; it is the semantic reference for the RTL only. The
   working base is the environment of 71ce199d8fea (QEMU 9e69171f1c, native route), where the collaborator's port
   was verified on the test input. One-operand CREVOKE, CRENEW for the pools,
   typed free lists, objects of 4 KiB and more, growing obstack objects. SPEC's `ggc_collect` returns at once, so
   release at collection is not available. Split the cost per allocator. The collaborator's port is the design
   reference; contact with the collaborator goes through the lead.
3. **perlbench:** bodies, hash entries and op slabs; land patch 0002; a variant with CRENEW and the liveness test;
   attribute the protected image's extra cycles on test.pl (+15,591 M) and Perl 5.36's on suns.pl (+788.8 M).
4. **Applications:** the one-operand revisions of Fig. 10 in the same environment, ready for silicon when
   bitstream 4 exists; PHP with a pristine second binary; SQLite speedtest1 at `--size 100`; the Wireshark
   non-return root-caused (it returned in 2 of 3 boots on P1; no mechanism is recorded).
5. SQLite and PHP bug corpora on the board: prepare the images and the pre-registration; the runtime lane queues
   them on a board with the Full bitstream.

### 7.5 runtime lane (the coordinating session)

1. Boards: the queue of section 4, pre-registrations, kits, flashes, Tier 1 boots, records.
2. Tooling: the classifier fix into the kits (native transcripts read as NO RESULT today, so readings come from
   `read/result.json`); the firmware size cap (the kits stop at 26 MB because the console refuses uploads above
   32 MB); a board watchdog that starts the next queued run.
   State, 24 Nov:
   - The classifier fix (e2ca116a965b) is in bkit112. Older native kits still print NO RESULT, so their rows are
     read with read-one.sh.
   - The firmware cap is avoided rather than raised. Kits bkit110 (native), bkit112 (native) and bkit113 (Plain)
     move a stage's input to an initrd that is loaded separately at 0x98000000. The probe passed on silicon
     (train Addendum 231): xz test 12/12, in band.
   - The watchdog is a watcher, not a starter. Chains are queued behind one another by pid, and
     `capstone/ports/spec-cpu2017/board/board-idle-watch.py` reports any board that holds no booking for 45
     minutes. A start is not automatic: each one needs its own pre-registration, and a refused start must be read
     first.
3. SPEC train stages for the seven benchmarks that have none, with a memory check on the QEMU twin and the board
   hours per arm; then three runs per arm with intervals.
4. Kernel and module levers: board readings for kfast and NO_HZ; huge pages in the module; a record table that
   grows on demand; the cause of the 65,532-record subtree limit.
   State, 24 Nov: the 65,532 limit is understood, and it does not bind the native route.
   - It belongs to the core's physical record pool. That pool is used when no table is set at `srevroot`, as in
     bare-metal tests.
   - Its 16-bit head starts at 3, and 65,535 means full. 65,535 - 3 = 65,532. At d83d3a633,
     `capstone_rev_node.anvil:397,840,849` hold the head, and `csr_regfile.sv:1088` reads it as `csnodefree`.
   - The RTL simulation that read 65,532 (`crevoke-cap.S`) is such a bare-metal test.
   - The native route keeps its records in the table at `srevroot`. On silicon F, through that route, a
     100,000-node subtree builds and revokes. The 10^6 probe dies between 196,609 and 262,145 nodes, at the table's
     262,144 records (nested-workloads doc, 54b5fb9b7578).
   - So the limit to remove is the table's fixed size: the record table that grows on demand, above.
5. The silicon check of the `ccsrrw` defect, run last in its boot.
