# The two remaining SPEC test programs, perlbench and x264, in the final configuration: pre-registration

Written and pushed before the board calls (6 Nov, 01:00). The lead's goal for today is every SPEC test under 1.20×
Plain. These two have no measurement in the final configuration yet.

## The runs (kit bkit26-gt: C.5a, module 65ede944a87ea742, 18-bit, counter firmware, all2 params, exact-image launcher)

**Board 1: perlbench test, Temporal (devmk images).**
- Stage `perf/rest/stages/500.perlbench_r-test-capstone-lifetimes-devmk`, built by make-stage.py from
  tree-boarddevmk with the test options of make-stages-hfl.sh: strip, native harness, keep-lib.
- `after-gate.sh` starts the driver only once the QEMU gate (`qgate.sh`, kit-etq3h18) has completed every command,
  rc 0. A specdiff failure there is recorded, not gating. Before the heapfix, perl's reg_email failed specdiff
  on silicon and on the twin.
- It then waits for ns-run.sh 1 (pid 1171962) to end with DONE.

**Board 2: x264 test, Plain.** `spec/stages-test-plain/525.x264_r-test-plain-rv64`, the 3 Nov Plain stage. It runs
after perf/mx's Plain mcf (driver pid 1370451) ends with DONE.

**Still owed:**
- x264 Temporal goes on the next free board.
- perlbench Plain is the 3 Nov board reading (out-hfl: 501.52 + 22,809.06 s, specdiff 2/2).

**Preempted for these runs.** Two drivers were killed while still waiting, before any board call. Their changes go
into the next gcc and mcf runs:
- perf/mx board 1, mcf with resume_quantum;
- perf/gx board 2, gcc train with the quiet launcher.

## Readings and predictions

| # | Prediction | Point estimate |
|---|---|---|
| R0 | each arm: rc 0, control passes; QNT0 007A1200 | |
| R1 | perlbench Temporal run below the 30 Oct baseline (939.67 + 43,358.42 = 44,298 s) | 28,000-36,000 s: 1.2-1.55× Plain's 23,311 s. Run 1 had 1.37 M exits on 30 Oct, so the outside levers carry it |
| R2 | x264 Plain run within 30,000-48,000 s | Capstone's 30 Oct run held 43,160 s of in-context cycles for 543.7 G instructions |

**Reported:**
- run time against Plain and against the 30 Oct baseline (x264 Temporal 51,855.95 s);
- inside and outside time, and exits by kind.

x264's board stage has no imagevalidate step (by design, since 30 Oct), so specdiff reads 0/3. Its outputs are compared with the twin's.

**Dry run** (perf/rest/dryrun, board calls stubbed): A-gateok, with a gate log reading rc 0, starts the driver; the
chain stub checks every list path. `board-env-check.py` passes for both boards' lists.

**Limits.** N = 1. Plain and Temporal of x264 run on different boards and days.

## Addendum 1 (6 Nov 01:35, before either board call)

**The images change to the arena-cache runtime, and the kit to bkit27-gtq.**
- Both programs are relinked with the arena-cache SDK (results/2026-11-06-arena): perlbench bd186c2bd219b194 (stripped
  as make-stage does), x264 00565bd74f342fc4.
- A relink of the same recompiled objects with the production SDK reproduces the tree's exes byte for byte: perl
  1c3a6f43, x264 1c57ad01.
- bkit27-gtq is bkit26-gt with the quiet launcher.
- **Reason:** gcc's exits were allocator map/unmap, and perlbench's 1.37 M exits on 30 Oct are likely the same.
  Measuring the old images first would cost a second 10-hour run.

**The QEMU gates are now `q-perl-ac` and `q-x264-ac`** (kit-profq). The old-image gates keep running as twins.

**The order changes:**
- Board 1, after perf/mx's mcf (driver pid 1521568): perlbench-ac, then x264-ac Temporal, as one list.
- Board 2, after perf/ga's gcc runs: x264 Plain.

**R1 is unchanged.** Its point estimate is now open: the cache's effect on perl has not been measured, even on QEMU.

## Addendum 2 (6 Nov 15:15, before either board call): both programs in gcc train's best configuration

The lead asked for the two remaining tests to run in the best configuration. That is now gcc train's queued one
(results/2026-11-06-arena Addenda 3-7):
- **compiler:** llvm-c3 (llvm-c2 + llvm-capstone perf/shrinkto 1836a2142f7d), `-capstone-shrinkto-globals
  -capstone-shrinkto-end-adjust=1` (C.5a only, R-32);
- **runtime:** SDK ac12 (arena cache, slot cache 65,536, simple 16-bit hash, null-capability memset);
- **module parameters, per stage:** step budget 10 s, resume_quantum 1, retire_exit_sweep 2, arena_pin 1 (ga2's
  block, each read back and restored).

Workload sources are unchanged. The objects are recompiled from each program's production compile list with only
the compiler swapped (perf/best/build-best-c3s1.sh).

| program | image (stripped) | QEMU instructions (kit-profq) | outputs |
|---|---|---|---|
| perlbench | 95f4bbb17e74c543 | run1 96.13 G, against 109.51 G for -ac (−12.2 %); c2ac12 alone 100.45 G | test.out, makerand.out = -ac's byte for byte; specdiff 1/2 as every arm (japh/abigail.t) |
| x264 | d003385b71056a8a | gate running at the time of writing | |

**Order:**
- **Board 1:** perlbench-ac continues. x264-ac is replaced in `perf/rest/b1.list` by
  `perf/best/stages/525.x264_r-test-capstone-lifetimes-devmk-c3s1`. The edit is in place: same inode, line 1
  unchanged, so the running chain reads it. If x264's QEMU gate fails before the boot, the line goes back to -ac, and
  that is recorded here.
- **Board 2** (`perf/best/b2-seq.sh`, one task): gcc train (ga2) → the rtl lane's R-32 SHRINKTO readback bare boot
  (land/rtl-r32-kit 624924575c53, graded by its own PREREG) → the THP mcf runs (results/2026-11-06-thp, unchanged) →
  perlbench in the best configuration (`perf/best/b2.list`). The waiting THP driver (pid 2024109) is killed before
  any board call and runs again inside the sequence, unchanged.

**Predictions:**
- **B1 (perlbench best):** run ≤ 0.92 × perlbench-ac's board run. Point −10…−15 %: inside −12 % of instructions, plus
  arena_pin and the 10 s step budget outside. perl's 318 contexts make the exit-sweep and launch costs uncertain.
- **B2 (x264 best):** run ≤ x264's 30 Oct Temporal 51,855.95 s. A point estimate follows with the QEMU gate.
- Reported: both against Plain (perl 23,311 s; x264 from out-rest-b2).

**Dry run** (perf/best/dryrun/A-ok, every board call stubbed): the sequencer waits for the pid, runs R32, THP and
BEST, and every list path passes the chain stub. board-env-check passes for the perl stage.

## Addendum 3 (6 Nov 15:24): perlbench-ac read on silicon; the images move to the fixed heap base (c3f)

**perlbench-ac on board 1** (out-rest-b1, image bd186c2bd219b194):
- run0 760.41 s, run1 34,126.44 s, 34,886.9 s in all: **1.50×** Plain's 23,311 s;
- inside 25,364 s of cycles over 318 contexts, 108.6 G instructions;
- outside about 9,500 s over 699,817 entries. CV_STEP 2,878 s, of which resolve 855; between steps 5,014 s; other
  ioctls 571 s; teardown 377 s;
- exits by kind: 361,658 calls, 317,411 faults, 20,745 quantum;
- **specdiff 1/2, outputs differ from the twin:** `t/re/reg_email ... FAILED--no leader found`. R1 is therefore
  VOID: a failed sub-test is not a valid run.

**Cause, reproduced on the twin** (perf/regemail/sweep2, kit-profq, re/reg_email.t alone):
- the -ac image ends with `domain fault cause=29` at realloc+0x5b4 after 0.741e9 instructions, under 7 of 8
  environments (board-path mimic, PAD 0..2048; only PAD=1024 passes);
- this is the October defect (perl-regemail/FINDINGS.md). Its fixes, dev 348d9cf2c29c and de7cdacda737, are not
  ancestors of this branch's base 86f638759ecf.
- The full-suite QEMU gate had passed only because of its own allocation history.

**The fix.** Both commits are cherry-picked (perf/arena-cache e53540bb84de, 0ee4200bfc0d) and SDK ac15 is built
(runtime 8bbbb79615f95b91). The c3s1 objects are relinked as **c3f**: perl e42f4649eccf10c0, x264 a52c991174079595,
gcc 5642377943d724c6. On the twin, perl c3f passes re/reg_email alone, 13 of 13, in all 3 environments run
(board mimic, PAD 0, PAD 512); AC fails 2 of those 3.

**What changes:**
- perl and x264 go to the board as c3f, not c3s1. The c3s1 images lack the fix and are withdrawn.
- perl's stage in perf/best/b2.list becomes the c3f stage. Its full-suite QEMU gate (q-perl-c3f) is running and
  must pass before the board reads the list.
- x264 c3f replaces the running x264-ac on board 1 once its QEMU gate passes. x264-ac is preempted then; it is not
  the best configuration, and it lacks the fix.
- B1 now reads against Plain directly: perl c3f run ≤ 0.85 × 34,886.9 s.

## Addendum 4 (6 Nov 16:03, before perlbench c3f's boot): the gate for c3f, and the production control

**perlbench c3f, full-suite QEMU gate** (q-perl-c3f): 2/2 commands rc 0, **specdiff 2/2 PASS**. It is the first
Capstone perlbench arm to pass both. Every earlier arm (production, -ac, c2ac12, c3s1) read 1/2. run1 is 96.74 G
instructions (-ac 109.51 G).

**re/reg_email alone, 8 environments per image** (board-path mimic and PAD 0, 8, 64, 256, 512, 1024, 2048;
perf/regemail sweep2-4):

| image | fails | at |
|---|---|---|
| HF: the production devmk image (sdks-dev2mk), no arena cache | 7 of 8 | cause 29, 0.7405e9 |
| -ac (the board's) | 7 of 8 | cause 29, 0.7408e9 |
| c3s1 | 1 of 8 (PAD 256) | 0.7279e9 |
| **c3f** | **0 of 8** | all `1..13`, 5.6997e9 |

**Correction to Addendum 3:**
- The defect is not new on this branch. The production SDK used by every earlier Capstone perlbench and gcc
  measurement lacks the two fixes as well, and fails the same way.
- Earlier perlbench results whose re/reg_email passed passed by allocation history.

## Addendum 5 (6 Nov 16:10, before the boots): order, after the lead's "focus on gcc train"

- **Board 1:** x264-ac is stopped now (its run-stage process group, checked by cmdline). Next comes gcc train
  (arena Addendum 9), then x264 c3f, which is appended as line 4 only after its QEMU gate passes, by the rule of
  x264-preempt.sh. The x264-preempt watcher is stopped before it acts.
- **Board 2:** x264 Plain continues. `b2-seq.sh` is relaunched to wait on x264 Plain's driver (pid 1655525,
  perf/rest/gt-b2.log), then runs R-32, THP mcf and perlbench c3f, as in Addendum 2. The ga2 step is gone from
  board 2.

## Addendum (6 Nov 16:11, before any board call): the board move is reverted

The lead's correction: x264 Plain has at most about 2 h left on board 2, so gcc train waits for it there.
Arena Addendum 9 and rest Addendum 5 are withdrawn before any board call.
- **Board 2:** the ga2 driver is relaunched unchanged (perf/ga2/gt-run.sh 2) behind x264 Plain's driver, pid
  1655525. b2-seq.sh waits on it again, then R-32, THP mcf and perlbench c3f.
- **Board 1:** as rest Addendum 3. Line 3 of b1.list is x264 c3f again, and x264-preempt.sh is relaunched to stop
  x264-ac once x264 c3f's QEMU gate passes.

## Addendum 6 (6 Nov 17:14, before perlbench's boot): perlbench goes as c3g, one exit fewer per delegated call

- **What c3g is.** c3f's objects relinked with SDK ac16, which is ac15 plus `CAPSTONE_DELEGATE_SELF_CACHE`
  (perf/arena-cache 46d67471e727). Image 32397dcf75222a64.
- **What it changes.** The delegate wire lock asked the module for the context's identity (CV_SERVICE_THREAD_SELF,
  one exit) on every delegated call. With the switch, an image that has started no second context asks once.
  capstone_virtual_thread_create sets a flag before its THREAD_CREATE call, and from then on every acquisition asks.
  - dev solved the same exit differently on 4 Nov (a476237c079b: the identity rides in gp's cursor). The two
    variants are equivalent for a single-context program.
  - This branch's base predates that.
- **Twin trace** (perf/perlx trace1 against trace2): THREAD_SELF exits per process 51/186/119 → 1/1/1, and all
  boundary exits 165/682/676 → 115/497/558 (perl -e 1, op/array.t, re/reg_email.t).
- **Full QEMU gate** (q-perl-c3g): specdiff 2/2 PASS; run1 96.74 G instructions; entries 353,780 against c3f's
  458,733 (−23 %).
- **Not qualified.** Threads were not tested: the pthread contract cannot run on kit-profq, whose one-slot module
  answers pthread_create EBUSY, and its control (ac15) fails the same way. perlbench starts no second context.
- **perf/best/b2.list** now names the c3g stage. **B1** is unchanged.

## RETRACTION and corrections (6 Nov 17:40), from a claim-auditor pass over today's claims

**RETRACTED: the SHRINKTO edge-trap argument.** Arena Addendum 7 and the dev landing message 0001a099efa3 say the
virtual image window is rounded up to a power of two (runtime/virtual/exec.c `rounded()`), so a global ending at the
end of gp's region cannot occur. That is false for the kits used here:
- kit-profq's and the board kits' launcher is built `-DCAPSTONE_VEXEC_EXACT_IMAGE` and grants the exact image. The
  gtr-c3s1 grant reads `bytes=16302080`, and the board control reads `bytes=405504`.
- Only dev's launcher rounds (kit-devq: `bytes=16777216`). `rounded()` in the old exec.c belongs to the nested spawner.
- The images on the board are safe by MARGIN only, by `shrinkto-edge-check.py` (this folder; negative-tested with a
  stub where the last object ends at the window: TRAPS, rc 1):

  | image | last SHRINKTO-sized object | slack to the window's end |
  |---|---|---|
  | gcc c3f | ends 0xf99920 | 1,760 B |
  | perlbench c3g | ends 0x51fdd0 | 560 B |
  | x264 c3f | ends 0x2dc6a4 | 2,396 B |

- "A QEMU pass rules it out" holds only for the SHRINKTOs that executed.
- Rule from here: every adjust-1 image is checked with shrinkto-edge-check.py before its boot.

**Corrections to the reg_email account** (rest Addenda 3-4):
- The board's own failure cause was NOT observed. The board run has no test.err and no fault line, and its twin of the
  same full suite passes reg_email. What is established is the twin reproduction on reg_email alone: cause 29 in
  realloc's publish derive, 7/8 for the production image and -ac. It is consistent with October's defect, not proven
  to be the board's cause.
- The fix control is not one variable. c3s1 → c3f (same objects) moves the runtime 01967cf → 0ee4200, which carries
  c7d7ca0 besides the two cherry-picks, and its musl differs only in meta.h. AC → c3f also changes compiler and musl.
  1/8 against 0/8 cannot attribute it.
- "HF" is the production devmk image (sdks-dev2mk: wt-perf 86f638759ecf runtime), not the heapfix build. The sweep
  script's docstring is corrected.

**Corrections to the dev comparison:**
- dev's runtime against this branch's is a stack comparison, not one variable. The kits' launchers, modules and
  parameters differ, and so do the SDK configs.
- Its instruction figures count in-context instructions only. Under icount dev's run took 16.41 s against c3f's 14.57 s.
- The one-variable result is PR #306 (dev with and without in-domain reuse, same kit, same objects): entries
  50,991 → 7,698.

**CAPSTONE_DELEGATE_SELF_CACHE:** correct only for contexts created through capstone_virtual_thread_create; a raw
service-3 ecall bypasses the flag. Its threaded path has executed zero times.
