# SPEC CPU2017 on the virtual profile: where the instructions go, what was removed, what comes next

State as of 2026-11-02 evening. Everything below is measured with the profiling tools of
`capstone/ports/spec-cpu2017/tools/profile.md` (branch `perf/capprof-hooks`) on p12, test inputs,
instruction attempts counted by capstone-qemu's capprof plugin (branch `perf/capprof-v2`), reported
without the guest's idle loop. The plain arm is the same llvm-capstone source for riscv64 with pristine
musl 1.2.5; the Capstone arm is dev b9dd0ccc's compiler, musl-capstone and virtual SDK. Benchmark
sources are never changed (the src.alt packages are portability fixes); compiler, libc, SDK runtime,
launcher and module are what gets optimized.

## 1. The loop

A change to the runtime, module or launcher reaches a number in minutes: build the adapter
(seconds), put it in a platform directory, `profile-batch.sh` (one guest per arm and benchmark,
all in parallel, 70-115 s wall for gcc/imagick/nab/lbm test), `prof-compare.py` for the difference
per function, `prof-buckets.py` per cause, `prof-annotate.py` per instruction. Counts are
reproducible: between two runs of the same arm only the idle loop differs (reported separately).
The compiler loop (incremental clang, libc/SDK, benchmark rebuild, profile) has not been timed yet;
item 1 of section 5 is its first iteration.

## 2. Measurement artifacts found and removed (not Capstone cost)

| artifact | effect on 502.gcc_r test (Capstone) | fix |
|---|---|---|
| the SSH channel encrypted the launcher's 403 `CAPSTONE_VM_ADD_OK` lines inside the window | 9.3 M attempts in dropbear | guest output goes to files, not the channel |
| firmware without SBI RFENCE: the kernel formatted ~1800 warnings per run | 13.3 M | the RFENCE monitor (platform-rf3) |
| images on the 9p share: CV_IMAGE denial is EOPNOTSUPP there, the launcher copied 23.5 MB into a sealed memfd | ~50 M (copy, shmem pages, memcg accounting, teardown) | `stage_image=tmpfs`: the image is copied to the guest's tmpfs before the window, as a deployment's local file |
| each grant line a write(2) on a 9p file | 5.2 M | guest output on its tmpfs, copied back after the window |
| guest ASLR | fault-heavy windows varied 2.7x | off in the guest |

The plain arm fell too (62.8 M to 32.4 M): demand-paging its 10.7 MB binary over 9p.

## 3. Real changes landed on lane branches

- `runtime/retire-stores` (88c7bdaf, pushed, no PR yet): `clear_page_tags` stores zeros, eight
  granules per iteration, instead of load-and-rewrite per granule. gcc test: module 4.28 M to 2.40 M
  (-1.9 M, -2.1 %); for lbm the retire sweep is 111 M, so about -65 M there. Codex (gpt-6.1-sol)
  adversarial review: no functional defect. Needs the runtime gates before a PR.
- Rejected after review: buffering the grant lines in the launcher (-5.9 M) had five P2 defects
  (flush before the fault record can block or SIGPIPE, fd 2 reuse by the application, lines lost on
  die()/signals that vp2 readers count, the php reader on unterminated output). The harness-side fix
  above has the same effect without touching the launcher.
- Measured and found NOT to be levers: pin batching (`pin_ahead_pages=16/64`, `arena_pin`: fewer
  faults, 1.5-3 M MORE attempts, the materialized pages cost more than the faults saved); the
  5 ms quantum (the step counter counts domain entries including in-ioctl fault resolves; the timer
  share is below 0.7 M).

## 4. Where the overhead is now (test inputs, attempts, plain -> Capstone)

| benchmark | plain | Capstone | delta | what the delta is |
|---|---|---|---|---|
| 502.gcc_r | 32.4 M | 77.9 M | +45 M | kernel per-page work ~30 M (pin, allocate, zero, memcg, locks; Capstone touches 2.7x the pages: 2970 vs 1111 minor faults, 21.5 vs 11.4 MB RSS), program 11.6 M (code +5.2, malloc +4.4, mem* +1.4), module 2.4 M, launcher 2.0 M |
| 538.imagick_r | 200 M | 239 M | +19 % | compiled code 71 % (HorizontalFilter, fgetc 2x), libc mem* +5.2 M |
| 519.lbm_r | 6.67 G | 8.27 G | +24 % | LBM_performStreamCollideTRT +1.24 G (80 %), module retire 111 M, fgetc 2x |
| 544.nab_r | 6.88 G | 8.00 G | +16 % | mme34 +532 M, nbond +101 M, libm exp +208 M (+50 %), log +92 M (+32 %), sqrt 0.3 -> 8.7 M |

Once a program computes, the kernel share is irrelevant and the cost is compiler and libc codegen.
The long four (perlbench, x264, mcf, xz) are not profiled yet: 45-220 min test inputs; the reviewers
rejected "first N attempts" slices (unequal work between arms); a reduced input that finishes and
validates is the honest form and is still to be chosen.

## 5. Codegen plan (three independent reviews, converging; details in the lane's scratch notes)

| # | change | mechanism, verified in source | expected saving | effort, risk |
|---|---|---|---|---|
| 1 | frame pointer only in non-leaf functions | clang `CommonArgs.cpp` `useFramePointerForTargetByDefault` lists riscv64 (:92) but not capstone64, which falls to `return true` (:150), leaf FP also true (:153): `"frame-pointer"="all"`, 8 instructions per leaf call. Fix: NonLeaf for capstone; keep `s0` reserved (RegisterInfo.cpp:149 reserves it only with a frame) so the launcher's fp walk survives; check tail calls (`hasCalls`) | exp -123 M, log -41 M, fgetc -21 M (lbm) / -8..14 M (imagick); frees s0 in hot loops | a day; diagnostics only, judged safe by all three |
| 2 | fold offsets through capabilities | `cincoffset(x,0)` and nested constant `cincoffset` are not collapsed; `SelectAddrRegImm` peels one level; `CapstoneFoldMemOffset` is the integer pass (no CIncOffsetImm, LDC, STC, FLD) | small per site (fgetc 2.6 M, exp 7.6 M, mme34 <= 33 M), everywhere | 2-3 days; legal: bounds are checked at the access, no cursor farther from the object |
| 3 | constant-pool and const-table capabilities | every `.LCPI` is its own `auipc; scc gp` derivation and 128-bit live range: spills and reloads (LBM 112 M ldc, mme34 ~200 M scc); exp derives its table per call with 8 instructions including `shrink` | reviewers disagree: -390 M nab / -176 M lbm (Argon) vs tens of M (Fable): measure | medium |
| 4 | pointer induction variables in LBM | plain hoists `grid + K*8` 16 KiB outside the object, illegal on Capstone (R-60 window, IVUsers.cpp:134-164); legal form: capability IV plus `cincoffset a, p, rK` per access, 1-2 instructions instead of `addi; slli; cincoffset` | -730..960 M (59-75 % of LBM's gap) | 1-2 weeks; register pressure; highest risk |
| 5 | libm sqrt | the port copies `arch/riscv64` but not `src/math/riscv64`, and clang defines `__capstone_flen`, not `__riscv_flen`: generic C sqrt (~87 instructions) instead of `fsqrt.d` | -8.4 M nab | an hour, libc port |
| 6 | prologue `movc s0,sp; cincoffsetimm s0,s0,N` as one instruction | FrameLowering.cpp:1098 calls adjustReg twice | 1 per frame (non-leaf functions, gcc) | an hour |

Order: 1, 5, 6 first (small, safe, ~-200 M on nab together), then 2, then 3 measured rather than
estimated, then 4 as its own project.

## 6. Open decisions

- The corrected platform (RFENCE firmware, images staged to tmpfs) as the standard measurement setup:
  recommended, it is the deployment state.
- Reduced inputs for perlbench, x264, mcf, xz.
- Hardware range clear for the retire sweep (lbm 111 M) is an ISA question.
