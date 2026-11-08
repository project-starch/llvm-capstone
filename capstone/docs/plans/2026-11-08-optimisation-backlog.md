# Optimisation backlog: every lever, to board-ready, while the boards run SPEC train (8-11 Nov)

The lead's instruction (8 Nov): measure train now with each benchmark's best board-validated configuration
(results/2026-11-08-train); a universal configuration comes later. Meanwhile keep optimising with everything but the
boards (QEMU, RTL simulation, synthesis), so that every lever is **board-ready** when the boards free after the
Wednesday deadline. Several bitstream variants are to be built and sealed in advance.

**Board-ready** means all of:
- the image or bitstream exists, with its hash recorded;
- it passed the QEMU twin (rc 0, specdiff, outputs equal) on every benchmark it will run;
- safety fixtures pass, with their positive controls;
- the kit is built, SHA256SUMS are current, and the driver dry run reached its last rung;
- the pre-registration is drafted;
- for a bitstream: lint plus the synthesis-only gate (40 GB guard), a full build under synth-guard, the bitstream
  sealed with its sha256, and a known-good fallback registered.

Where the overhead is (test size; results/2026-11-08-dcnofp, the mcf counter reading of 8 Nov, the compiler lane's
per-PC profiles):
- perlbench: 1.38×, of which in context only 1.13×. The rest is exits: ~500 k cycles and ~68 k instructions per exit,
  mostly Linux work around the exit.
- mcf: 1.24×. In-context instructions are 0.99× Plain. The inside half is footprint: 16-byte pointers make struct arc
  72 → 112 B, with L1D misses 1.31× and DTLB misses 1.36× Plain. The outside half is exits.
- gcc: 1.196×. The others are already under 1.20×.

## A. Compiler (compiler lane)

| # | Lever | State | Missing for board-ready |
|---|---|---|---|
| A1 | direct calls + no frame pointer (dcnofp) | gcc board-validated (1.196×); QEMU twins pass for perl −2.61 %, xz −1.88 %, nab −2.19 %, imagick −0.36 %, mcf −0.26 % net | train twins; images for the universal config |
| A2 | drop redundant SHRINK(TO), declarations mode + link-time size check | QEMU perl −1.25 %, gcc −0.80 %; authority suite clean at -O2; compiler-spec-c4 5069028a65b0 | wire the check into capstone-cc's link; images for all 8 |
| A3 | libc built with the application's flags (-fomit-frame-pointer, direct calls) | proposed (allocator agent) | build and QEMU |
| A4 | shrink-wrapping of libc entry paths (__unlock, free) | proposed | analysis |
| A5 | 6-7 instructions per global access, 2 movc per byte copy, software CLZ, slot_allzero not inlined | proposed | analysis |

## B. Runtime and libc

| # | Lever | State | Missing |
|---|---|---|---|
| B1 | allocator and copy patch (perf/heap-excess 76455aff7e7a): calloc check reuse, inline unlock test, libc -O2, word tails in copies | QEMU perl −1.27 % in context; malloc fixtures 24/24, hosted 94/94, controls fire | an SDK with it; QEMU twins of all 8; libc-test at -O2 |

## C. Exits: module, kernel, firmware

| # | Lever | State | Missing |
|---|---|---|---|
| C1 | kfast: serve read/write/lseek/fstat/clock_gettime in the module (perf/kernel-fastpath 3f7c8e19fecc) | QEMU: −4.4 k instructions per served call, 31 % of perl's calls served, exits −29 %; audit fixes in. Board module 40e2e0a9fd831a53 (0 compressed instructions; the recipe's control rebuild of bkit30-unify's module matches it section for section), kit bkit35-kfast (+ launcher 02f1a98eaf4d209d, kfast=1), driver perf/univmod/un-run-kf.sh dry-run to DONE | pre-registration: Tier 1 boot (gcc train01 unify-nd) then perl test |
| C2 | the Linux work per exit: mostly the periodic tick REPLAYED after each masked step (HZ_PERIODIC: ~30 k cycles per step + 13.5-14.4 k cycles per 10 ms masked, R² ≥ 0.999 over 5 perl runs; predicts gcc/mcf/x264/xz within 1-6 %). Replay share: gcc train 194 s, mcf test ~510 s, perl test ~1,290 s, x264 test ~2,140 s (perf/exit-kernel 90b69152e63f, ACCOUNTING.md) | K1 built: a NO_HZ_IDLE board kernel, kit t3-smoke/bkit35-nohz (= bkit27-gtq + that kernel), passes QEMU gcc train / perl test, dry run reaches its last rung, all six gates fire on negatives, PREREG-nohz.md drafted. Ceiling perl 1,000-1,280 s, gcc 150-190 s; QEMU suggests about half | the board run with its Plain control (the kernel is shared, so Plain runs on it too) |
| C6 | the board heartbeat's cost: wc over /tmp/step.out every 30 s on the one hart (~6.8 s per beat with a 762 KB dump) | a measurement fix, not a speed-up. The train stages are unaffected (step output 136 B) | make the heartbeat O(1) (stat sizes) for every arm in future runs; V2 hbfix stage drafted (perf/exit-kernel PREREG-hbfix.md) |
| C3 | 32 M supervisor quantum on every benchmark | gcc −38 s; used for 6 of 8 in train | — |
| C4 | the unify stack's outside excess on gcc: +440 s against c3f, board-only (QEMU 0.1 M cycles per step on both) | lead: the dev launcher's sched_yield() on every quantum exit; grant log and console ruled out | exit agent; then the 2×2 on the board (c3f image on the unify kit, devrg image on the gtq kit) |
| C5 | THP in the unify module | the thp commit b23b38ce15b4 conflicts with dev's new next_resident() page walk; not a mechanical port (the walk carries the retire's tags) | a careful port with fixtures; mcf keeps bkit29-thp meanwhile |

## D. Native Linux-task integration (native/linux-task-integration)

| # | Lever | State | Missing |
|---|---|---|---|
| D1 | direct register syscall ABI, firmware-emulated check (native/checked-emu-measure) | QEMU perl −1.72 %, gcc 0; 10-16 checks per call. The C.5a port (native/c5a-route fbfed17d2b1a) STOPS on three hardware gaps: the unpinned memory model needs CPTAG/CLDRAW/CSTRAW, which the monitor cannot emulate (S registers are saved as integers); a second live native task waits forever without slot transport (CSRUNV 4/5); the adapter targets Linux 6.1, the board 6.4.14 | **lead's decision**: park native on silicon (recommended), or the RTL (transport, paged table, raw-tag instructions) + a 6.4.14 port |
| D2 | the RTL checker (plan steps 2-4: epoch/kinds publication, hidden snapshot, multi-cycle DYN check) | stopped: without D1's hardware gaps closed it has no consumer on C.5a | the lead's decision on D1 |

## E. Bitstream variants (rtl lane, synthesis lane): built and sealed before the boards free

| # | Variant | Why | Ready when |
|---|---|---|---|
| E0 | stage-B base a481031ca = C.5a + R-32 fix + R-68 NI fix + DYN handshake trim + LDC rs1; the control every variant builds on | trim: stc/ldc pair through one base 11.5 → 9.5 cycles per iteration in sim; synthesis job #1 on apollo | sealed; board gates as C.5a's plus one SPEC test stage, first after Wednesday; C.5a stays the fallback |
| E1 | native checker (D2) | stopped with D1 | the lead's decision |
| E2 | 16-way L1D (64 KiB) and/or DTLB 32/64 entries (a 32-byte line is out: the tag path assumes line = 16-byte granule) | mcf's footprint half; Plain re-measured on the same bitstream | the QEMU cache model (perf/dcm) must reproduce silicon's 2.90 G / 3.80 G L1D misses first; built only if ≥ 1-2 % of mcf's in-context cycles |
| E3 | larger lifetime verdict cache | if mcf's hot set exceeds 256 nodes (64,004 nodes in its run); misses cost ~12-13 cycles | the hot-set count first (desk or QEMU), then sealed |
| E4 | cheaper escape/resume | sized and NOT a lever: ~760 cycles per round trip in sim, ≤ ~5 k on silicon with the TLB refill, ≤ 1 % of an exit, ≤ 0.2 % of a run | — |

## F. Instruments

| # | Item | State |
|---|---|---|
| F1 | GDB stuck-pc reader (perf/tickgdb, kit bkit34-tickgdb) | built, dry-run, pre-registered (tickcost Addendum 1); boards after Wednesday |
| F2 | board-idle watchdog v3 (perf/watch/board-idle-watch3.py) | running |

## After Wednesday: first board days

1. The universal configuration: A1 + A2 + B1 (+ A3 if ready) on one kit, test size first on all 8.
2. C1 kfast: Tier 1 boot alone, then perl.
3. The bitstream variants E1-E3 in order of their expected gain, each with its Plain pair where the change also helps
   Plain.
