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
| C1 | kfast: serve read/write/lseek/fstat/clock_gettime in the module (perf/kernel-fastpath 3f7c8e19fecc) | QEMU: −4.4 k instructions per served call, 31 % of perl's calls served, exits −29 %; audit fixes in | the board kit, Tier 1 boot plan, dry run; board go/no-go from the board's own per-exit kcyc/ucyc |
| C2 | the Linux work per exit (~60 k instructions: interrupt replay, missed-tick catch-up, fault resolution, scheduling) | the largest lever for perl and mcf; not yet attacked | find which of it is avoidable (board kernel config: tick and NO_HZ, catch-up; module: fault-ahead, lazy grants, resume_quantum), build kernel/kit variants, QEMU |
| C3 | 32 M supervisor quantum on every benchmark | gcc −38 s; used for 6 of 8 in train | — |

## D. Native Linux-task integration (native/linux-task-integration)

| # | Lever | State | Missing |
|---|---|---|---|
| D1 | direct register syscall ABI, firmware-emulated check (native/checked-emu-measure) | QEMU perl −1.72 %, gcc 0; 10-16 checks per call | a C.5a port (RTL type numbering +1, flat RECYCLE table, Q-35) to measure on silicon without a bitstream |
| D2 | the RTL checker (plan steps 2-4: epoch/kinds publication, hidden snapshot, multi-cycle DYN check) | on hold after D1's number | lint, synthesis-only gate, full build, sealed: one of the bitstream variants (E1) |

## E. Bitstream variants (rtl lane, synthesis lane): built and sealed before the boards free

| # | Variant | Why | Ready when |
|---|---|---|---|
| E1 | native checker (D2) | the native line, ready if D1 on silicon justifies it | sealed |
| E2 | larger L1D (more ways; VIPT limits a way to 4 KiB) | mcf's footprint half; Plain must be re-measured on the same bitstream | sealed, with a Plain pair planned |
| E3 | larger lifetime verdict cache | if mcf's hot set exceeds 256 nodes (64,004 nodes in its run); misses cost ~12-13 cycles | the hot-set count first (desk or QEMU), then sealed |
| E4 | cheaper escape/resume (switcher save/restore) | the per-exit cost in hardware | rtl lane to size it |

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
