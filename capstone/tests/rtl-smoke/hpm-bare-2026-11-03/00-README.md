# hpm-bare-2026-11-03

The stock CVA6 hardware performance counters, run as bare JTAG arms on bitstream 3 (`caplifive_vp2c_e7af48457`) and on
the ObjBounds variant (`caplifive_vp2c_603528b8b`). The SPEC lane asked whether these counters can explain the SPEC
overhead without a new bitstream. This kit is the silicon half of that answer; the simulation half is capstone-ariane
`hpm-probe.S` on `hpm-probe-b3` 87368cc69 and `hpm-probe-objb` fc1114913 (the arms were first run at eedb57e45 / 58f84f0b6; arm 7's region moved out of `.data` after the first kit simulation read 65 misses of 256 there, because the wrapper's prologue loads every data doubleword).

What the source says (e7af48457; the perf path is byte-identical on 603528b8b):
- There are 6 counters, `mhpmcounter3..8`, programmed through `mhpmevent3..8` (`perf_counters.sv:105-135`). They are
  writable from M and inhibited by `mcountinhibit[8:3]`.
- There is no privilege filter.
- "Load" (event 5) and "store" (event 6) count committed instructions with fu LOAD or STORE. LDC and STC decode and
  commit with fu CAPSTONE_DYN, so they count as neither.
- Each image programs counters 3..8 as: load (5), store (6), L1D access cycles (17), L1D miss (2), branch (9, the
  control) and issue stall (22, which is `stall_raw[0]`: operand-not-ready at issue port 0).
- Each image zeroes the counters, runs N = 256 iterations of one arm, and prints:
  - the six deltas, in counter order 3..8;
  - the mcycle and minstret deltas;
  - `mhpmevent3` read back;
  - the trap count and the trap cause.

| image | arm | what it shows |
|---|---|---|
| hpm-5 | empty loop | control: only branch = 256 |
| hpm-1 | ld from one doubleword | load = 256, store = 0 |
| hpm-2 | sd | store = 256 |
| hpm-3 | ldc | load = store = 0; the check's cost shows as issue stall, on bitstream 3 only |
| hpm-4 | stc | the same as hpm-3, for stores |
| hpm-6 | hpm-1 with `mcountinhibit[8:3]` set | all six read 0; `mhpmevent3` reads 5 |
| hpm-7 | ld at a 16 B stride over 4 KiB at 0x80100000, outside the image | L1D miss counts one per line |

Every image returns on both bitstreams, so they can run in any order. The recommended order puts the hpm-5 control
first. Each image is a single bare boot (`SUP_IMG`, `SUP_REC_ADDR` = its `.rec`) with the runner
`sup-bare-2026-10-03-ladder/run_sup_bare.py`.

How the kit is built and graded:
- `build.sh` rebuilds the images from `tests/`. `tests/hpm-probe.S` is equal to the test at both capstone-ariane
  commits, and `build.sh` checks that against `VP_SRC` when it is given.
- The wrapper is `objb-bare-2026-10-30`'s `inc/`.
- `images/SHA256SUMS` pins the images.
- `PREREG.md` holds the vectors, simulated from THESE images on both trees, and the grading rule.

Siblings:
- `objb-bare-2026-10-30` reads the same check cost as cycles (its s6 and s7 shapes).
- That kit's `objb-arcfour` sim record explains why issue stall is not the whole story for a loop bound by the store
  buffer.
