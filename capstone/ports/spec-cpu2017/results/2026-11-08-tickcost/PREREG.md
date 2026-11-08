# Board runs for the rtl lane's tickcost kit (pointer record)

The predictions and grading are the rtl lane's: llvm-capstone land/rtl-tickcost 485e2fe8917e,
capstone/tests/rtl-smoke/tickcost-2026-11-08/PREREG.md (pushed before any run). This record only says how the SPEC
lane runs it on the board.

- Board 3, two boots, after the quantum run there (results/2026-11-08-quantum): stage tickcost-E1, then tickcost-E2
  (perf/tickcost/stages; run.sh calls the kit's tickcost-guest.sh with tickcost.ko c16c5c56585b84c1, arm E1 or E2).
- Kit bkit33-tickcost = bkit27-gtq (C.5a, module 65ede944a87ea742) with its own driver copy
  (perf/tickcost/driver, run_baked_rungs_fpga.py 286f322bed76424a): the only change reads aperture 224 ten more
  times after a wedge (BAKED_WEDGE_REPEAT_224=10); each aperture read is still a pair 0.6 s apart, unstable marked.
- The control rungs run first in each boot. The raw transcripts go to the rtl lane for grading.
