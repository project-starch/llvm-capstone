#!/usr/bin/env bash
# MPY-S31 / upstream 19129: unbounded alloca when the VM state allocation fails.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# Two arms. A shows the alloca fallback runs untrapped; B shows MicroPython's own
# mp_cstack_check ends recursion at depth 8, so this row cannot demonstrate a
# hardware stack trap however it is arranged. Both are expected to RETURN.

repro_scripts capstone/benchmarks/micropython/spatial-corpus/cases/MPY-S31_fun-bc-call-unbounded-alloca mpy_s31
check_row 01_s31_alloca_taken.py     UNSCORED -
check_row 02_s31_alloca_recursive.py UNSCORED -
show_output 01_s31_alloca_taken.py
show_output 02_s31_alloca_recursive.py
