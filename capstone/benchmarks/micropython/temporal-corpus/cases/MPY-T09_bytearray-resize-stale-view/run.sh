#!/usr/bin/env bash
# MPY-T09 / upstream 18168: bytearray resized under a live memoryview.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# The retval is an FNV-1a of the output and was computed from STOCK MicroPython's
# output BEFORE the domain run, so a match is a prediction that held.
#
# All six script-driven temporal rows share ONE image; running them separately
# rebuilds it each time. That is the price of a case folder that stands alone.

repro_scripts capstone/benchmarks/micropython/temporal-corpus/repros/domain-measured mpy_temporal
check_row 01_t09_stale_view.py UNSCORED 0x0018e966
show_output 01_t09_stale_view.py
