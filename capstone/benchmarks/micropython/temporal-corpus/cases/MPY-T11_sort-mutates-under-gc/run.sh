#!/usr/bin/env bash
# MPY-T11 / upstream 17941: sort over an array while the collector runs.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# Faults with cause 24, an untagged word used as a pointer. Note cause 24 does NOT
# distinguish a temporal defect from an uninitialised one -- see MPY-T25.
#
# All six script-driven temporal rows share ONE image; running them separately
# rebuilds it each time. That is the price of a case folder that stands alone.

repro_scripts capstone/benchmarks/micropython/temporal-corpus/repros/domain-measured mpy_temporal
check_row 04_t11_sort_gc.py FAULT -
show_output 04_t11_sort_gc.py
