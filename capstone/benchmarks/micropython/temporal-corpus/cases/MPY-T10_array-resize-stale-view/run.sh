#!/usr/bin/env bash
# MPY-T10 / upstream 18171: array('I') resized leaves a stale memoryview.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# An adaptation: upstream uses open()+readinto() as the write vehicle and this
# domain has no filesystem. The defect under test is unchanged.
#
# All six script-driven temporal rows share ONE image; running them separately
# rebuilds it each time. That is the price of a case folder that stands alone.

repro_scripts capstone/benchmarks/micropython/temporal-corpus/repros/domain-measured mpy_temporal
check_row 02_t10_array_stale_view.py UNSCORED 0x00289ce0
show_output 02_t10_array_stale_view.py
