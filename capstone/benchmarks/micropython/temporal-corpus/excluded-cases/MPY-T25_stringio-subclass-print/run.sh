#!/usr/bin/env bash
# MPY-T25 / upstream 10402: printing to a pure-Python io.StringIO subclass.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# KEPT AS A LABELLED COUNTER-EXAMPLE. This is CWE-908, uninitialised state, NOT a
# temporal defect -- and it produces the SAME cause 24 as MPY-T11 and MPY-T12.
#
# All six script-driven temporal rows share ONE image; running them separately
# rebuilds it each time. That is the price of a case folder that stands alone.

repro_scripts capstone/benchmarks/micropython/temporal-corpus/repros/domain-measured mpy_temporal
check_row 06_t25_stringio_print.py FAULT -
show_output 06_t25_stringio_print.py
