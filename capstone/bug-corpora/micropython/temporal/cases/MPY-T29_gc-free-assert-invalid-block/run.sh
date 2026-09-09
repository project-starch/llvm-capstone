#!/usr/bin/env bash
# MPY-T29 / upstream 4705: a live block whose only reference the root scan cannot see.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# 0x29007701: the stale read returned 0x77, the value a LATER allocation wrote --
# reuse of the freed storage DEMONSTRATED. A RECONSTRUCTION.

repro_glue MPY_T29_HIDDEN_ROOT mpy_t29 687896321
