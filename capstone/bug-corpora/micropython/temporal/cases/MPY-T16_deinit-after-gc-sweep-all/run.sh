#!/usr/bin/env bash
# MPY-T16 / upstream 5487: a raw pointer to a GC block held across gc_sweep_all().
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# 0x16005a01: the byte read back was 0x5A and the write completed -- untrapped,
# staleness NOT demonstrated. A RECONSTRUCTION of the ESP32 port's ordering.

repro_glue MPY_T16_DEINIT_AFTER_SWEEP mpy_t16 369121793
