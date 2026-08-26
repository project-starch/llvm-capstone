#!/usr/bin/env bash
# MPY-T13 / upstream 19075: write() whose slice assignment reallocates under the caller.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# Stock segfaults, the domain prints 'T13 survived'. A separate probe confirmed the
# domain really does hand write() a bytearray, so the trigger IS created.
#
# All six script-driven temporal rows share ONE image; running them separately
# rebuilds it each time. That is the price of a case folder that stands alone.

repro_scripts capstone/benchmarks/micropython/temporal-corpus/repros/domain-measured mpy_temporal
check_row 05_t13_modio_write.py PASS 0x005da053
show_output 05_t13_modio_write.py
