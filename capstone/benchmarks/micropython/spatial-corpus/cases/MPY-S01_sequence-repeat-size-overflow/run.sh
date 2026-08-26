#!/usr/bin/env bash
# MPY-S01 / upstream 19314: seq * n sized in unchecked size_t.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# OPEN upstream, so present at the pin and needing no parent build. The three arms
# are expected to FAULT with cause 7; the bounds width is the finding, not the fault:
# 0x60000 = 384 KiB = the whole heap for the bytes and list arms, so the write ran the
# entire length of the heap before anything stopped it.

repro_scripts capstone/benchmarks/micropython/spatial-corpus/cases/MPY-S01_sequence-repeat-size-overflow mpy_s01
check_row 01_s01_bytes_repeat.py FAULT -
check_row 02_s01_list_repeat.py  FAULT -
check_row 03_s01_tuple_repeat.py FAULT -
echo "   bounds reported by the hardware:"
grep -ah "Cap mem access" /tmp/capstone/mpy_s01-run/round-*.log | sed 's/^/     /'
