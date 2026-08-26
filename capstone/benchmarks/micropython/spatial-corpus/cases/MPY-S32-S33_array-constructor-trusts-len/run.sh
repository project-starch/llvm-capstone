#!/usr/bin/env bash
# MPY-S32 / MPY-S33: array constructors trusting a user __len__.
#
# Both trap, and both trap with cause 24 -- the TAG check, not a bounds check -- so
# neither belongs in untrapped-cma-objects. The first arm is the one that matters:
# a BOUNDED overrun returns, which is what shows the write itself is invisible and
# the fault in the other two arms is downstream of it.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

repro_scripts capstone/benchmarks/micropython/spatial-corpus/cases/MPY-S32-S33_array-constructor-trusts-len mpy_s32
check_row 01_s32_bounded_overrun.py     UNSCORED -
check_row 02_s33_array_len_overflow.py  FAULT    -
check_row 03_s32_bytearray_short_len.py FAULT    -
show_output 01_s32_bounded_overrun.py
echo "   expected: S32b 1 -1  (the 47-byte overrun completed untrapped)"
