#!/usr/bin/env bash
# MPY-S05 / upstream 15271: append after a grow that raised MemoryError.
#
# Re-runs the measurement recorded in RESULT.txt and says whether it still holds.
# The control runs first and a run whose control fails exits 75 with NO verdict,
# the same convention as tests/fpga-repros/*/run.sh.
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"

# This row needs the upstream fix REVERTED on the pinned tree first, and reversed
# again afterwards -- that tree is shared by every other build here, and a stale
# revert would put this defect into images that are supposed to be stock. The trap
# below runs on any exit, including a failed build or an interrupt.
MPY_TREE=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/micropython
PATCH=$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/spatial-corpus/cases/MPY-S05_array-append-after-failed-grow/revert-the-fix.patch

git -C "$MPY_TREE" apply -R --check "$PATCH" 2>/dev/null \
  && die "the fix is already reverted in $MPY_TREE -- reverse it before running"
git -C "$MPY_TREE" apply "$PATCH" || die "could not apply the reversal"
trap 'git -C "$MPY_TREE" apply -R "$PATCH" 2>/dev/null && echo "   fix restored"' EXIT
echo "== fix reverted in the working tree; it will be restored on exit"

repro_scripts capstone/benchmarks/micropython/spatial-corpus/cases/MPY-S05_array-append-after-failed-grow mpy_s05
check_row 01_s05_append_after_failed_grow.py UNSCORED -
show_output 01_s05_append_after_failed_grow.py
echo "   expected: S05 1 4104 -1  (1 = the MemoryError fired, so the precondition WAS created)"
