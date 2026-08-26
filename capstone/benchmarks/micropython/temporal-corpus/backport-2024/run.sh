#!/usr/bin/env bash
# MPY-T02 / CVE-2024-8947 and MPY-T05 / #13283, measured in the domain by building
# the fix commit's PARENT with our port. Both are fixed at the pin, so there is no
# other way to run them here.
#
# Sets up the worktree, applies the portability patches, builds and runs. Safe to
# re-run: the worktree and the patch application are both idempotent.
#
# WHY THE FEATURE PROFILE IS CORE AND NOT EXTRA: at CORE_FEATURES
# MICROPY_PY_ARRAY_SLICE_ASSIGN is off, so the fix's SECOND case -- assigning to a
# slice from itself -- is not reachable and the script reports -1 for it. The primary
# case still produces a verdict. Raising the profile is not a free improvement; it
# changes which of the two upstream cases is under test.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
export CAPSTONE_REPO_ROOT="$PWD"
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1

D=capstone/benchmarks/micropython/temporal-corpus/backport-2024
TREE=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/mpy-t02dom
PARENT=4bed614e707c^        # parent of the commit that fixes both rows

if [[ ! -d $TREE ]]; then
  echo "== creating the 2024 worktree at $PARENT"
  git -C "${CAPSTONE_TMP_ROOT:-/tmp/capstone}/micropython" worktree add "$TREE" "$PARENT" \
    || { echo "worktree failed"; exit 1; }
  echo "== applying the portability patches"
  # 17 of 20 apply directly, 0003 needs -3, 0010 is a context conflict that must be
  # hand-applied (both hunks are character-identical in the 2024 tree, only at other
  # line numbers), and 0012 is dropped -- it is stream ioctl, which this trigger
  # does not touch. Anything unexpected is reported rather than skipped silently.
  for p in capstone/benchmarks/micropython/patches/*.patch; do
    b=$(basename "$p")
    case $b in 0012-*) echo "   skipped $b (stream ioctl, not on this path)"; continue;; esac
    if git -C "$TREE" apply --3way "$(realpath "$p")" 2>/dev/null; then
      echo "   applied $b"
    else
      echo "   MANUAL: $b did not apply -- see $D/README.md"
    fi
  done
fi

echo "== building the domain from the 2024 tree"
MPY_SRC_DIR="$TREE" \
MPY_TESTS=all MPY_TEST_BASE_DIR=capstone-temporal MPY_TEST_INCLUDE_UNSUPPORTED=1 \
DOM_NAME=t02dom \
DOMAIN_EXTRA_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES \
                   -I$PWD/$D/shim2024 -include $PWD/$D/mpy2024-compat.h" \
  bash capstone/benchmarks/micropython/build-micropython-silicon.sh \
  >/tmp/capstone/t02dom-build.log 2>&1 \
  || { echo "BUILD FAILED -- tail:"; tail -20 /tmp/capstone/t02dom-build.log; exit 1; }
echo "   built $(md5sum "${CAPSTONE_TMP_ROOT:-/tmp/capstone}/micropython-silicon/t02dom.dom" | cut -c1-12)"
echo
echo "Run it with tools/run-resumable-suite.py as in ../../REPRODUCING.md;"
echo "recorded result: extend-from-self runs in the domain and does NOT fault"
echo "(untrapped-no-crash), which is the measurement."
