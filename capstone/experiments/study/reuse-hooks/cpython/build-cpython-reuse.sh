#!/usr/bin/env bash
# One CPython 3.13.7 image with the reuse-distance observer at pymalloc's block hand-out and free
# (pre-registration capstone/docs/plans/2026-10-11-nested-reuse-distance.md, 16d2171a3edc).
#
#   build-cpython-reuse.sh stock|protected NEW_OUTPUT
#
# The cpython recipe of ports/common/application/build-virtual.sh, unchanged, with one step
# inserted between configure and compile: the arm's observer patch is applied to the prepared
# source, the observer header is copied beside obmalloc.c, and -DEXP_REUSE_GAP is added to the
# configured Makefile's CONFIGURE_CPPFLAGS. protected = CPY_SUBLET=1 (patch 0014); 0014 rewrites
# the two lines the hooks sit next to, so each arm has its own patch file. The added lines are
# the same in both and sit at the same place: the hand-out hook just before pymalloc_alloc
# returns (block_issue in protected), the release hook in pymalloc_free once the block is known
# to be pymalloc's and p is obmalloc's own pointer (after arena_inside in protected), before
# the block is linked into the pool's free list.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../../.." && pwd)
source "$CAP/tests/capstone-test-env.sh" >/dev/null
[[ $# == 2 ]] || { echo 'usage: build-cpython-reuse.sh stock|protected NEW_OUTPUT' >&2; exit 2; }
ARM=$1
OUT=$(realpath -m "$2")
case "$ARM" in
  stock) export CPY_SUBLET=0; NESTED=none ;;
  protected) export CPY_SUBLET=1; NESTED=cpython ;;
  *) echo "arm: stock or protected, not $ARM" >&2; exit 2 ;;
esac
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT/source"
export CAPSTONE_APPLICATION_PROFILE=virtual CPY_ROOT=$OUT/source
PORTS=$CAPSTONE_REPO_ROOT/capstone/ports
PATCH=$HERE/obmalloc-reuse-gap-$ARM.patch

bash "$PORTS/cpython/app/prepare-cpython-capstone.sh"
SRC=$CPY_ROOT/src/Python-3.13.7
patch -d "$SRC" --batch --forward --fuzz=0 -p1 < "$PATCH"
cp "$CAP/experiments/study/reuse-gap-observer.h" "$SRC/Objects/reuse-gap-observer.h"
sed -i 's/^CONFIGURE_CPPFLAGS=/CONFIGURE_CPPFLAGS=-DEXP_REUSE_GAP /' "$CPY_ROOT/build/Makefile"
[[ $(grep -c '^CONFIGURE_CPPFLAGS=-DEXP_REUSE_GAP ' "$CPY_ROOT/build/Makefile") == 1 ]] \
  || { echo "EXP_REUSE_GAP did not reach CONFIGURE_CPPFLAGS" >&2; exit 2; }
echo "[reuse] $ARM: applied $(basename "$PATCH"), EXP_REUSE_GAP in CONFIGURE_CPPFLAGS" >&2

source "$CPY_ROOT/build/capstone-env.sh"
python3 "$PORTS/cpython/app/survey-cpython-capstone.py" "$CPY_ROOT/build" --jobs "${JOBS:-12}"
python3 "$PORTS/cpython/app/link-cpython-capstone.py" "$CPY_ROOT/build" \
  --native "$CPY_ROOT/build-python" --out "$CPY_ROOT/link"
# As build-virtual.sh: an 8 MiB main stack (CPython's C recursion limit is sized for it).
python3 "$PORTS/common/application/build.py" --app cpython --root "$CPY_ROOT" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 8388608 --out "$OUT/image"
# The hooks must be in the image: their report string is.
grep -c 'REUSE-GAP pymalloc capacity=' "$OUT/image/cpython.dom" >/dev/null \
  || { echo "the image has no REUSE-GAP report" >&2; exit 2; }
sha256sum "$PATCH" "$SRC/Objects/obmalloc.c" "$SRC/Objects/reuse-gap-observer.h" \
  "$OUT/image/cpython.dom" > "$OUT/reuse-build.sha256"
echo "[reuse] $ARM image $OUT/image/cpython.dom" >&2
