#!/usr/bin/env bash
# Build one program per defect through the port's one-source seam, with the
# bucket allocator switched on (-DAPRP_BUCKETS=ON: the pools port carries it).
#
#   shared/build-cases.sh native OUT [cmake options]
#       one port build, then each case.c linked with the native driver:
#       OUT/bin/defect-NN, run as `defect-NN buggy|fixed NN`
#   shared/build-cases.sh capstone-domain OUT [cmake options]
#       the seam invoked once per case: OUT/bin/defect-NN.dom, spatial or
#       sublet chosen by the mode argument at run time
#   shared/build-cases.sh cheribsd OUT [cmake options]
#       the seam once per case for CheriBSD purecap: OUT/bin/defect-NN, plus
#       the platform's ABI probe, the revocation control and the supervisor
#       that observes each program from outside. CHERI_SDK and CHERI_SYSROOT
#       must name a matching pair.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
TARGET=${1:?usage: build-cases.sh native|capstone-domain|cheribsd OUT [cmake options]}
OUT=${2:?select an output directory}
shift 2
PORT="$REPO/capstone/ports/apr/pools"
BUCKETS=-DAPRP_BUCKETS=ON
mkdir -p "$OUT/bin" "$OUT/work"
built=0
case "$TARGET" in
native)
  work="$OUT/work/port"
  cmake --preset native -S "$PORT" -B "$work" $BUCKETS "$@" >"$OUT/work/port.log" 2>&1
  cmake --build "$work" >>"$OUT/work/port.log" 2>&1
  SRC=$(ls -d "$work"/source/apr-1.*)  # APR, not apr-util, which sits beside it
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    # The same ABI knob the port builds with, so header sizes agree.
    "${CC:-cc}" -std=c11 -O1 -g -Wall -Wextra -DAPR_ALIGN_DEFAULT_BOUNDARY=16 \
      -I"$CORPUS/shared" -I"$PORT/src/shared" -I"$SRC/include" -I"$PORT/../adapted" \
      -o "$OUT/bin/defect-$number" "$dir/case.c" "$CORPUS/shared/driver.c" \
      "$work/libapr-pools.a"
    built=$((built + 1))
  done
  ;;
capstone-domain)
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    work="$OUT/work/$number"
    cmake --preset capstone-domain -S "$PORT" -B "$work" $BUCKETS \
      -DAPRP_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects.dom" "$OUT/bin/defect-$number.dom"
    built=$((built + 1))
  done
  ;;
cheribsd)
  : "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
  : "${CHERI_SYSROOT:?set CHERI_SYSROOT to its matching rootfs}"
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    work="$OUT/work/$number"
    cmake --preset cheribsd -S "$PORT" -B "$work" $BUCKETS \
      -DAPRP_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    # The runner's platform controls come from the same build.
    cp -f "$work/bin/cheribsd-abi-probe" "$work/bin/revocation-control" "$OUT/bin/"
    # The supervisor that observes each program's fault from outside it: the
    # pymalloc corpus's, built with this corpus's label. Referenced, not copied.
    if [ ! -x "$OUT/bin/supervise" ]; then
      "$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 \
        -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin" \
        --sysroot="$CHERI_SYSROOT" -std=gnu11 -O1 -Wall -Wextra -fuse-ld=lld \
        -DPROBE_SYMBOL='"apr_defect_read"' \
        "$REPO/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" \
        -lutil -o "$OUT/bin/supervise"
    fi
    built=$((built + 1))
  done
  ;;
*) echo "Unknown target: $TARGET" >&2; exit 2;;
esac
# No cases found must be an error, never an empty success.
if [ "$built" -eq 0 ]; then
  echo "build-cases.sh: no case directories under $CORPUS" >&2
  exit 2
fi
echo "built $built programs into $OUT/bin"
