#!/usr/bin/env bash
# Build one program per defect through the port's one-source seam.
#
#   shared/build-cases.sh native OUT [cmake options]
#       one port build, then each case.c linked with the native driver:
#       OUT/bin/defect-NN, run as `defect-NN buggy|fixed NN`
#   shared/build-cases.sh capstone-domain OUT [cmake options]
#       the seam invoked once per case: OUT/bin/defect-NN.dom, spatial or
#       sublet chosen by the mode argument at run time
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
TARGET=${1:?usage: build-cases.sh native|capstone-domain OUT [cmake options]}
OUT=${2:?select an output directory}
shift 2
PORT="$REPO/capstone/ports/apr/pools"
mkdir -p "$OUT/bin" "$OUT/work"
built=0
case "$TARGET" in
native)
  work="$OUT/work/port"
  cmake --preset native -S "$PORT" -B "$work" "$@" >"$OUT/work/port.log" 2>&1
  cmake --build "$work" >>"$OUT/work/port.log" 2>&1
  SRC=$(ls -d "$work"/source/apr-*)
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
    cmake --preset capstone-domain -S "$PORT" -B "$work" \
      -DAPRP_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects.dom" "$OUT/bin/defect-$number.dom"
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
