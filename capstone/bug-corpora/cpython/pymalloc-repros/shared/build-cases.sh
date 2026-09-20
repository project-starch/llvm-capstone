#!/usr/bin/env bash
# Build one program per defect through the port's one-source seam.
#
# The seam takes a single corpus source and produces a single target, so it is
# invoked once per case. The programs land side by side in OUT/bin as
# defect-NN, which is what both runners expect.
#
#   shared/build-cases.sh cheribsd OUT [extra cmake options]
#   shared/build-cases.sh capstone-domain OUT [extra cmake options]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
TARGET=${1:?usage: build-cases.sh cheribsd|capstone-domain OUT [cmake options]}
OUT=${2:?select an output directory}
shift 2
PORT="$REPO/capstone/ports/cpython/pymalloc"
mkdir -p "$OUT/bin" "$OUT/work"
built=0
for dir in "$CORPUS"/[0-9][0-9]_*/; do
  number=$(basename "$dir" | cut -c1-2)
  work="$OUT/work/$number"
  case "$TARGET" in
  cheribsd)
    bash "$PORT/host/cheribsd/poisoncap/build.sh" "$work" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    # The runner's platform control comes from the same build.
    cp -f "$work/bin/cheribsd-abi-probe" "$OUT/bin/" 2>/dev/null || true
    ;;
  capstone-domain)
    cmake --preset capstone-domain -S "$PORT" -B "$work" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects.dom" "$OUT/bin/defect-$number.dom"
    ;;
  *) echo "Unknown target: $TARGET" >&2; exit 2;;
  esac
  built=$((built + 1))
done
# No cases found must be an error, never an empty success.
if [ "$built" -eq 0 ]; then
  echo "build-cases.sh: no case directories under $CORPUS" >&2
  exit 2
fi
echo "built $built programs into $OUT/bin"
