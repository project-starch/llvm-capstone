#!/usr/bin/env bash
# One program per case, bin/defect-NN, against the buffer-pool port's capstone-application library:
# FFmpeg's buffer.c and refstruct.c in a Capstone process, their memory from the process's malloc
# (virtual mallocng). With -DFFPOOL_SUBLET=ON the library carries the port's patch 0003, the pools'
# entries as Sublet lifetimes; without it, the stock pools (virtual-malloc).
#
#   CAPSTONE_SDK=<virtual SDK> build-cases.sh capstone-application OUT [cmake options]
#
# Invoked as controls/virtual/shared/build-cases.sh it builds the virtual controls the same way.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(dirname -- "$HERE")
REPO=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && git rev-parse --show-toplevel)
PORT=$REPO/capstone/ports/ffmpeg/buffer-pool
TARGET=${1:?usage: build-cases.sh capstone-application OUT [cmake options]}
OUT=${2:?usage: build-cases.sh capstone-application OUT [cmake options]}
shift 2
[[ $TARGET == capstone-application ]] || { echo "build-cases.sh builds capstone-application only" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "$OUT exists; use a new directory" >&2; exit 2; }
: "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual Capstone application SDK}"
mkdir -p "$OUT/work" "$OUT/bin"
work=$OUT/work/port
cmake --preset capstone-application -S "$PORT" -B "$work" -DCAPSTONE_SDK="$CAPSTONE_SDK" "$@" \
  > "$OUT/work/port.log" 2>&1
cmake --build "$work" >> "$OUT/work/port.log" 2>&1
SRC=$(ls -d "$work"/sources/ffmpeg-*)
built=0
for dir in "$CORPUS"/[0-9][0-9]_*/; do
  number=$(basename "$dir" | cut -c1-2)
  "$CAPSTONE_SDK/capstone-cc" -std=gnu11 -O1 -g -DFF2_SYSTEM_MEMORY \
    -I"$CORPUS/shared" -I"$PORT/src/shared" -I"$SRC" -I"$PORT/cmake/replay-config" \
    -I"$REPO/capstone/runtime/include" \
    -o "$OUT/bin/defect-$number" "$dir/case.c" "$CORPUS/shared/driver.c" "$work/libffmpeg-pool.a"
  built=$((built + 1))
done
[[ $built -gt 0 ]] || { echo "no NN_*/case.c under $CORPUS" >&2; exit 2; }
echo "built $built programs in $OUT/bin"
