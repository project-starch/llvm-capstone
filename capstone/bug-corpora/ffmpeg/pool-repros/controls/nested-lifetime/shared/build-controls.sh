#!/usr/bin/env bash
# One program per nested-lifetime control, bin/ctl-NN, against the buffer-pool port's
# capstone-application library: FFmpeg 9.0.1's libavutil/buffer.c (and refstruct.c) in a Capstone
# process, their memory from the process's malloc (virtual mallocng). A copy of
# controls/virtual/shared/build-cases.sh that takes the arm and passes it to the control as
# CTL_PROTECTED, because control 91 reads a field (sublet_life) only patch 0003 has.
#
#   CAPSTONE_SDK=<virtual SDK> build-controls.sh capstone-application OUT protected|release [cmake options]
#
#   protected  -DFFPOOL_SUBLET=ON: buffer-pool patches 0001 + 0003, every hand-out a CDERIVE child of
#              its entry's lifetime, revoked (CREVOKE) by pool_release_buffer
#   release    -DFFPOOL_SUBLET=OFF: the same source without 0003 (0001's event hooks only)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTROLS=$(dirname -- "$HERE")
REPO=$(cd -- "$HERE" && git rev-parse --show-toplevel)
PORT=$REPO/capstone/ports/ffmpeg/buffer-pool
usage="usage: build-controls.sh capstone-application OUT protected|release [cmake options]"
TARGET=${1:?$usage}
OUT=${2:?$usage}
ARM=${3:?$usage}
shift 3
[[ $TARGET == capstone-application ]] || { echo "build-controls.sh builds capstone-application only" >&2; exit 2; }
case $ARM in
  protected) SUBLET=ON PROTECTED=1 ;;
  release) SUBLET=OFF PROTECTED=0 ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
[[ ! -e $OUT ]] || { echo "$OUT exists; use a new directory" >&2; exit 2; }
: "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual Capstone application SDK}"
mkdir -p "$OUT/work" "$OUT/bin"
work=$OUT/work/port
cmake --preset capstone-application -S "$PORT" -B "$work" -DCAPSTONE_SDK="$CAPSTONE_SDK" \
  -DFFPOOL_SUBLET=$SUBLET "$@" > "$OUT/work/port.log" 2>&1
cmake --build "$work" >> "$OUT/work/port.log" 2>&1
SRC=$(ls -d "$work"/sources/ffmpeg-*)
[[ $(wc -w <<< "$SRC") == 1 ]] || { echo "expected one prepared source under $work/sources: $SRC" >&2; exit 2; }
# The control includes buffer_internal.h; its BufferPoolEntry must be the one the library was
# built from, so the header must carry patch 0003 exactly when the arm does.
if grep -q sublet_life "$SRC/libavutil/buffer_internal.h"; then has=1; else has=0; fi
[[ $has == "$PROTECTED" ]] || { echo "$SRC/libavutil/buffer_internal.h: patch 0003 present=$has, arm $ARM" >&2; exit 2; }
# buffer_internal.h includes <stdatomic.h>, which the library's compile finds through the toolchain
# file's CMAKE_C_FLAGS (the compiler's own resource headers); the controls take the same flags.
cflags_line=$(sed -n 's/^CMAKE_C_FLAGS:STRING=//p' "$work/CMakeCache.txt")
eval "cflags=($cflags_line)"
built=0
for dir in "$CONTROLS"/[0-9][0-9]_*/; do
  number=$(basename "$dir" | cut -c1-2)
  "$CAPSTONE_SDK/capstone-cc" "${cflags[@]}" -std=gnu11 -O1 -g -DFF2_SYSTEM_MEMORY -DCTL_PROTECTED=$PROTECTED \
    -I"$CONTROLS/shared" -I"$PORT/src/shared" -I"$SRC" -I"$PORT/cmake/replay-config" \
    -I"$REPO/capstone/runtime/include" \
    -o "$OUT/bin/ctl-$number" "$dir/case.c" "$CONTROLS/shared/driver.c" "$work/libffmpeg-pool.a"
  built=$((built + 1))
done
[[ $built -gt 0 ]] || { echo "no NN_*/case.c under $CONTROLS" >&2; exit 2; }
echo "built $built programs in $OUT/bin ($ARM, FFPOOL_SUBLET=$SUBLET)"
