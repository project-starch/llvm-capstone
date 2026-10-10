#!/usr/bin/env bash
# Build one program's full-configuration objects: its nested allocator's Sublet port plus the
# constructor that brings the port up before main(). The objects go to OUT and are handed to
# tools/run-capstone-domain.py --arm sublet-full as --cc-arg inputs.
#
#   build-objects.sh ffmpeg    SDK OUT FFMPEG_SRC FFMPEG_BUILD   (the app port's --sublet source and
#                                                              its ffmpeg-build-poolsublet tree)
#   build-objects.sh wireshark SDK OUT
#
# SDK must be an application SDK built with CAPSTONE_APPLICATION_HEAP=sublet: every port lends its
# blocks from the Sublet heap (__capstone_sublet_malloc_linear). memcached's full configuration
# (its allocators port's ledger on the Sublet authority) was removed on 2026-10-11 with that
# authority; its recorded sublet-full verdicts stay in the plain corpora's case files.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../.." && pwd)
PROG=${1:?ffmpeg|wireshark}; SDK=${2:?SDK}; OUT=${3:?OUT}
CC=$SDK/capstone-cc
mkdir -p "$OUT"
grep -q '^CAPSTONE_APPLICATION_HEAP:STRING=sublet$' "$SDK/CMakeCache.txt" \
  || { echo "build-objects: $SDK is not a Sublet-heap SDK" >&2; exit 2; }
case $PROG in
ffmpeg)
  SRC=${4:?FFMPEG_SRC}; BLD=${5:?FFMPEG_BUILD}
  CI=$("$CC" -print-resource-dir)/include
  "$CC" -O1 -c -I"$SRC" -I"$CAP/sublet" "$CAP/ports/ffmpeg/app/src/capstone-domain/ffsublet.c" -o "$OUT/ffsublet.o"
  "$CC" -O0 -c -isystem "$CI" -I"$SRC" -I"$BLD" "$HERE/ffmpeg.c" -o "$OUT/full-config.o"
  cp "$BLD/libavutil/libavutil.a" "$OUT/libavutil.a"
  ;;
wireshark)
  WP=$CAP/ports/wireshark/wmem
  for s in "$WP/src/allocators/sublet/chunks.c" "$CAP/ports/wireshark/app/src/tsapp-wmem-chunks.c"; do
    "$CC" -O1 -std=c11 -DWM_DOMAIN -I"$WP/src/shared" -I"$CAP/runtime/include" -c "$s" -o "$OUT/$(basename "${s%.c}").o"
  done
  "$CC" -O0 -std=c11 -DWM_DOMAIN -I"$WP/src/shared" -I"$CAP/runtime/include" -c "$HERE/wireshark.c" -o "$OUT/full-config.o"
  ;;
*) echo "build-objects: no program $PROG" >&2; exit 2 ;;
esac
ls "$OUT"
