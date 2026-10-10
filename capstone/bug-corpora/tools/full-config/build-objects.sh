#!/usr/bin/env bash
# Build one program's full-configuration objects: its nested allocator's Sublet port plus the
# constructor that brings the port up before main(). The objects go to OUT and are handed to
# tools/run-capstone-domain.py --arm sublet-full as --cc-arg inputs.
#
#   build-objects.sh ffmpeg    SDK OUT FFMPEG_SRC FFMPEG_BUILD   (the app port's --sublet source and
#                                                              its ffmpeg-build-poolsublet tree)
#   build-objects.sh memcached SDK OUT MEMCACHED_SRC            (the allocators port's prepared
#                                                              memcached-1.6.45 source)
#
# SDK must be an application SDK built with CAPSTONE_APPLICATION_HEAP=sublet: every port lends its
# blocks from the Sublet heap (__capstone_sublet_malloc_linear). memcached's needs HEAP_LOG >= 27,
# for its 64 MiB payload and 16 MiB of metadata.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../.." && pwd)
PROG=${1:?ffmpeg|memcached}; SDK=${2:?SDK}; OUT=${3:?OUT}
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
memcached)
  MS=${4:?MEMCACHED_SRC}; MP=$CAP/ports/memcached/allocators
  F=(-DNDEBUG -DCHUNK_ALIGN_BYTES=16 -DMCP_DOMAIN -I"$MP/src/shared" -I"$MS" -I"$MP/../adapted" -I"$CAP/runtime/include")
  for s in "$MP/src/shared/leases.c" "$MP/src/shared/metadata.c" "$MP/src/allocators/sublet/authority.c"; do
    "$CC" -O1 "${F[@]}" -c "$s" -o "$OUT/$(basename "${s%.c}").o"
  done
  "$CC" -O0 "${F[@]}" -c "$HERE/memcached.c" -o "$OUT/full-config.o"
  ;;
*) echo "build-objects: no program $PROG" >&2; exit 2 ;;
esac
ls "$OUT"
