#!/usr/bin/env bash
# Build one program's full-configuration objects: its nested allocator's Sublet port plus the
# constructor that brings the port up before main(). The objects go to OUT and are handed to
# tools/run-capstone-domain.py --arm sublet-full as --cc-arg inputs. FFmpeg's (the Sublet port of its
# pools, ports/ffmpeg/sublet) was removed on 2026-10-11 with that port.
#
#   build-objects.sh wireshark SDK OUT
#   build-objects.sh memcached SDK OUT MEMCACHED_SRC            (the allocators port's prepared
#                                                              memcached-1.6.45 source)
#
# SDK must be an application SDK built with CAPSTONE_APPLICATION_HEAP=sublet: every port lends its
# blocks from the Sublet heap (__capstone_sublet_malloc_linear). memcached's needs HEAP_LOG >= 27,
# for its 64 MiB payload and 16 MiB of metadata.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../.." && pwd)
PROG=${1:?wireshark|memcached}; SDK=${2:?SDK}; OUT=${3:?OUT}
CC=$SDK/capstone-cc
mkdir -p "$OUT"
grep -q '^CAPSTONE_APPLICATION_HEAP:STRING=sublet$' "$SDK/CMakeCache.txt" \
  || { echo "build-objects: $SDK is not a Sublet-heap SDK" >&2; exit 2; }
case $PROG in
wireshark)
  WP=$CAP/ports/wireshark/wmem
  for s in "$WP/src/allocators/sublet/chunks.c" "$CAP/ports/wireshark/app/src/tsapp-wmem-chunks.c"; do
    "$CC" -O1 -std=c11 -DWM_DOMAIN -I"$WP/src/shared" -I"$CAP/runtime/include" -c "$s" -o "$OUT/$(basename "${s%.c}").o"
  done
  "$CC" -O0 -std=c11 -DWM_DOMAIN -I"$WP/src/shared" -I"$CAP/runtime/include" -c "$HERE/wireshark.c" -o "$OUT/full-config.o"
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
