#!/usr/bin/env bash
# Build one program's full-configuration objects: its nested allocator's Sublet port plus the
# constructor that brings the port up before main(). The objects go to OUT and are handed to
# tools/run-capstone-domain.py --arm sublet-full as --cc-arg inputs. FFmpeg's (the Sublet port of its
# pools, ports/ffmpeg/sublet) was removed on 2026-10-11 with that port.
#
# SDK must be an application SDK built with CAPSTONE_APPLICATION_HEAP=sublet: every port lends its
# blocks from the Sublet heap (__capstone_sublet_malloc_linear). memcached's full configuration
# (its allocators port's ledger on the Sublet authority) was removed on 2026-10-11 with that
# authority; its recorded sublet-full verdicts stay in the plain corpora's case files.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../.." && pwd)
PROG=${1:?PROG}; SDK=${2:?SDK}; OUT=${3:?OUT}
CC=$SDK/capstone-cc
mkdir -p "$OUT"
grep -q '^CAPSTONE_APPLICATION_HEAP:STRING=sublet$' "$SDK/CMakeCache.txt" \
  || { echo "build-objects: $SDK is not a Sublet-heap SDK" >&2; exit 2; }
case $PROG in
*) echo "build-objects: no program $PROG" >&2; exit 2 ;;
esac
ls "$OUT"
