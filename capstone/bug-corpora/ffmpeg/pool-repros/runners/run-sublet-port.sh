#!/usr/bin/env bash
# The corpus's cases, each case.c unchanged, against the Sublet port of FFmpeg's OWN pools, in
# the FFmpeg app port's Capstone domain. The corpus's other protected arm (the buffer-pool port's
# probe cases 36-38) runs against that port's substitute allocator; here FFmpeg's buffer.c is
# the only allocator a case talks to.
#
#   run-sublet-port.sh <poolsublet|poolstock> [rounds]
#
# Prerequisite: ports/ffmpeg/app/host/build-domain.sh with FFAPP_HEAP=sublet,
# FFAPP_POOL=sublet (or stock) and FFAPP_CORPUS_DIR=<this corpus>. One boot per case and round:
# the fix's image first, the defect's last, because on poolsublet the defect is predicted to fault
# and a fault ends the emulator. Predictions: sublet-port-expect.txt; verdict:
# sublet-port-verdict.py (a fault is named by its case.c line). Environment as run-safety.sh.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
APP=$(cd -- "$ROOT/../../../ports/ffmpeg/app" && pwd)
ARM=${1:?arm: poolsublet or poolstock}; ROUNDS=${2:-1}
case $ARM in poolsublet|poolstock) ;; *) echo "arm must be poolsublet or poolstock" >&2; exit 2 ;; esac
export FFAPP_SAFETY_EXPECT=$HERE/sublet-port-expect.txt FFAPP_SAFETY_VERDICT=$HERE/sublet-port-verdict.py
status=0
for r in $(seq 1 "$ROUNDS"); do
  for dir in "$ROOT"/[0-9][0-9]_*; do
    c=$(basename "$dir"); c=$((10#${c%%_*}))
    echo "=== $ARM round $r case $c"
    FFAPP_POOL=${ARM#pool} bash "$APP/host/run-safety.sh" "$ARM" $((41 + 2 * c)) $((40 + 2 * c)) || status=1
  done
done
exit $status
