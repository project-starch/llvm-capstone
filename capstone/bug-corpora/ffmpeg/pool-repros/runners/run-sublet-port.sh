#!/usr/bin/env bash
# The corpus's cases, each case.c unchanged, against the Sublet port of FFmpeg's OWN pools, in
# the FFmpeg app port's Capstone domain. The corpus's other protected arm (the buffer-pool port's
# probe cases 36-38) runs FFmpeg's buffer.c with the payloads served by that port's own allocator
# and its leases; here the pools themselves are ported, and a return is the pool's own revoke.
#
#   FFAPP_CORPUS_OUT=<new dir> run-sublet-port.sh <poolsublet|poolstock> [rounds]
#
# Prerequisite: ports/ffmpeg/app/host/build-domain.sh with FFAPP_HEAP=sublet,
# FFAPP_POOL=sublet (or stock) and FFAPP_CORPUS_DIR=<this corpus>, and a running VM
# (CAPSTONE_VM_STATE, as run-safety.sh). Each case and round is one run-safety.sh call into its
# own result directory under FFAPP_CORPUS_OUT: the fix's image first, then the defect's. A fault
# ends only its own application, so the order is kept for comparability, not survival.
# Predictions: sublet-port-expect.txt; verdict: sublet-port-verdict.py (a fault is named by its
# case.c line).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
APP=$(cd -- "$ROOT/../../../ports/ffmpeg/app" && pwd)
ARM=${1:?arm: poolsublet or poolstock}; ROUNDS=${2:-1}
case $ARM in poolsublet|poolstock) ;; *) echo "arm must be poolsublet or poolstock" >&2; exit 2 ;; esac
OUT=${FFAPP_CORPUS_OUT:?FFAPP_CORPUS_OUT: a new result directory}
[ ! -e "$OUT" ] || { echo "$OUT exists; results are never overwritten" >&2; exit 2; }
mkdir -p "$OUT"
export FFAPP_SAFETY_EXPECT=$HERE/sublet-port-expect.txt FFAPP_SAFETY_VERDICT=$HERE/sublet-port-verdict.py
status=0
for r in $(seq 1 "$ROUNDS"); do
  for dir in "$ROOT"/[0-9][0-9]_*; do
    c=$(basename "$dir"); c=$((10#${c%%_*}))
    echo "=== $ARM round $r case $c"
    FFAPP_SAFETY_OUT=$OUT/$ARM-r$r-case$c \
      bash "$APP/host/run-safety.sh" "$ARM" $((41 + 2 * c)) $((40 + 2 * c)) || status=1
  done
done
exit $status
