#!/bin/bash
# Profile several benchmarks in several arms at once: one guest per (arm, benchmark), up to JOBS
# guests in parallel. Attempt counts do not depend on the host's load (tools/profile.md), so the
# guests share the host freely; only wall-clock time changes.
#
# usage: profile-batch.sh ARMS SIZE OUTROOT ARM... -- BENCH...
#   OUTROOT/ARM/BENCH is each run's OUT (tools/profile.sh); an existing one is skipped, so a batch
#   can be resumed. JOBS (env, default 8) bounds the parallel guests.
set -euo pipefail
ARMS=$(readlink -f "$1") SIZE=$2 OUTROOT=$3; shift 3
KIT=$(cd "$(dirname "$0")/.." && pwd -P)
arms=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do arms+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
benches=("$@")
[ ${#arms[@]} -gt 0 ] && [ ${#benches[@]} -gt 0 ] || { echo "usage: profile-batch.sh ARMS SIZE OUTROOT ARM... -- BENCH..." >&2; exit 2; }
mkdir -p "$OUTROOT"; OUTROOT=$(cd "$OUTROOT" && pwd -P)
jobs=()
for b in "${benches[@]}"; do for a in "${arms[@]}"; do
  [ -e "$OUTROOT/$a/$b" ] && { echo "skip $a $b (exists)"; continue; }
  jobs+=("$a $b")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P "${JOBS:-8}" -L 1 bash -c '
  a=$0 b=$1; out='"$OUTROOT"'/$a/$b; mkdir -p "$(dirname "$out")"
  start=$(date +%s)
  bash '"$KIT"'/tools/profile.sh '"$ARMS"' "$a" '"$SIZE"' "$b" "$out" > "$out.log" 2>&1; rc=$?
  v=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"verdict\"])" "$out/result.json" 2>/dev/null || echo NONE)
  t=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"total_attempts\"])" "$out/profile.json" 2>/dev/null || echo -)
  echo "$a $b rc=$rc verdict=$v attempts=$t wall=$(( $(date +%s) - start ))s"
' | tee -a "$OUTROOT/batch.log"
