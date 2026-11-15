#!/usr/bin/env bash
# run.sh KIT OLD NEW WORK: every mode of two builds of build.sh, OLD with patch 0006 before the layout
# fix and NEW with it, in one trap-route boot (runtime/virtual/trap_platform.py), then check.py's
# verdict on each step against the table of docs/plans/2026-11-15-memcached-port-layout-fix.md.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../../.." && pwd)
[[ $# == 4 ]] || { echo 'usage: run.sh KIT OLD NEW WORK' >&2; exit 2; }
KIT=$1 OLD=$2 NEW=$3 WORK=$4

args=(--file old.dom="$OLD/sublet-layout.dom" --file new.dom="$NEW/sublet-layout.dom")
for mode in scan stale bounds move; do
  args+=(--step "old-$mode=./capstone-vexec ./old.dom $mode")
done
for mode in scan stale bounds move move-prealloc; do
  args+=(--step "new-$mode=./capstone-vexec ./new.dom $mode")
done
python3 "$REPO/capstone/runtime/virtual/trap_platform.py" run --kit "$KIT" --work "$WORK" --step-timeout 300 "${args[@]}"
python3 "$HERE/check.py" "$WORK/results.json"
