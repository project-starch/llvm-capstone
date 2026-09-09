#!/usr/bin/env bash
set -euo pipefail

# Serial aggregate gate for the baseline and split null_blk regressions.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/../select.sh"

NULLBLK_RUNNERS=(
  "$SCRIPT_DIR/run-nullblk-baseline.sh"
  "$SCRIPT_DIR/run-nullblk-split-io.sh"
  "$SCRIPT_DIR/run-nullblk-split-rmmod.sh"
)

capstone_select_banner nullblk
# A runner's failure must fail the suite. Until 2026-09-08 `bash "$runner"` ran without checking
# its status, so a split-io abort (guest exit 134) still ended in "all null_blk regressions
# completed" and rc 0 -- a gate that could not fire.
failed=0
for runner in "${NULLBLK_RUNNERS[@]}"; do
  capstone_selected "$(basename "$runner" .sh)" || { echo "SKIP  $runner"; continue; }
  echo "run-nullblk-all.sh: running $runner"
  rc=0; bash "$runner" || rc=$?   # `|| rc=$?`, not `; rc=$?`: this script runs under set -e
  if [ "$rc" -eq 0 ]; then echo "PASS  $(basename "$runner" .sh)"; else echo "FAIL  $(basename "$runner" .sh) (rc=$rc)"; failed=$((failed + 1)); fi
done

capstone_select_verify || exit 2
if [ "$failed" -ne 0 ]; then echo "run-nullblk-all.sh: $failed null_blk regression(s) FAILED"; exit 1; fi
echo "run-nullblk-all.sh: all null_blk regressions completed"
