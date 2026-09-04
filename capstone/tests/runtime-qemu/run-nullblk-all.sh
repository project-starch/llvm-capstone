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
for runner in "${NULLBLK_RUNNERS[@]}"; do
  capstone_selected "$(basename "$runner" .sh)" || { echo "SKIP  $runner"; continue; }
  echo "run-nullblk-all.sh: running $runner"
  bash "$runner"
done

capstone_select_verify || exit 2
echo "run-nullblk-all.sh: all null_blk regressions completed"
