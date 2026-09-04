#!/usr/bin/env bash
set -euo pipefail

# Serial aggregate gate for the default HostCall proof wrappers. The
# second-PENDING wrappers are targeted diagnostics and intentionally omitted.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/../select.sh"

HOSTCALL_RUNNERS=(
  "$SCRIPT_DIR/run-hostcall-stdout-probe.sh"
  "$SCRIPT_DIR/run-hostcall-filewrite-probe.sh"
  "$SCRIPT_DIR/run-hostcall-fileread-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-open-close-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-handle-write-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-handle-read-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-handle-sync-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-handle-stat-probe.sh"
  "$SCRIPT_DIR/run-hostcall-file-handle-truncate-probe.sh"
  "$SCRIPT_DIR/run-hostcall-path-access-probe.sh"
  "$SCRIPT_DIR/run-hostcall-path-delete-probe.sh"
  "$SCRIPT_DIR/run-hostcall-combined-file-object-probe.sh"
)

capstone_select_banner hostcall
for runner in "${HOSTCALL_RUNNERS[@]}"; do
  capstone_selected "$(basename "$runner" .sh)" || { echo "SKIP  $runner"; continue; }
  echo "run-hostcall-all.sh: running $runner"
  bash "$runner"
done

capstone_select_verify || exit 2
echo "run-hostcall-all.sh: all HostCall proof wrappers completed"
