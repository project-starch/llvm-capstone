#!/usr/bin/env bash
# Run a silicon-ladder rung under QEMU through the *board* controller.
#
#   usage: run-ladder-perf-qemu.sh <rung> [rung ...]
#
# WHY THIS EXISTS, given run-ladder-qemu.sh already runs the rungs on QEMU.
# That runner uses the stock capstone-test `call_dom` path, where the domain's
# `res` argument is a pointer to a SINGLE unsigned on the monitor's stack --
# there is no shared region at all. Rungs that only write res[0] are fine with
# it, but the v3 shared-region diagnostic reads and writes a whole 4 KiB region,
# and its entire subject is iterated access through the region capability. Under
# the call_dom path that is not merely untested, it is unsafe (res[3..] would
# land in the monitor's stack frame).
#
# So this runner boots QEMU and executes the SAME freestanding controller the
# board runs -- rtl-smoke/ladder_perf_ctl.c: create domain, create + map a 4 KiB
# region, share it (the share IS the entry), read back res[0..2] and the debug
# slots. Exact parity with run_ladder_perf_fpga.py, minus the hardware. A rung
# that passes here and fails on the board is therefore a silicon divergence and
# not a harness difference, which is the whole claim these diagnostics rest on.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

[[ $# -ge 1 ]] || { echo "usage: $0 <rung> [rung ...]" >&2; exit 1; }
RUNGS=("$@")

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${OUT_DIR:-$TMP_ROOT/ladder-fpga}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-ladder-perf.log}
mkdir -p "$TMP_ROOT" "$SHARE_DIR" "$OUT_DIR"

# Build controller + domains + native oracles (same script the board driver uses).
bash "$SCRIPT_DIR/build-ladder-fpga.sh" "${RUNGS[@]}"

cp -f "$OUT_DIR/ladder_perf_ctl" "$SHARE_DIR/"

# One guest command for all rungs: run-domain-smoke.py requires EVERY
# --success-marker in EVERY --guest-command's output, so separate commands per
# rung would cross-check each other's markers and always fail.
GUEST_CMD="cp /mnt/host/ladder_perf_ctl /tmp/lpc && chmod 0755 /tmp/lpc"
MARKERS=()
for R in "${RUNGS[@]}"; do
  cp -f "$OUT_DIR/${R}.dom" "$SHARE_DIR/"
  ORACLE=$(cat "$OUT_DIR/${R}.oracle")
  echo "oracle: $R = $ORACLE"
  GUEST_CMD="$GUEST_CMD; /tmp/lpc $R /mnt/host/${R}.dom"
  # The controller prints: "ladder-perf: RESULT <name> retval=<n> cycles=<n> ran=<n>"
  MARKERS+=(--success-marker "RESULT $R retval=$ORACLE")
done

python3 "$SCRIPT_DIR/../runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "$GUEST_CMD" "${MARKERS[@]}"

echo "run-ladder-perf-qemu.sh: ${RUNGS[*]} matched their oracles. Log: $LOG_FILE"
