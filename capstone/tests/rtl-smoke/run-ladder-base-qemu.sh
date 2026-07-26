#!/usr/bin/env bash
# QEMU parity leg for the silicon-ladder BASELINE controller.
#
#   usage: run-ladder-base-qemu.sh
#
# Runs the same `ladder_base_ctl` the board will run, in the QEMU guest, and
# gates every rung on `BASE RESULT <rung> retval=<native oracle>`. This exists to
# spend a board boot on a binary already known to compute the right answers: a
# wiring or codegen mistake found here costs a minute, found on the board it costs
# a power-cycle and a lock window.
#
# What it does NOT validate is the cycle counts. QEMU's counter CSRs do not model
# the CVA6 pipeline, so the `cycles=` field here is meaningless and is deliberately
# not gated on -- only the retvals are. The counter probes are still worth running
# because whether a counter READS AT ALL is a counteren question, and a difference
# between QEMU and the board is itself informative.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

LAD="$SCRIPT_DIR/../runtime-qemu/silicon-ladder"
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${OUT_DIR:-$TMP_ROOT/ladder-base}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-ladder-base.log}
mkdir -p "$TMP_ROOT" "$SHARE_DIR" "$OUT_DIR"

bash "$SCRIPT_DIR/build-ladder-base-fpga.sh"
cp -f "$OUT_DIR/ladder_base_ctl" "$SHARE_DIR/"

RUNGS=(matmult_int coremark_matrix rv8_primes beebs_crc32 beebs_insertsort \
       beebs_prime beebs_recursion)

# Each probe is its own invocation so a gated CSR kills only that one; `|| true`
# keeps the guest command going when the shell reports the illegal instruction.
GUEST_CMD="cp /mnt/host/ladder_base_ctl /tmp/lbc && chmod 0755 /tmp/lbc"
for C in cycle time instret; do
  GUEST_CMD="$GUEST_CMD; /tmp/lbc probe $C || true"
done
GUEST_CMD="$GUEST_CMD; /tmp/lbc all"

# Gate on pass 1 only. Pass 2 is the warm repeat, and for a stateful kernel it is
# SUPPOSED to return something else (that is how the harness detects statefulness),
# so requiring the oracle twice would fail the run for working as designed.
MARKERS=()
for R in "${RUNGS[@]}"; do
  cc -O0 -o "$OUT_DIR/${R}_host" "$LAD/${R}_host.c"
  ORACLE=$("$OUT_DIR/${R}_host")
  echo "oracle: $R = $ORACLE"
  MARKERS+=(--success-marker "BASE RESULT $R pass=1 retval=$ORACLE")
done
# The null control has no kernel and no host oracle: it returns a zeroed volatile.
MARKERS+=(--success-marker "BASE RESULT null pass=1 retval=0")
MARKERS+=(--success-marker "BASE DONE all")

python3 "$SCRIPT_DIR/../runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "$GUEST_CMD" "${MARKERS[@]}"

echo "run-ladder-base-qemu.sh: all 7 baseline rungs matched their oracles."
echo "Counter-probe outcomes (board may differ -- counteren is per-platform):"
grep -E "BASE PROBE|Illegal instruction" "$LOG_FILE" || echo "  (none captured)"
