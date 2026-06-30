#!/usr/bin/env bash
set -euo pipefail

# M0 borrow->revoke->use-after-revoke characterization probe.
#
# IMPORTANT: this wrapper is "green" on the *currently observed* behaviour,
# which is a GAP, not the desired safe-fail behaviour. As of 2026-06-29 the
# borrower's cached pointer to a revoked region is still dereferenceable: the
# round-2 store lands and the lender observes the stage-2 sentinel. The markers
# below pin that observed sequence so the probe is a stable regression artifact.
#
# The DESIRED end state (the SQLite marshalling direction's safe-fail claim) is
# that round 2 instead produces a deterministic capability fault. When that is
# achieved, flip the final marker to the fault diagnostic. See
# design/sqlite-marshalling-feasibility.md (M0 result) for the root-cause
# hypothesis (the borrower's SBI-queried mapping appears not to be a tracked
# child of the lender's revocable capability, so the revoke sweep misses it).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-borrow-revoke-uaf-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/borrow_revoke_uaf_probe.user "$SHARE_DIR"/borrow_revoke_uaf_probe.smode

bash "$SCRIPT_DIR/build-borrow-revoke-uaf-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/borrow_revoke_uaf_probe.user /tmp/borrow_revoke_uaf_probe.user && chmod 0755 /tmp/borrow_revoke_uaf_probe.user && /tmp/borrow_revoke_uaf_probe.user /mnt/host/borrow_revoke_uaf_probe.smode" \
  --success-marker "borrow-revoke-uaf-probe: word after round 1 = 0x1111111111111111" \
  --success-marker "borrow-revoke-uaf-probe: region revoked" \
  --success-marker "borrow-revoke-uaf-probe: NO-TRAP-GAP use-after-revoke store landed"

echo "run-borrow-revoke-uaf-probe.sh completed. Full serial log: $LOG_FILE"
echo "M0 RESULT: revoke succeeds but use-after-revoke is NOT trapped (documented gap)."
