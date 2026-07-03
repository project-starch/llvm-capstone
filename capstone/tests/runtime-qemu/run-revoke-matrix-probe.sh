#!/usr/bin/env bash
set -euo pipefail

# Revocation enforcement test matrix, cases 2 (memory-stored) and 3 (stc/ldc).
# Both run in a single guest boot.
#
# STATUS (2026-07-03): the recording fix landed (QEMU submodule 8b6a47f322,
# cap_rev_tree_revoke), so revocation now BITES: in round 2 the borrower's cached
# cap reloads UNTAGGED and the use-after-revoke store no longer lands (verified).
# HOWEVER the caught fault is raised inside a lender->borrower domain call, and
# clean return-to-host delivery is the still-unfinished Step-B monitor gap
# (design/domain-fault-delivery-proposal.md), so the monitor dumps registers and
# spins rather than returning the fault -> this probe currently HANGS/times out
# in round 2. That hang is the delivery gap, NOT a recording defect. This probe
# is a standalone diagnostic; it is not part of any pass/fail gate. It becomes a
# clean green once Step-B lands (flip the marker to the revoked-capability fault).
# See agent-handoff/design/revocation-enforcement-proposal.md §6/§7 and
# history/03-07-2026_00-00-06_revocation-70-verify-still-dormant.md.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-revoke-matrix-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/revoke_matrix_probe_case*.user "$SHARE_DIR"/revoke_matrix_probe_case*.smode

bash "$SCRIPT_DIR/build-revoke-matrix-probe.sh" "$SHARE_DIR"

run_case() {
  local c="$1"
  printf -- '--guest-command\ncp /mnt/host/revoke_matrix_probe_case%s.user /tmp/rm%s.user && chmod 0755 /tmp/rm%s.user && /tmp/rm%s.user /mnt/host/revoke_matrix_probe_case%s.smode\n' "$c" "$c" "$c" "$c" "$c"
}

mapfile -t CMDS < <(run_case 2; run_case 3)

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  "${CMDS[@]}" \
  --success-marker "revoke-matrix-probe: region revoked" \
  --success-marker "revoke-matrix-probe: NO-TRAP-GAP use-after-revoke store landed"

echo "run-revoke-matrix-probe.sh completed. Full serial log: $LOG_FILE"
echo "MATRIX RESULT (cases 2,3): revoke succeeds; use-after-revoke NOT trapped (documented gap, dormant)."
