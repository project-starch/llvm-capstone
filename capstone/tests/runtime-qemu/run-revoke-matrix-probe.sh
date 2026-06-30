#!/usr/bin/env bash
set -euo pipefail

# Revocation enforcement test matrix, cases 2 (memory-stored) and 3 (stc/ldc).
# Both run in a single guest boot.
#
# IMPORTANT: like the M0 probe, this is "green" on the CURRENTLY OBSERVED gap
# (use-after-revoke store lands) because the recording side is dormant pending
# the author. When the recording fix lands, flip the final marker from the
# NO-TRAP-GAP line to the QEMU "Cap mem access" / revoked-capability fault.
# See agent-handoff/design/revocation-enforcement-proposal.md §6/§7.

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
