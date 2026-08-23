#!/usr/bin/env bash
set -euo pipefail

# Revocation enforcement test matrix, cases 2 (memory-stored) and 3 (stc/ldc).
# Both run in a single guest boot.
#
# STATUS (2026-07-03): revocation is END-TO-END. The recording fix (QEMU submodule
# 8b6a47f322, cap_rev_tree_revoke) makes revoke BITE: in round 2 the borrower's
# cached cap reloads UNTAGGED and the use-after-revoke store no longer lands. The
# resulting domain fault is now CLEANLY DELIVERED (Step B): the monitor's
# swap_cpmp/handle_exception recognise the unrecoverable fault and terminate the
# domain via fault_return_from_domain(), returning the sentinel 0x0FA017ED to the
# lender (visible as "round 2 returned 0xfa017ed") instead of spinning in
# capstone_error(). The lender then observes the store did not land and exits
# cleanly. This probe is a green diagnostic; it is not part of any pass/fail gate.
# See design/domain-fault-delivery-proposal.md (Step B),
# design/revocation-enforcement-proposal.md, and
# history/03-07-2026_00-00-06_revocation-70-verify-still-dormant.md.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/infra-retry.sh"

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

capstone_retry_infra_flake \
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  "${CMDS[@]}" \
  --success-marker "revoke-matrix-probe: region revoked" \
  --success-marker "revoke-matrix-probe: use-after-revoke did not update lender view"

echo "run-revoke-matrix-probe.sh completed. Full serial log: $LOG_FILE"
echo "MATRIX RESULT (cases 2,3): revoke succeeds; use-after-revoke is CAUGHT (store dropped)"
echo "and the domain is cleanly terminated -- the monitor returns fault sentinel"
echo "0x0FA017ED to the lender (round 2 retval) instead of spinning."
