#!/usr/bin/env bash
# Boot Capstone QEMU once and run musl's write(2) through the hostcall.
#
# FOUR MARKERS, and each one rules out a different way of passing by accident:
#
#   the boot control  -- a boot that never reaches a shell prints nothing, so
#                        its absence VOIDS the result instead of looking like a
#                        probe failure. It shares the guest command with the
#                        probe on purpose: run-domain-smoke.py checks every
#                        marker against every command, so two commands would
#                        make the first fail for missing the second's markers.
#   the payload       -- the bytes musl wrote, printed by the host after it
#                        compared them. Proves the transfer, not just the call.
#   the DONE line     -- one serviced request and capstone_main = 0 (OK), which
#                        is where the negative control lands: a domain whose
#                        bad-fd write succeeded, or whose errno was lost at the
#                        boundary, reaches DONE with a different status and this
#                        marker does not match.
#   the pass line     -- printed only when rounds, status and the byte
#                        comparison all agree.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-write-probe}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-musl-write-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/write_probe.dom" "$SHARE_DIR/write_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/write_probe.dom" \
  OUT_HOST="$SHARE_DIR/write_probe.user" \
  bash "$SCRIPT_DIR/build-write-probe.sh"

# TIMEOUT_MULTIPLIER 8 for the same reason run-yield-probe.sh uses it: boot to
# login on this host measured either side of eight minutes under TCG, and at
# multiplier 4 two runs in three came back as an infra flake indistinguishable
# from a stall.
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/write_probe.user /tmp/write_probe.user && chmod 0755 /tmp/write_probe.user && /tmp/write_probe.user /mnt/host/write_probe.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'musl write through hostcall v0' \
  --success-marker 'write-probe: DONE, serviced 1 request(s), capstone_main = 0 (OK)' \
  --success-marker '__CAPSTONE_MUSL_WRITE_PROBE_PASSED__'

echo "run-write-probe.sh completed. Full serial log: $LOG_FILE"
