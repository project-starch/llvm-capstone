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
#   the open line     -- the helper resolved the path and handed out token 1.
#                        Proves the request layout, since a wrong path offset
#                        would open something else or nothing at all.
#   the DONE line     -- FIVE serviced requests and status OK. The count is
#                        load-bearing: open, write, read, close and the refused
#                        open are five, and the refused read is not, because the
#                        domain's own table answers it. A sixth round would mean
#                        the closed descriptor still reached the helper.
#   the pass line     -- printed only when rounds, status and the byte
#                        comparison all agree.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-file-probe}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-musl-file-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/file_probe.dom" "$SHARE_DIR/file_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/file_probe.dom" \
  OUT_HOST="$SHARE_DIR/file_probe.user" \
  bash "$SCRIPT_DIR/build-file-probe.sh"

# TIMEOUT_MULTIPLIER 8 for the same reason run-yield-probe.sh uses it: boot to
# login on this host measured either side of eight minutes under TCG, and at
# multiplier 4 two runs in three came back as an infra flake indistinguishable
# from a stall.
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/file_probe.user /tmp/file_probe.user && chmod 0755 /tmp/file_probe.user && /tmp/file_probe.user /mnt/host/file_probe.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'file-probe: opened /tmp/musl_file_probe.txt as token 1' \
  --success-marker 'file-probe: DONE, serviced 5 request(s), capstone_main = 0 (OK)' \
  --success-marker '__CAPSTONE_MUSL_FILE_PROBE_PASSED__'

echo "run-file-probe.sh completed. Full serial log: $LOG_FILE"
