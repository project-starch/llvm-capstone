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

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-stdio-probe}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-musl-stdio-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/stdio_probe.dom" "$SHARE_DIR/stdio_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/stdio_probe.dom" \
  OUT_HOST="$SHARE_DIR/stdio_probe.user" \
  bash "$SCRIPT_DIR/build-stdio-probe.sh"

# TIMEOUT_MULTIPLIER 8 for the same reason run-yield-probe.sh uses it: boot to
# login on this host measured either side of eight minutes under TCG, and at
# multiplier 4 two runs in three came back as an infra flake indistinguishable
# from a stall.
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/stdio_probe.user /tmp/stdio_probe.user && chmod 0755 /tmp/stdio_probe.user && /tmp/stdio_probe.user /mnt/host/stdio_probe.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'stdio-probe: opened /tmp/musl_stdio_probe.txt as token 1' \
  --success-marker 'musl printf through hostcall v0' \
  --success-marker 'stdio-probe: DONE, serviced' \
  --success-marker '__CAPSTONE_MUSL_STDIO_PROBE_PASSED__'

echo "run-stdio-probe.sh completed. Full serial log: $LOG_FILE"
