#!/usr/bin/env bash
# Boot Capstone QEMU once and run the resumable-yield probe.
#
# The success markers are ALL THREE of message 1, message 2 and the pass line.
# Message 2 alone is what separates a resume from a restart, so a run that only
# prints message 1 repeatedly must not be read as a pass.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-yield-probe}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-musl-yield-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/yield_probe.dom" "$SHARE_DIR/yield_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/yield_probe.dom" \
  OUT_HOST="$SHARE_DIR/yield_probe.user" \
  bash "$SCRIPT_DIR/build-yield-probe.sh"

# TIMEOUT_MULTIPLIER defaults to 8, not the 4 this started with, because the
# login timeout is 120 * multiplier and boot-to-login on this host measured
# either side of 8 minutes: at multiplier 4 two of three runs came back
# `__CAPSTONE_INFRA_FLAKE__ phase=boot-login` and one succeeded. That is boot
# latency under TCG, not a domain failure -- but a flake indistinguishable from
# a stall costs a whole run to diagnose, so buy the margin.
#
# The boot control and the probe are ONE guest command, not two.
# run-domain-smoke.py checks EVERY --success-marker against EVERY
# --guest-command, so passing them as two commands makes the first one fail for
# missing the second one's markers -- and it then never runs the probe at all.
# Measured that way once; the control is only useful if it shares the command.
#
# It is still a control: a boot that never reaches a shell prints neither
# marker, so its absence voids the probe result rather than looking like a
# probe failure. That distinction is worth keeping while the QEMU core suites
# are RED (ISSUES.md).
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/yield_probe.user /tmp/yield_probe.user && chmod 0755 /tmp/yield_probe.user && /tmp/yield_probe.user /mnt/host/yield_probe.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'yield-probe: round 1 before yield' \
  --success-marker 'yield-probe: round 2 AFTER RESUME, stack intact' \
  --success-marker '__CAPSTONE_YIELD_PROBE_PASSED__'

echo "run-yield-probe.sh completed. Full serial log: $LOG_FILE"
