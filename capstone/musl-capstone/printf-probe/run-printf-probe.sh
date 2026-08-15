#!/usr/bin/env bash
# Boot Capstone QEMU once and run the printf-probe domain.
#
# Structure copied from run-file-probe.sh, including the quiet console and the
# infra-flake retry; see that file for why each is there.
#
# THE MARKER LIST IS THE TEST. Two of them cannot be produced by the domain
# alone and that is deliberate:
#   PRINTF STDOUT: ...   is printf'd, so it can only appear if musl's FILE
#                        buffering, __stdout_write and SYS_writev all work AND
#                        domain_main's __stdio_exit flushed on the way out.
#                        Everything else here is written through the raw
#                        hostcall, which was already known to work.
#   ..._PASSED__         is printed only when every snprintf case matched the
#                        string glibc produces for the same format, and the
#                        allocator handed out no overlapping block.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-printf}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-printf-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR"/printf_probe*.dom "$SHARE_DIR/printf_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/printf_probe.dom" \
  OUT_HOST="$SHARE_DIR/printf_probe.user" \
  bash "$SCRIPT_DIR/build-printf-probe.sh"

run_smoke() {
  local attempt status
  for attempt in 1 2 3; do
    set +e
    "$@"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      return 0
    fi
    if [[ $status -ne 75 ]]; then
      return "$status"
    fi
    echo "run: infra flake on attempt $attempt (exit 75), retrying" >&2
  done
  echo "run: 3 consecutive infra flakes, giving up" >&2
  return 75
}

# TIMEOUT_MULTIPLIER 12, not the usual 8: this domain is ~310 KB against
# file-probe's 6 KB, because linking malloc brings its 256 KiB static heap, and
# the guest's ELF loader zeroes and copies the whole image before the ioctl under
# TCG. The Lua probe needed 20 for 795 KB; this scales between.
if ! CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300} \
   run_smoke python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro loglevel=1" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-12}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/printf_probe.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/printf_probe.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'PRINTF S2: snprintf cases checked' \
  --success-marker 'PRINTF S3: allocator checked' \
  --success-marker 'PRINTF S4: capability survived memcpy and realloc' \
  --success-marker 'PRINTF STDOUT: 42 ok 1.50 <end>' \
  --success-marker '__CAPSTONE_PRINTF_PROBE_PASSED__' \
  --success-marker '__CAPSTONE_HOSTCALL_HOST_DONE__ status=0'
then
  echo "run-printf-probe.sh FAILED" >&2
  exit 1
fi

echo "run-printf-probe.sh completed. Full serial log: $LOG_FILE"
