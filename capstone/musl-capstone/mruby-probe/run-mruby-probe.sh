#!/usr/bin/env bash
# Boot Capstone QEMU once and run the mruby domain.
#
# Structure copied from run-printf-probe.sh; see that file for the quiet console
# and the infra-flake retry.
#
# EVERY STAGE IS A REQUIRED MARKER, not just the last. S2 says a VM exists, S3
# that bytecode executed, S4 that it computed the right number, S5 that the GC
# tore the whole thing down again. A build that opened a state and stopped would
# otherwise look like a pass.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

# MRUBY_WITH_PARSER=1 selects the variant that evaluates Ruby SOURCE. Separate
# OUT_DIR so the two images never overwrite each other.
if [[ ${MRUBY_WITH_PARSER:-0} == 1 ]]; then
  OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-mruby-parser}
  PARSER_MARKERS=(--success-marker 'MRUBY S6: parsing Ruby source'
                  --success-marker 'MRUBY S7: parsed source produced 400')
else
  OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-mruby}
  PARSER_MARKERS=()
fi
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-mruby-probe.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR"/mruby_probe*.dom "$SHARE_DIR/mruby_probe.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/mruby_probe.dom" \
  OUT_HOST="$SHARE_DIR/mruby_probe.user" \
  bash "$SCRIPT_DIR/build-mruby-probe.sh"

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

# THE .dom IS COPIED TO /tmp FIRST, and that is a measurement, not a tidy-up.
# The loader mmaps the image and memcpys out of it (libcapstone.c:160-186); read
# straight from /mnt/host that source is a 9p mapping, so every 4 KiB page is an
# RPC through emulated virtio under TCG, with cache=none and therefore no
# readahead. Two runs of the 1.35 MB image never finished the copy, at 900 s and
# at 1800 s, while the 313 KB printf probe loads inside 360 s including boot --
# so the cost is not linear in size and the instrument was worth suspecting.
#
# T0/T1/T2 make the phases separable, which one number could not:
#   T1-T0  sequential 9p read of the whole file (cp)
#   T2-T1  mmap + memset + memcpy from LOCAL storage, then the domain itself
# Fast cp and fast load means 9p demand paging was the problem. Fast cp and slow
# load puts it in the loader or TCG. Slow cp means 9p bandwidth, and then none of
# the three fixes on the table help.
if ! CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300} \
   run_smoke python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro loglevel=1" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-30}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; date +T0=%s; cp /mnt/host/mruby_probe.user /tmp/h && chmod 0755 /tmp/h && cp /mnt/host/mruby_probe.dom /tmp/d; date +T1=%s; /tmp/h /tmp/d; date +T2=%s' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'MRUBY S2: mrb_open ok' \
  --success-marker 'MRUBY S3: irep executed' \
  --success-marker 'MRUBY S4: t[19] == 400' \
  --success-marker 'MRUBY S5: state closed' \
  "${PARSER_MARKERS[@]}" \
  --success-marker '__CAPSTONE_MRUBY_PROBE_PASSED__' \
  --success-marker '__CAPSTONE_HOSTCALL_HOST_DONE__ status=0'
then
  echo "run-mruby-probe.sh FAILED" >&2
  exit 1
fi

echo "run-mruby-probe.sh completed. Full serial log: $LOG_FILE"
