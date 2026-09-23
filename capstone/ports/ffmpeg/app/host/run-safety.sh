#!/usr/bin/env bash
# Run safety fixtures of one heap arm in ONE boot and judge them against the pre-registered
# predictions.     usage: run-safety.sh <level0|shrink|sublet|pool0|pool2> <fixture>...
#
# A capability fault inside a domain ENDS THE EMULATOR (capstone-qemu cpu_helper.c prints
# "domain halted by capability fault" and exits), so a boot can hold at most one fixture
# expected to fault, and it must come LAST: everything after it is never run. This script
# refuses an order that would put a predicted fault anywhere else.
#
# The smoke runner's own exit status is not the verdict -- a boot that ends in a fault makes
# it fail by construction. The verdict is host/safety-verdict.py over each fixture's own
# section of the log; a fixture with no section (the boot never reached it) is an error.
#
# Prerequisites: build-domain.sh with FFAPP_HEAP=<arm> (the level0 arm is the default build).
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

ARM=${1:?arm: level0, shrink, sublet, pool0 or pool2}; shift
[ $# -gt 0 ] || { echo "no fixtures given" >&2; exit 2; }
case $ARM in level0) DOM_DIR=domain ;; shrink) DOM_DIR=domain-shrink ;; sublet) DOM_DIR=domain-sublet ;;
  pool0|pool2) DOM_DIR=domain-sublet-$ARM ;;   # FFAPP_HEAP=sublet FFAPP_POOL=0|2
  *) echo "arm must be level0, shrink, sublet, pool0 or pool2" >&2; exit 2 ;; esac
WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
DOM="$WORK/$DOM_DIR"
SHARE="$WORK/share-safety"
LOG=${LOG_FILE:-$WORK/safety-$ARM-$(printf '%s' "$*" | tr ' ' '-').log}
# Never overwrite an earlier attempt: a stalled boot's log is the evidence that it stalled
# before any image loaded (audit, 2026-09-23: two retries overwrote theirs).
if [ -e "$LOG" ]; then n=2; while [ -e "${LOG%.log}.try$n.log" ]; do n=$((n + 1)); done; LOG=${LOG%.log}.try$n.log; fi
EXPECT="$SCRIPT_DIR/safety-expect.txt"

# A predicted fault anywhere but last would silently cost every fixture after it.
n=$#; i=0
for fx in "$@"; do
  i=$((i + 1))
  if [ "$i" -lt "$n" ] && grep -qE "^$ARM +$fx +FAULT" "$EXPECT"; then
    echo "fixture $fx is predicted to FAULT on $ARM and is not last; the rest would never run" >&2
    exit 2
  fi
done

[ -f "$DOM/ffapp.user" ] || { echo "missing $DOM/ffapp.user; run FFAPP_HEAP=$ARM build-domain.sh" >&2; exit 2; }
rm -rf "$SHARE"; mkdir -p "$SHARE"
cp "$DOM/ffapp.user" "$SHARE/"
# images load from the guest's /tmp, not through 9p-backed mmap faults (see run-qemu.sh)
RUN="dmesg -n 7; cp /mnt/host/ffapp.user /tmp/ffapp.user && chmod 0755 /tmp/ffapp.user && cp /mnt/host/*.dom /tmp/"
for fx in "$@"; do
  [ -f "$DOM/ffapp_fx$fx.dom" ] || { echo "missing $DOM/ffapp_fx$fx.dom" >&2; exit 2; }
  cp "$DOM/ffapp_fx$fx.dom" "$SHARE/"
  RUN="$RUN; echo __FFAPP_BEGIN_FX${fx}__; /tmp/ffapp.user /tmp/ffapp_fx$fx.dom 0; echo __FFAPP_END_FX${fx}__"
done
( cd "$SHARE" && sha256sum ffapp.user ffapp_fx*.dom ) > "$LOG.sha256"

smoke=(python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
       --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}"
       --guest-command "echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; $RUN"
       --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__')
[ -n "${FFAPP_BUILDROOT_DIR:-}" ] && smoke+=(--buildroot-dir "$FFAPP_BUILDROOT_DIR")
[ -n "${FFAPP_QEMU_BINARY:-}" ] && smoke+=(--qemu-binary "$FFAPP_QEMU_BINARY")
if [ "${CAPSTONE_QEMU_LOCK_HELD:-0}" = 1 ]; then
  "${smoke[@]}" > "$LOG.smoke" 2>&1
else
  mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
  CAPSTONE_QEMU_LOCK_HELD=1 flock -x -w "${FFAPP_LOCK_WAIT:-3600}" "$CAPSTONE_QEMU_LOCK" "${smoke[@]}" > "$LOG.smoke" 2>&1
fi
echo "run-domain-smoke exit status $? (not the verdict); serial log $LOG"
python3 "$SCRIPT_DIR/safety-verdict.py" "$LOG" "$EXPECT" "$ARM" "$@"
