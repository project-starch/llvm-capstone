#!/usr/bin/env bash
# Run one FFmpeg milestone image in Capstone QEMU.     usage: run-qemu.sh [stage 1..5]
#
# QEMU runs are serialized across lanes: this takes $CAPSTONE_QEMU_LOCK itself (the shared
# rootfs.ext2 must never see two guests), unless the caller already holds it.
#
# MARKERS, each ruling out a different way of passing by accident:
#   the boot control   -- a boot that never reaches a shell prints nothing, so its absence
#                         VOIDS the run instead of reading as an FFmpeg failure;
#   "STAGE M<n>"       -- the program itself got that far (printed by ffapp_decode.c);
#   the reached line   -- the host saw capstone_main return exactly the milestone this
#                         image was built for (src/linux-guest/ffapp_host.c).
# For stage 5 the same boot also runs the flipped-input control image, and the frame
# lines from both are then compared against the native reference with
# host/compare-md5.py: the real image must MATCH and the control must NOT.
#
# Prerequisites: build-native.sh (the oracle) and build-domain.sh (the images).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

STAGE=${1:-1}
case $STAGE in 1|2|3|4|5) ;; *) echo "stage must be 1..5" >&2; exit 2;; esac
WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
DOM="$WORK/domain"
SHARE="$WORK/share"
LOG=${LOG_FILE:-$WORK/qemu-m$STAGE.log}
for f in "$DOM/ffapp_m$STAGE.dom" "$DOM/ffapp.user" "$WORK/input.mkv" "$WORK/stock.framemd5"; do
  [ -f "$f" ] || { echo "missing $f; run build-native.sh and build-domain.sh first" >&2; exit 2; }
done
rm -rf "$SHARE"; mkdir -p "$SHARE"
cp "$DOM/ffapp_m$STAGE.dom" "$DOM/ffapp.user" "$WORK/input.mkv" "$WORK/input.flip.mkv" "$SHARE/"

RUN="cp /mnt/host/ffapp.user /tmp/ffapp.user && chmod 0755 /tmp/ffapp.user"
RUN="$RUN && echo __FFAPP_BEGIN_MAIN__ && /tmp/ffapp.user /mnt/host/ffapp_m$STAGE.dom $STAGE && echo __FFAPP_END_MAIN__"
MARKERS=(--success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__'
         --success-marker "STAGE M$STAGE"
         --success-marker "__CAPSTONE_FFAPP_STAGE_REACHED__ $STAGE")
if [ "$STAGE" = 5 ]; then
  cp "$DOM/ffapp_m5flip.dom" "$SHARE/"
  RUN="$RUN; echo __FFAPP_BEGIN_FLIP__; /tmp/ffapp.user /mnt/host/ffapp_m5flip.dom 5; echo __FFAPP_END_FLIP__"
  MARKERS+=(--success-marker '__FFAPP_END_FLIP__')
fi

smoke=(python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
       --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}"
       --guest-command "echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; $RUN" "${MARKERS[@]}")
set +e
if [ "${CAPSTONE_QEMU_LOCK_HELD:-0}" = 1 ]; then
  "${smoke[@]}"
else
  mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
  CAPSTONE_QEMU_LOCK_HELD=1 flock -x -w "${FFAPP_LOCK_WAIT:-3600}" "$CAPSTONE_QEMU_LOCK" "${smoke[@]}"
fi
rc=$?
set -e
echo "run-domain-smoke exit status $rc; serial log $LOG"
[ "$rc" = 0 ] || exit "$rc"

if [ "$STAGE" = 5 ]; then
  sed -n '/__FFAPP_BEGIN_MAIN__/,/__FFAPP_END_MAIN__/p' "$LOG" > "$WORK/domain-m5.out"
  sed -n '/__FFAPP_BEGIN_FLIP__/,/__FFAPP_END_FLIP__/p' "$LOG" > "$WORK/domain-m5flip.out"
  python3 "$SCRIPT_DIR/compare-md5.py" "$WORK/stock.framemd5" "$WORK/domain-m5.out" \
    --control "$WORK/domain-m5flip.out"
fi
