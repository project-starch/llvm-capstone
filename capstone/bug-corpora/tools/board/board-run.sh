#!/bin/bash
# board-run.sh <stage dir> <work dir> <timeout s>: the board kit's recipe for boardvm.BoardBatch.execute (Tier 3).
# A kit is a directory holding this script (or a copy), board.env and adapter/. The recipe:
#   1. builds a manifest: the board controls' base files (BOARD_BASE_MANIFEST, `src|dst|mode` lines) plus every file of
#      the stage under test-domains/vbatch/;
#   2. bakes a private image (bake-manifest.sh, beside this script);
#   3. builds the firmware on it (BOARD_FW_RECIPE with FW_WRAP/FW_MON/FW_IMAGE_DIR and BOARD_FW_DEFINES), and requires
#      the dom_stack gate and the firmware size under BOARD_FW_CAP (the DECLARED-bake cap);
#   4. boots it through the baked-rung driver: b0-stats (the board control) first, then the batch rung, with the
#      watchdog;
#   5. writes <work>/board.txt (the driver's per-rung transcript), <work>/SEQ and <work>/firmware.txt.
# Exit 0 when the board ran (the transcript exists); 2 on a refused input; 1 on a failed step.
# BOARD_STUB=1 prints what would run and runs nothing (exit 3): the dry run of the recipe itself.
main() {
  set -u
  local STAGE=${1:?stage dir} W=${2:?work dir} TO=${3:?timeout s}
  local KIT; KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck disable=SC1091
  source "$KIT/board.env" || { echo "STOP: no board.env in $KIT"; return 2; }
  local v
  for v in BOARD_BITSTREAM BOARD_BASE_MANIFEST BOARD_FW_RECIPE BOARD_FW_WRAP BOARD_FW_MON BOARD_MONITOR_COMMIT \
           BOARD_FW_DEFINES BOARD_FW_CAP BOARD_DRIVER_DIR BOARD_DRIVER_SHA; do
    [ -n "${!v:-}" ] || { echo "STOP: board.env does not set $v"; return 2; }
  done
  [ -d "$W" ] || { echo "STOP: no work dir $W"; return 2; }
  local L=$W/SEQ MAN=$W/manifest.txt IMG=$W/image FW=$W/fw
  echo "BOARD_RUN_START $(date +%T) bitstream $BOARD_BITSTREAM monitor $BOARD_MONITOR_COMMIT" >> "$L"
  # 1. the manifest
  cp "$BOARD_BASE_MANIFEST" "$MAN"
  (cd "$STAGE" && find . -type f | sed 's|^\./||' | sort) | while read -r f; do
    case "$f" in *.sh|capstone-vexec) m=0755;; *) m=0644;; esac
    echo "$STAGE/$f|test-domains/vbatch/$f|$m"
  done >> "$MAN"
  echo "manifest $(wc -l < "$MAN") files, stage $(du -sb "$STAGE" | cut -f1) bytes" >> "$L"
  if [ "${BOARD_STUB:-}" = 1 ]; then
    echo "STUB bake: bash $KIT/bake-manifest.sh $MAN $IMG"
    echo "STUB firmware: FW_WRAP=$BOARD_FW_WRAP FW_MON=$BOARD_FW_MON FW_IMAGE_DIR=$IMG bash $BOARD_FW_RECIPE $FW \"$BOARD_FW_DEFINES\""
    echo "STUB driver: BAKED_CTL=/test-domains/vbatch/ctl.sh BAKED_RUNGS=\"b0-stats vbatch\" BAKED_TIMEOUT=$TO BAKED_OUT=$W/board.txt in $BOARD_DRIVER_DIR"
    return 3
  fi
  # 2. bake
  bash "$KIT/bake-manifest.sh" "$MAN" "$IMG" > "$W/bake.out" 2>&1 \
    || { echo "STOP: bake failed: $(tail -2 "$W/bake.out" | tr '\n' ' ')" >> "$L"; return 1; }
  grep -q BAKE_DONE "$W/bake.out" || { echo "STOP: the bake did not finish" >> "$L"; return 1; }
  # 3. firmware
  [ "$(git -C "$BOARD_FW_MON" rev-parse --short=7 HEAD)" = "${BOARD_MONITOR_COMMIT:0:7}" ] \
    && [ -z "$(git -C "$BOARD_FW_MON" status --porcelain)" ] \
    || { echo "STOP: the monitor worktree is not a clean $BOARD_MONITOR_COMMIT" >> "$L"; return 2; }
  FW_WRAP=$BOARD_FW_WRAP FW_MON=$BOARD_FW_MON FW_IMAGE_DIR=$IMG nice -n 10 bash "$BOARD_FW_RECIPE" "$FW" "$BOARD_FW_DEFINES" \
    > "$W/fw.log" 2>&1 || { echo "STOP: firmware build failed" >> "$L"; return 1; }
  grep -q "^gate: PASS" "$W/fw.log" || { echo "STOP: the dom_stack gate did not pass" >> "$L"; return 1; }
  local sz; sz=$(stat -c %s "$FW/fw_payload.bin")
  [ "$sz" -le "$BOARD_FW_CAP" ] || { echo "STOP: firmware $sz B over the cap $BOARD_FW_CAP" >> "$L"; return 2; }
  { echo "firmware $(sha256sum "$FW/fw_payload.bin" | cut -d' ' -f1)"; echo "kernel $(sha256sum "$IMG/Image" | cut -d' ' -f1)"; } > "$W/firmware.txt"
  echo "firmware $(cut -c10-25 "$W/firmware.txt" | head -1) $sz B" >> "$L"
  # 4. the driver
  [ "$(sha256sum "$BOARD_DRIVER_DIR/fpga_driver/run_baked_rungs_fpga.py" | cut -c1-16)" = "$BOARD_DRIVER_SHA" ] \
    || { echo "STOP: the driver is not the pinned one" >> "$L"; return 2; }
  mkdir -p "$W/oracles"; echo 0 > "$W/oracles/b0-stats.oracle"; echo 0 > "$W/oracles/vbatch.oracle"
  ( cd "$BOARD_DRIVER_DIR" || exit 1
    FPGA_BITSTREAM=$BOARD_BITSTREAM.bit FPGA_URL="$(cat "${BOARD_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")" \
      FPGA_FW=$FW/fw_payload.bin BAKED_CTL=/test-domains/vbatch/ctl.sh BAKED_RUNGS="b0-stats vbatch" LADDER_FPGA_DIR=$W/oracles \
      BAKED_TIMEOUT=$TO BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$W/board.txt \
      timeout $((TO + 1800)) python3 -m fpga_driver.run_baked_rungs_fpga > "$W/driver.log" 2>&1 & runner=$!
    ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh "$W/driver.log" 900 $runner > "$W/watchdog.log" 2>&1 & wd=$!
    wait $runner; rc=$?; sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
    echo "DRIVER rc=$rc $(date +%T)" >> "$L" )
  [ -s "$W/board.txt" ] || { echo "NO DATA: no board transcript" >> "$L"; return 1; }
  echo "BOARD_RUN_DONE $(date +%T)" >> "$L"
}
main "$@"; exit $?
