#!/bin/bash
# The S-27 fence discriminator (2026-10-18): pym0's 21 images on kit bkit7-fence = bkit7 (module 3005ed52) with
# adapter/module/params fence_before_entry=1, the one change the workaround module d71f207a shares with it. Scored
# against pym0's 3005ed52 board reading (t3-run/pym0/out) and against its QEMU prediction (t3-pred-pym0/out). The batch
# itself refuses a boot whose module did not read the parameter back (virtualvm.check_params); the transcript's
# "@@PARAM fence_before_entry=Y" line is copied into the record.
# Runs after the fix6 appvm chain ends (FIX6APPVM DONE, HOLD or STOP). Usage: tier3-fence-arm.sh <fix6 appvm out> <out>.
# FENCE_ARM / FENCE_RUN / FENCE_TAG / FENCE_DRVPAT replace the arm script, the run dir, the tag and the running-driver
# guard's pattern for the stubbed dry run. Ends by itself.
main() {
  set -u
  local W=${1:?fix6 appvm out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fence
  local LB=/tmp/capstone/vp2-prep/llvm-bin-v6 C=/tmp/capstone/wt-gate/capstone/bug-corpora
  local ARM=${FENCE_ARM:-$S/tier3-appvm-arm.sh} TAG=${FENCE_TAG:-pym0-fence} end=$(( $(date +%s) + 28800 ))
  local R=${FENCE_RUN:-$H/t3-run-fix6/pym0-fence}
  until grep -a -q "^FIX6APPVM DONE\|^FIX6APPVM HOLD\|^FIX6APPVM STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 3005ed52e860c7af ] || { echo "FENCE STOP: the kit's module is not 3005ed52" >> "$O"; return 1; }
  grep -qx "fence_before_entry=1" $K/adapter/module/params || { echo "FENCE STOP: the kit does not set fence_before_entry=1" >> "$O"; return 1; }
  pgrep -f "${FENCE_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "FENCE STOP: a driver is running" >> "$O"; return 1; }
  echo "T3APPVM $TAG BEGIN $(date +%T) (after: $(grep -a '^FIX6APPVM' "$W" | tail -1))" >> "$O"
  EXPECT_IMAGES=$I/pym0.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$ARM" --board-kit $K "$R" \
    $C/cpython/pymalloc-repros/runners/virtual/run-virtual.py --arm virtual-malloc --build $I/pym/cases-OFF \
    --controls-build $I/pym/controls-OFF --llvm-bin $LB >> "$O" 2>&1
  echo "T3APPVM $TAG END rc=$? $(date +%T)" >> "$O"
  echo "FENCE param read-back: $(grep -a -h '^@@PARAM ' "$R"/session/guest/board.txt 2>/dev/null | tr -d '\r' | sort | uniq -c | tr '\n' ' ')" >> "$O"
  echo "FENCE fault lines: $(grep -a -h -o 'domain fault cause=[0-9]* pc=0x[0-9a-f]* address=0x[0-9a-f]*' "$R"/session/guest/board.txt 2>/dev/null | sed 's/pc=0x[0-9a-f]*\([0-9a-f]\{5\}\) /pc-low=\1 /' | sort | uniq -c | tr '\n' ';')" >> "$O"
  python3 "$S/compare-pilot.py" "$H/t3-run/pym0/out" "$R/out" > "$R/compare-vs-3005ed52-board.txt" 2>&1
  echo "FENCE vs pym0 on 3005ed52 (board) rc=$? :: $(tail -1 "$R/compare-vs-3005ed52-board.txt")" >> "$O"
  KIT_DESC="kit bkit7-fence (module 3005ed52e860c7af with fence_before_entry=1, the S-27 fence discriminator), pin off" \
    RESULTS_DIR=2026-10-18-vp2-fence bash "$S/score-arm.sh" "$TAG" "$H/t3-pred-pym0/out" "$R" "$O" >> "$O" 2>&1
  echo "FENCE DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
