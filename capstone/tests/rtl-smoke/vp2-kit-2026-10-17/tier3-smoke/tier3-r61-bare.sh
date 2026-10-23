#!/bin/bash
# R-61 (a stale LDC-refusal flag; dev e63ff602f3fb) as bare JTAG boots on one board on the batch bitstream (2026-10-24;
# pre-registered in README.md before the boots): the RTL lane's kit capstone/tests/rtl-smoke/r61-stale-bare-2026-10-24/
# (inc/ = va-alloc-bare's, tests/vb-trans.S = capstone-ariane dcc6daf6e's, VB_TABLE builds loaded at 0x80000000), taken
# from the commit that carries its PREREG, three images in the PREREG's order, one cold bare boot each, each expected to
# RETURN: r61-control (the full vp2 arm list, the session's known-good control; board_rec 0x80013000), r61-stale2 (a
# refused consuming ldc, then a legal first-touch load: the R-61 reading; 0x80014000), r61-stale (a refused consuming ldc,
# then a load from an unmapped page; 0x80014000). run_sup_bare.py from the kit's driver tree, as tier3-va-alloc-bare.sh;
# graded by r61-grade.py (the PREREG's vectors and VOID rule). A control that does not read its vector exactly makes the
# session VOID and stops it before the other two images.
# Usage: tier3-r61-bare.sh <board 1|2> <out log> <kit commit>. R61_RUN / R61_BUSY / R61_BOOK / R61_ROOT / R61_GRADE stub
# the dry run; R61_KIT names the board kit (default bkit8-batch-b<board>, whose board.env names the flashed bitstream).
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} C=${3:?kit commit}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K R BUSY rc bid D=capstone/tests/rtl-smoke/r61-stale-bare-2026-10-24
  case $BRD in
    1) K=${R61_KIT:-$H/t3-smoke/bkit8-batch-b1}; R=${R61_ROOT:-$H/t3-run-batch-c721c1a2a-b1/r61-bare}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=${R61_KIT:-$H/t3-smoke/bkit8-batch-b2}; R=${R61_ROOT:-$H/t3-run-batch-c721c1a2a-b2/r61-bare}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "R61 STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "R61 STOP: kit $K" >> "$O"; return 1; }
  local bs drv url fb
  bs=$(sed -n 's/^BOARD_BITSTREAM=//p' $K/board.env); drv=$(sed -n 's/^BOARD_DRIVER_DIR=//p' $K/board.env)
  url=$(sed -n 's/^BOARD_URL_FILE=//p' $K/board.env); url=${url:-$HOME/.claude-kisp/secrets/fpga-console-url}
  fb=$(sed -n 's/^FPGA_BOARD=//p' $K/board.env); fb=${fb:-1}
  [ "$fb" = "$BRD" ] && [ -n "$bs" ] && [ -f "$drv/sup-bare-2026-10-03-ladder/run_sup_bare.py" ] && [ -s "$url" ] \
    || { echo "R61 STOP: board.env (board $fb, bitstream ${bs:-none}, driver $drv)" >> "$O"; return 1; }
  git -C /tmp/capstone/wt-gate cat-file -e $C:$D/PREREG.md 2>/dev/null || { echo "R61 STOP: $C carries no $D/PREREG.md" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "R61 STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R; git -C /tmp/capstone/wt-gate show $C:$D/PREREG.md > $R/PREREG.md
  local img sha rec want_rec n=0
  for spec in "r61-control c710e0d5156c2c0d 80013000" "r61-stale2 993876a985b934b9 80014000" "r61-stale f4d9ea5b47bd8b40 80014000"; do
    read -r img sha want_rec <<<"$spec"; n=$((n+1)); mkdir -p $R/$img
    git -C /tmp/capstone/wt-gate show $C:$D/images/$img.bin > $R/$img/$img.bin 2>/dev/null
    rec=$(git -C /tmp/capstone/wt-gate show $C:$D/images/$img.rec 2>/dev/null | tr -d '[:space:]')
    [ "$(sha256sum $R/$img/$img.bin | cut -c1-16)" = $sha ] && [ "$rec" = $want_rec ] \
      || { echo "R61 STOP: $img is not $sha with board_rec 0x$want_rec at $C (read rec ${rec:-none})" >> "$O"; return 1; }
    ${R61_BUSY:-$BUSY} > /dev/null || { echo "R61 STOP: board $BRD is busy before $img" >> "$O"; return 1; }
    bid=$(${R61_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 20 "vp2 R-61 $img bare (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "R61 STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "R61 NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "R61 $img BEGIN $(date +%T) board $BRD bitstream $bs image $sha rec 0x$rec booking ${bid:-none}" >> "$O"
    # The URL is read into the environment only, never echoed.
    ( cd "$drv" && FPGA_BOARD=$fb FPGA_URL="$(cat "$url")" FPGA_BITSTREAM=$bs.bit SUP_IMG=$R/$img/$img.bin SUP_REC_ADDR=0x$rec \
        SUP_OUT=$R/$img SUP_TIMEOUT=120 timeout 900 ${R61_RUN:-python3 sup-bare-2026-10-03-ladder/run_sup_bare.py} ) > $R/$img/runner.log 2>&1; rc=$?
    [ -n "$bid" ] && ${R61_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "R61 $img END rc=$rc $(date +%T) :: $(grep -a 'UART:' $R/$img/runner.log | tail -1 | cut -c1-160)" >> "$O"
    ${R61_GRADE:-python3 $S/r61-grade.py} $img $R/$img/uart.txt >> "$O" 2>&1; rc=$?
    echo "R61 $img GRADE rc=$rc" >> "$O"
    [ $img = r61-control ] && [ $rc != 0 ] && { echo "R61 VOID: the control did not read its vector; the session stops before the R-61 images" >> "$O"; return 1; }
  done
  echo "R61 board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
