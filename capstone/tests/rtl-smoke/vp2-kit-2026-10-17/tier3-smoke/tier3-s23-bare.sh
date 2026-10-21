#!/bin/bash
# The RTL lane's s23-cpmp-rearm as a bare JTAG boot on one board (2026-10-21; pre-registered in README.md before the boot):
# the image from dev 0c215bf0983f (capstone/tests/rtl-smoke/s23-bare-2026-10-21/, sha256 bcefd969e128113f..., board_rec
# 0x80004040), run by sup-bare-2026-10-03-ladder/run_sup_bare.py from the kit's own driver tree (BOARD_DRIVER_DIR) with
# the kit's board, URL file and resident bitstream, then graded by s23-grade.py (PREREG.md's rule). No monitor, no Linux.
# Usage: tier3-s23-bare.sh <board 1|2> <out log> [pre|batch]. S23_RUN / S23_BUSY / S23_BOOK / S23_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} EXP=${3:-pre}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K R BUSY rc bid IMGSHA=bcefd969e128113f0c58e18571023f1269efce816d0587a85e987577b9e3c0fe
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; R=${S23_ROOT:-$H/t3-run-treeA/s23-bare}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; R=${S23_ROOT:-$H/t3-run-treeB/s23-bare}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "S23 STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "S23 STOP: kit $K" >> "$O"; return 1; }
  local bs drv url fb
  bs=$(sed -n 's/^BOARD_BITSTREAM=//p' $K/board.env); drv=$(sed -n 's/^BOARD_DRIVER_DIR=//p' $K/board.env)
  url=$(sed -n 's/^BOARD_URL_FILE=//p' $K/board.env); url=${url:-$HOME/.claude-kisp/secrets/fpga-console-url}
  fb=$(sed -n 's/^FPGA_BOARD=//p' $K/board.env); fb=${fb:-1}
  [ "$fb" = "$BRD" ] && [ -n "$bs" ] && [ -f "$drv/sup-bare-2026-10-03-ladder/run_sup_bare.py" ] && [ -s "$url" ] || { echo "S23 STOP: board.env (board $fb, bitstream ${bs:-none}, driver $drv)" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "S23 STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  git -C /tmp/capstone/wt-gate show 0c215bf0983f:capstone/tests/rtl-smoke/s23-bare-2026-10-21/images/s23-cpmp-rearm.bin > $R/s23-cpmp-rearm.bin
  [ "$(sha256sum $R/s23-cpmp-rearm.bin | cut -d' ' -f1)" = $IMGSHA ] || { echo "S23 STOP: image sha256" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-gate show 0c215bf0983f:capstone/tests/rtl-smoke/s23-bare-2026-10-21/images/s23-cpmp-rearm.rec | tr -d '[:space:]')" = 80004040 ] || { echo "S23 STOP: board_rec" >> "$O"; return 1; }
  ${S23_BUSY:-$BUSY} > /dev/null || { echo "S23 STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${S23_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 20 "vp2-tier12 s23-cpmp-rearm bare (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "S23 STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "S23 NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "S23 BEGIN $(date +%T) board $BRD bitstream $bs image ${IMGSHA:0:16} driver $drv booking ${bid:-none} expect $EXP" >> "$O"
  # The URL is read into the environment only, never echoed.
  ( cd "$drv" && FPGA_BOARD=$fb FPGA_URL="$(cat "$url")" FPGA_BITSTREAM=$bs.bit SUP_IMG=$R/s23-cpmp-rearm.bin SUP_REC_ADDR=0x80004040 \
      SUP_OUT=$R SUP_TIMEOUT=120 timeout 900 ${S23_RUN:-python3 sup-bare-2026-10-03-ladder/run_sup_bare.py} ) > $R/runner.log 2>&1; rc=$?
  [ -n "$bid" ] && ${S23_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "S23 END rc=$rc $(date +%T) :: $(grep -a 'UART:' $R/runner.log | tail -1 | cut -c1-160)" >> "$O"
  python3 $S/s23-grade.py $R/uart.txt --expect $EXP >> "$O" 2>&1; echo "S23 GRADE rc=$?" >> "$O"
  echo "S23 board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
