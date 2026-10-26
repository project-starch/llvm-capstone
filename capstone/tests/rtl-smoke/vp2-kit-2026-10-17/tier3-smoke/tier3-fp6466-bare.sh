#!/bin/bash
# R-64, R-65 and R-66 on silicon as five cold bare JTAG boots on BITSTREAM 3 (2026-10-27; pre-registered in README.md before
# the boots): the rtl lane's kit capstone/tests/rtl-smoke/fp-r64-r66-bare-2026-10-27/ at llvm-capstone land/rtl-r6466-kit
# 5adf7a954946, images fp-of, fp-special, fp-carry, fp-box and fp-edge in its PREREG's order, each loaded at 0x80000000
# with its board_rec by run_sup_bare.py from the board kit's driver tree. This is tier3-fprm-bare.sh with the kit, the
# image list, the bitstream-3 kit and output root, a check that the kit names caplifive_vp2c_e7af48457, and the grader
# changed. Graded by fp6466-grade.py against the PREREG.md the session ran (copied from the kit commit). Kit default
# bkit11-b3-b<board>; output <mnt-hdd>/capstone/t3-run-b3-e7af48457-b<board>/fp6466-bare.
# Usage: tier3-fp6466-bare.sh <board 1|2> <out log> <kit commit>. RF_RUN / RF_BUSY / RF_BOOK / RF_ROOT / RF_GRADE / RF_KIT stub.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} C=${3:?kit commit}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K R BUSY rc bid D=capstone/tests/rtl-smoke/fp-r64-r66-bare-2026-10-27
  case $BRD in
    1) K=${RF_KIT:-$H/t3-smoke/bkit11-b3-b1}; R=${RF_ROOT:-$H/t3-run-b3-e7af48457-b1/fp6466-bare}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=${RF_KIT:-$H/t3-smoke/bkit11-b3-b2}; R=${RF_ROOT:-$H/t3-run-b3-e7af48457-b2/fp6466-bare}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "RF STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "RF STOP: kit $K" >> "$O"; return 1; }
  local bs drv url fb
  bs=$(sed -n 's/^BOARD_BITSTREAM=//p' $K/board.env); drv=$(sed -n 's/^BOARD_DRIVER_DIR=//p' $K/board.env)
  url=$(sed -n 's/^BOARD_URL_FILE=//p' $K/board.env); url=${url:-$HOME/.claude-kisp/secrets/fpga-console-url}
  fb=$(sed -n 's/^FPGA_BOARD=//p' $K/board.env); fb=${fb:-1}
  [ "$fb" = "$BRD" ] && [ "$bs" = caplifive_vp2c_e7af48457 ] && [ -f "$drv/sup-bare-2026-10-03-ladder/run_sup_bare.py" ] && [ -s "$url" ] \
    || { echo "RF STOP: board.env (board $fb, bitstream ${bs:-none}, driver $drv)" >> "$O"; return 1; }
  git -C /tmp/capstone/wt-gate cat-file -e $C:$D/PREREG.md 2>/dev/null || { echo "RF STOP: $C carries no $D/PREREG.md" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "RF STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R; git -C /tmp/capstone/wt-gate show $C:$D/PREREG.md > $R/PREREG.md
  local img sha rec want_rec n=0
  for spec in "fp-of 90f69604fe905881 80003000" "fp-special d9517d17b67ad17f 80004000" "fp-carry e140fc92332fb378 80003000" \
              "fp-box dda547c2c0dc5d85 80003000" "fp-edge fa3e38de7e3e893b 80003000"; do
    read -r img sha want_rec <<<"$spec"; n=$((n+1)); mkdir -p $R/$img
    git -C /tmp/capstone/wt-gate show $C:$D/images/$img.bin > $R/$img/$img.bin 2>/dev/null
    rec=$(git -C /tmp/capstone/wt-gate show $C:$D/images/$img.rec 2>/dev/null | tr -d '[:space:]')
    [ "$(sha256sum $R/$img/$img.bin | cut -c1-16)" = $sha ] && [ "$rec" = $want_rec ] \
      || { echo "RF STOP: $img is not $sha with board_rec 0x$want_rec at $C (read rec ${rec:-none})" >> "$O"; return 1; }
    ${RF_BUSY:-$BUSY} > /dev/null || { echo "RF STOP: board $BRD is busy before $img" >> "$O"; return 1; }
    bid=$(${RF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 20 "vp2 bitstream 3 R-64/R-65/R-66 $img bare (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "RF STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "RF NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "RF $img BEGIN $(date +%T) board $BRD bitstream $bs image $sha rec 0x$rec booking ${bid:-none}" >> "$O"
    # The URL is read into the environment only, never echoed.
    ( cd "$drv" && FPGA_BOARD=$fb FPGA_URL="$(cat "$url")" FPGA_BITSTREAM=$bs.bit SUP_IMG=$R/$img/$img.bin SUP_REC_ADDR=0x$rec \
        SUP_OUT=$R/$img SUP_TIMEOUT=120 timeout 900 ${RF_RUN:-python3 sup-bare-2026-10-03-ladder/run_sup_bare.py} ) > $R/$img/runner.log 2>&1; rc=$?
    [ -n "$bid" ] && ${RF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "RF $img END rc=$rc $(date +%T) :: $(grep -a 'UART:' $R/$img/runner.log | tail -1 | cut -c1-160)" >> "$O"
  done
  ${RF_GRADE:-python3 $S/fp6466-grade.py} $R/PREREG.md $R >> "$O" 2>&1; echo "RF GRADE rc=$?" >> "$O"
  echo "RF board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
