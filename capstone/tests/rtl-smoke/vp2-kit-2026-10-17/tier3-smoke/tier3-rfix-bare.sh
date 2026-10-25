#!/bin/bash
# R-58, R-59 and Q-24 on silicon as bare JTAG boots on bitstream 2 (2026-10-26; pre-registered in README.md before the
# boots): the rtl lane's kit capstone/tests/rtl-smoke/r58-r59-q24-bare-2026-10-26/ at llvm-capstone land/rtl-fixes-kit
# 30117b74cc8f, its six images in the PREREG's order (control first; all expected to RETURN), one cold bare boot each, loaded
# at 0x80000000 with the image's own board_rec, run_sup_bare.py from the board kit's driver tree (tier3-r61-bare.sh's path).
# A control that does not read its vector stops the session before the other images. Graded by rfix-grade.py against the
# PREREG.md the session ran (copied from the kit commit). Kit default bkit9-b2-b<board>; output
# <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/rfix-bare.
# Usage: tier3-rfix-bare.sh <board 1|2> <out log> <kit commit>. RF_RUN / RF_BUSY / RF_BOOK / RF_ROOT / RF_GRADE / RF_KIT stub.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} C=${3:?kit commit}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K R BUSY rc bid D=capstone/tests/rtl-smoke/r58-r59-q24-bare-2026-10-26
  case $BRD in
    1) K=${RF_KIT:-$H/t3-smoke/bkit9-b2-b1}; R=${RF_ROOT:-$H/t3-run-b2-dcc6daf6e-b1/rfix-bare}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=${RF_KIT:-$H/t3-smoke/bkit9-b2-b2}; R=${RF_ROOT:-$H/t3-run-b2-dcc6daf6e-b2/rfix-bare}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "RF STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "RF STOP: kit $K" >> "$O"; return 1; }
  local bs drv url fb
  bs=$(sed -n 's/^BOARD_BITSTREAM=//p' $K/board.env); drv=$(sed -n 's/^BOARD_DRIVER_DIR=//p' $K/board.env)
  url=$(sed -n 's/^BOARD_URL_FILE=//p' $K/board.env); url=${url:-$HOME/.claude-kisp/secrets/fpga-console-url}
  fb=$(sed -n 's/^FPGA_BOARD=//p' $K/board.env); fb=${fb:-1}
  [ "$fb" = "$BRD" ] && [ -n "$bs" ] && [ -f "$drv/sup-bare-2026-10-03-ladder/run_sup_bare.py" ] && [ -s "$url" ] \
    || { echo "RF STOP: board.env (board $fb, bitstream ${bs:-none}, driver $drv)" >> "$O"; return 1; }
  git -C /tmp/capstone/wt-gate cat-file -e $C:$D/PREREG.md 2>/dev/null || { echo "RF STOP: $C carries no $D/PREREG.md" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "RF STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R; git -C /tmp/capstone/wt-gate show $C:$D/PREREG.md > $R/PREREG.md
  local img sha rec want_rec n=0
  for spec in "va-runv 274823b4df9e3a4a 8000c000" "va-srevroot 2fbafdfd2eaec3c1 8000d000" "va-runv-r59 f551e9d2492ca6ba 8000c000" \
              "va-runv-r59-low 981fcf633194ee20 8000c000" "q24-deleg 6fd2c0b60fe340ab 80003000" "q24-deleg-on 0e82f2adca300741 80003000"; do
    read -r img sha want_rec <<<"$spec"; n=$((n+1)); mkdir -p $R/$img
    git -C /tmp/capstone/wt-gate show $C:$D/images/$img.bin > $R/$img/$img.bin 2>/dev/null
    rec=$(git -C /tmp/capstone/wt-gate show $C:$D/images/$img.rec 2>/dev/null | tr -d '[:space:]')
    [ "$(sha256sum $R/$img/$img.bin | cut -c1-16)" = $sha ] && [ "$rec" = $want_rec ] \
      || { echo "RF STOP: $img is not $sha with board_rec 0x$want_rec at $C (read rec ${rec:-none})" >> "$O"; return 1; }
    ${RF_BUSY:-$BUSY} > /dev/null || { echo "RF STOP: board $BRD is busy before $img" >> "$O"; return 1; }
    bid=$(${RF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 20 "vp2 bitstream 2 R-58/R-59/Q-24 $img bare (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "RF STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "RF NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "RF $img BEGIN $(date +%T) board $BRD bitstream $bs image $sha rec 0x$rec booking ${bid:-none}" >> "$O"
    # The URL is read into the environment only, never echoed.
    ( cd "$drv" && FPGA_BOARD=$fb FPGA_URL="$(cat "$url")" FPGA_BITSTREAM=$bs.bit SUP_IMG=$R/$img/$img.bin SUP_REC_ADDR=0x$rec \
        SUP_OUT=$R/$img SUP_TIMEOUT=120 timeout 900 ${RF_RUN:-python3 sup-bare-2026-10-03-ladder/run_sup_bare.py} ) > $R/$img/runner.log 2>&1; rc=$?
    [ -n "$bid" ] && ${RF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "RF $img END rc=$rc $(date +%T) :: $(grep -a 'UART:' $R/$img/runner.log | tail -1 | cut -c1-160)" >> "$O"
    [ $img = va-runv ] && { ${RF_GRADE:-python3 $S/rfix-grade.py} $R/PREREG.md $R 2>&1 | grep -q "^RFIX va-runv EXACT" \
      || { echo "RF VOID: the control va-runv did not read its vector; the session stops before the fix images" >> "$O"; ${RF_GRADE:-python3 $S/rfix-grade.py} $R/PREREG.md $R >> "$O" 2>&1; return 1; }; }
  done
  ${RF_GRADE:-python3 $S/rfix-grade.py} $R/PREREG.md $R >> "$O" 2>&1; echo "RF GRADE rc=$?" >> "$O"
  echo "RF board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
