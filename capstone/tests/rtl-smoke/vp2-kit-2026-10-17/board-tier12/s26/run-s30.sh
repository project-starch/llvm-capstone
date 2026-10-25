#!/bin/bash
# S-30's matched pair on tree A, board 1 (2026-10-19 evening, pre-registered in tier3-smoke/README.md before the boots):
# two cold boots on the S-30 image (s30-prep.sh: the pair's stage with vmrun.sh's s30-read case), firmwares that differ
# only in the wrapper: s30-old (aab63d5, fix1's) and s30-new (9bee361, S-30's held carve); monitor 5b5e86b in both.
# Rungs b0-stats vm-contract b0-hello s30-read (the read last). Judged by s30-judge.py. Usage: run-s30.sh [arm...]
# (default s30-old s30-new). Ends by itself. S26R_* stub for the dry run, as run-s26.sh.
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
main() {
  set -u
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local X=/tmp/capstone/s26 a fw want rungs D O v r rc
  local -a arms; arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(s30-old s30-new)
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "S30 STOP: the driver is not $DRV_SHA"; return 1; }
  for a in "${arms[@]}"; do
    case $a in
      s30-old) want=4104ef2029145164;;
      s30-new) want=f187c4d4f3ade200;;
      *) echo "S30 STOP: unknown arm $a"; return 2;;
    esac
    rungs="b0-stats vm-contract b0-hello s30-read"
    fw=$X/fw-$a/fw_payload.bin
    D=${S26R_OUT:-$X}/run-$a O=$X/oracles-$a
    [ "$(sha256sum $fw | cut -c1-16)" = "$want" ] || { echo "S30 STOP: $a's firmware is not $want"; return 1; }
    [ -e $D ] && { echo "S30 STOP: $D exists"; return 1; }
    mkdir -p $D $O; for r in b0-stats b0-hello vm-contract; do echo 0 > $O/$r.oracle; done
    case $a in s30-old) echo 0;; s30-new) echo 3;; esac > $O/s30-read.oracle
    ${S26R_WAIT:-bash $DRV_DIR/board-free-wait.sh 3600 1} || { echo "S30 STOP: board 1 stayed busy"; return 1; }
    echo "S30 $a BEGIN $(date +%T) fw $want rungs [$rungs] board 1" | tee -a $D/SEQ
    ( cd $DRV_DIR || exit 1
      FPGA_BOARD=1 FPGA_BITSTREAM=caplifive_vp2c_8522d72fe.bit FPGA_URL="$(cat $HOME/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$fw \
        BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="$rungs" LADDER_FPGA_DIR=$O BAKED_TIMEOUT=1800 BAKED_IDLE_S=200 PREFLIGHT=0 \
        BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt timeout 4000 ${S26R_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
      runner=$!
      ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 FPGA_BOARD=1 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
      wd=$!; wait $runner; rc=$?; sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null; exit $rc ); rc=$?
    echo "S30 $a DRIVER rc=$rc $(date +%T); board line: $(grep -a -o '^\[fpga-board\] [12]' $D/driver.log | sort -u | tr '\n' ' ')" | tee -a $D/SEQ
    for r in $rungs; do
      v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt 2>/dev/null | tail -1 | sed 's/.*retval=//')
      echo "  RUNG $r retval=${v:-NONE} oracle=$(cat $O/$r.oracle)" | tee -a $D/SEQ
    done
    python3 $S/s30-judge.py $D/board.txt $a > $D/judge.txt 2>&1; echo "S30 $a JUDGE rc=$? :: $(tail -1 $D/judge.txt)" | tee -a $D/SEQ
    sed 's/^/    /' $D/judge.txt >> $D/SEQ
  done
  echo "S30 DONE $(date +%T)"
}
main "$@"; exit $?
