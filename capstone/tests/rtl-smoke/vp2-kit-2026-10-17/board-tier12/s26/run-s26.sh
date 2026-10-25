#!/bin/bash
# S-26's silicon pair on tree A, board 1 (2026-10-19, pre-registered in tier3-smoke/README.md before the boots): three cold
# boots in the runtime lane's order, var, then cur, then mir, each on its own firmware built on the pair's Image
# (89baf46451448872: rev7's stage with the bounded, counter-reading vm-contract.sh), module 39d8492d.
#   var: wrapper boot write 0x7C3 = 2,000,000; rungs b0-stats vm-contract; predicted to RETURN (identity branch).
#   cur: no boot write (today's monitor); rungs b0-stats vm-contract; predicted to SPIN with growing resume counters.
#   mir: census write 0x7C3 = 1,000; rungs b0-stats b0-hello b0-stats2 vm-contract; predicted to SPIN (VOID if b0-hello
#        does not return 0).
# Added 2026-10-19 evening (pre-registered in tier3-smoke/README.md before their boots):
#   fix1: the S-23a/S-26 monitor revision's canary (capstone-sbi 5b5e86b, wrapper aab63d5, CAPSTONE_BOOT_QUANTUM=2000000) on
#         the pair's image; rungs b0-stats vm-contract b0-hello, so vm-contract runs with NO process-ABI STEP before it.
#   q600, q300, q100: S-26's silicon bracket, var1k's recipe with CAPSTONE_BOOT_QUANTUM 600/300/100 on the bracket image
#         (the pair's stage with vm-contract's bound at 600 s); vm-contract is a reading (completed with steps, or a bound).
# Judged by s26-judge.py. Usage: run-s26.sh [arm...] (default var cur mir). Ends by itself. S26R_* stub for the dry run.
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
main() {
  set -u
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local X=/tmp/capstone/s26 a fw want rungs D O v r rc
  local -a arms; arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(var cur mir)
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "S26 STOP: the driver is not $DRV_SHA"; return 1; }
  for a in "${arms[@]}"; do
    case $a in
      var) want=883f3af2b9964251; rungs="b0-stats vm-contract";;
      cur) want=fafbc0e38fb54c60; rungs="b0-stats vm-contract";;
      mir) want=1a2244dfa19f9d40; rungs="b0-stats b0-hello b0-stats2 vm-contract";;
      cur-esc) want=9e4924294f94eafe; rungs="b0-stats vm-contract";;   # firming arm 1: cur with diag_escapes=1 (image 3537c881f1020464)
      var1k) want=2406f5a523ca00e4; rungs="b0-stats vm-contract";;     # firming arm 2: var's boot write with 1,000 (image 89baf46451448872)
      fix1) want=f40b38ae6b0f74f7; rungs="b0-stats vm-contract b0-hello";;
      q600) want=85df5930f42dfec1; rungs="b0-stats vm-contract";;
      q300) want=95f97ad9426a7232; rungs="b0-stats vm-contract";;
      q100) want=104d3f8f629ccdee; rungs="b0-stats vm-contract";;
      *) echo "S26 STOP: unknown arm $a"; return 2;;
    esac
    fw=$X/fw-s26-$a/fw_payload.bin; [ $a = fix1 ] && fw=/tmp/capstone/s23fix/fw-canary1/fw_payload.bin
    D=${S26R_OUT:-$X}/run-$a O=$X/oracles-$a
    [ "$(sha256sum $fw | cut -c1-16)" = "$want" ] || { echo "S26 STOP: $a's firmware is not $want"; return 1; }
    [ -e $D ] && { echo "S26 STOP: $D exists"; return 1; }
    mkdir -p $D $O; for r in b0-stats b0-hello b0-stats2; do echo 0 > $O/$r.oracle; done
    case $a in var|fix1|q*) echo 0;; *) echo 930;; esac > $O/vm-contract.oracle
    ${S26R_WAIT:-bash $DRV_DIR/board-free-wait.sh 3600 1} || { echo "S26 STOP: board 1 stayed busy"; return 1; }
    echo "S26 $a BEGIN $(date +%T) fw $want rungs [$rungs] board 1" | tee -a $D/SEQ
    ( cd $DRV_DIR || exit 1
      FPGA_BOARD=1 FPGA_BITSTREAM=caplifive_vp2c_8522d72fe.bit FPGA_URL="$(cat $HOME/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$fw \
        BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="$rungs" LADDER_FPGA_DIR=$O BAKED_TIMEOUT=1800 BAKED_IDLE_S=200 PREFLIGHT=0 \
        BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt timeout 4000 ${S26R_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
      runner=$!
      ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 FPGA_BOARD=1 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
      wd=$!; wait $runner; rc=$?; sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null; exit $rc ); rc=$?
    echo "S26 $a DRIVER rc=$rc $(date +%T); board line: $(grep -a -o '^\[fpga-board\] [12]' $D/driver.log | sort -u | tr '\n' ' ')" | tee -a $D/SEQ
    for r in $rungs; do
      v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt 2>/dev/null | tail -1 | sed 's/.*retval=//')
      echo "  RUNG $r retval=${v:-NONE} oracle=$(cat $O/$r.oracle)" | tee -a $D/SEQ
    done
    python3 $S/s26-judge.py $D/board.txt $a > $D/judge.txt 2>&1; echo "S26 $a JUDGE rc=$? :: $(tail -1 $D/judge.txt)" | tee -a $D/SEQ
    sed 's/^/    /' $D/judge.txt >> $D/SEQ
  done
  echo "S26 DONE $(date +%T)"
}
main "$@"; exit $?
