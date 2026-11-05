#!/bin/bash
# after-gate.sh <board> <gate log> <want READ regex> <waitpid> <waitlog> <waitpfx>: start gt-run.sh only once the QEMU gate
# log reads the wanted line (the QEMU gate before any board call); STOP after 4 h without it.
main() {
  local BRD=$1 GL=$2 RX=$3 H=${CAPSTONE_WORK:?} R=${CAPSTONE_WORK:?}/perf/ga t=0 L
  L=$R/gt-b$BRD.log
  until grep -q -E "$RX" "$GL" 2>/dev/null; do
    [ $t -ge 14400 ] && { echo "GT $(date '+%F %T') STOP: QEMU gate $GL never read $RX" >> $L; return 2; }
    sleep 60; t=$((t+60)); done
  echo "GT $(date '+%F %T') QEMU gate read: $(grep -E "$RX" "$GL" | head -1)" >> $L
  CAPSTONE_WORK=$H GT_O=$R GT_OUT=$H/spec/board/out-ga-b$BRD GT_PREREG=$GT_PREREG GT_WAITPID=$4 GT_WAITLOG=$5 GT_WAITPFX=$6 bash ${CAPSTONE_WORK:?}/perf/ga/gt-run.sh $BRD
}
main "$@"; exit $?
