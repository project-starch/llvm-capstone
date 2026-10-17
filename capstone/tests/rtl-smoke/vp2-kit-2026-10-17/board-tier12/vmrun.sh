#!/bin/sh
# vp2 Tier 1/2 controller (the baked-rung driver's BAKED_CTL): the vm-* rungs here, every other rung to b0run.sh
# unchanged (BL's controls). vm-suite STREAMS its output (the driver ends a rung after BAKED_IDLE_S of silence) and
# prints a heartbeat every 30 s while it runs; the heartbeat lines start with VMHB, which no run.py check reads.
rung=$1
case $rung in
  vm-contract) sh /test-domains/vm/vm-contract.sh ;;
  vm-suite)
    ( i=0; while [ $i -lt 200 ]; do sleep 30; i=$((i+1)); echo "VMHB $i"; done ) & hb=$!
    if command -v tee >/dev/null 2>&1; then
      sh /test-domains/vm/vm-suite-inner.sh 2>&1 | tee /tmp/vm-suite.out
    else
      sh /test-domains/vm/vm-suite-inner.sh > /tmp/vm-suite.out 2>&1; cat /tmp/vm-suite.out
    fi
    kill $hb 2>/dev/null
    n=$(grep -c '^VM_EXIT:' /tmp/vm-suite.out); echo "RESULT vm-suite retval=$n" ;;
  *) exec /test-domains/b0run.sh "$@" ;;
esac
