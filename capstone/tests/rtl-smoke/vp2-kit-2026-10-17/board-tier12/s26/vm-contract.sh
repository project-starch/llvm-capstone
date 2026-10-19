#!/bin/sh
# S-26's silicon pair (2026-10-19): rev7's vm-contract rung, BOUNDED, with the module-wide resume counters read while
# `ok` runs. The S-26 candidate (csupquantum 0 before any process-ABI STEP) predicts a context that escapes kind 1,
# is resumed and is preempted again forever: the counters then GROW at every read. A context that escaped and was
# never resumed reads them FLAT. `ok` runs in the background; the counters are read at +30, +90 and +150 s; at the
# bound a still-running `ok` is killed and the rung reports VM_SPIN. If `ok` returns, the rung ends as rev7's.
# RESULT vm-contract retval: 0 as rev7's identity branch; 921 the module did not load; 922-925 as rev7's; 930 `ok` still
# running at the bound (VM_SPIN).
cd /test-domains/vm || { echo "RESULT vm-contract retval=921"; exit 1; }
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko diag_emerg=1 node_flat_records=4096 || { echo "RESULT vm-contract retval=921"; exit 1; }
for p in diag_emerg node_flat_records supervisor_slots; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT)"; done
C="resume_sfences resume_satp_refreshes resume_satp_changed busy_frame_refusals"
rd() { for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):$1"; done; }
rd before-ok
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/c-ok.out 2>&1 &
pid=$!; t=0; ok=
for mark in 30 90 150; do
  while [ $t -lt $mark ]; do kill -0 $pid 2>/dev/null || break 2; sleep 5; t=$((t + 5)); done
  rd "t$mark"
done
if kill -0 $pid 2>/dev/null; then
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  cat /tmp/c-ok.out; echo "VM_SPIN:ok at the 150 s bound"; rd after-kill
  echo "RESULT vm-contract retval=930"; exit 0
fi
wait $pid; ok=$?
cat /tmp/c-ok.out; echo "VM_EXIT:normal:$ok"; rd after-ok
./capstone-vexec contract.dom stale > /tmp/c-st.out 2>&1; st=$?
cat /tmp/c-st.out; echo "VM_EXIT:stale:$st"
r=0
[ $ok -eq 0 ] || r=922
[ $r -eq 0 ] && { grep -q "^VIRTUAL_APPLICATION_OK " /tmp/c-ok.out || r=923; }
[ $r -eq 0 ] && { [ $st -eq 139 ] || r=924; }
[ $r -eq 0 ] && { grep -q "domain fault cause=25 " /tmp/c-st.out || r=925; }
echo "RESULT vm-contract retval=$r"
