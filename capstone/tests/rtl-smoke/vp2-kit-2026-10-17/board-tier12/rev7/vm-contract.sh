#!/bin/sh
# vp2 Tier 1, the first virtual rung, for the module revision after d71f207a (39d8492d, the five items): as before,
# with the module loaded with diag_emerg=1 (CV_EMIT_FIRST lines at KERN_EMERG, the console being level 1) and
# node_flat_records=4096 (so sparse-churn reaches the stopgap's must-fire within a few thousand allocations), both read
# back, and the module-wide resume counters read before `ok` and after `stale` (predictions are on the deltas).
# RESULT vm-contract retval: 0 both as QEMU's run.py record; 921 capstone_vm.ko did not load; 922 `ok` exited
# non-zero; 923 no VIRTUAL_APPLICATION_OK line; 924 `stale` did not exit 139; 925 no `domain fault cause=25` line.
cd /test-domains/vm || { echo "RESULT vm-contract retval=921"; exit 1; }
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko diag_emerg=1 node_flat_records=4096 || { echo "RESULT vm-contract retval=921"; exit 1; }
for p in diag_emerg node_flat_records supervisor_slots; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT)"; done
C="resume_sfences resume_satp_refreshes resume_satp_changed busy_frame_refusals"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):before-contract"; done
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/c-ok.out 2>&1; ok=$?
cat /tmp/c-ok.out; echo "VM_EXIT:normal:$ok"
./capstone-vexec contract.dom stale > /tmp/c-st.out 2>&1; st=$?
cat /tmp/c-st.out; echo "VM_EXIT:stale:$st"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):after-contract"; done
r=0
[ $ok -eq 0 ] || r=922
[ $r -eq 0 ] && { grep -q "^VIRTUAL_APPLICATION_OK " /tmp/c-ok.out || r=923; }
[ $r -eq 0 ] && { [ $st -eq 139 ] || r=924; }
[ $r -eq 0 ] && { grep -q "domain fault cause=25 " /tmp/c-st.out || r=925; }
echo "RESULT vm-contract retval=$r"
