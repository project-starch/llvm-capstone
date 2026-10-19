#!/bin/sh
# vp2 Tier 1, the first virtual rung, for module 39d8492d, rev8: rev7's rung with two changes. The module is loaded with
# node_flat_records=@NFR@, the smallest whole-page capacity at which `ok` passes on QEMU, so that dense-churn exhausts the
# table early. And the counters are also read BETWEEN `ok` and `stale`, so that `ok`'s deltas can be checked exactly
# against its own CAPSTONE_VM_STATS line (rev7's open check: `stale` faults and prints no STATS line).
# RESULT vm-contract retval: 0 both as QEMU's run.py record; 921 capstone_vm.ko did not load; 922 `ok` exited
# non-zero; 923 no VIRTUAL_APPLICATION_OK line; 924 `stale` did not exit 139; 925 no `domain fault cause=25` line.
cd /test-domains/vm || { echo "RESULT vm-contract retval=921"; exit 1; }
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko diag_emerg=1 node_flat_records=@NFR@ || { echo "RESULT vm-contract retval=921"; exit 1; }
for p in diag_emerg node_flat_records supervisor_slots; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT)"; done
C="resume_sfences resume_satp_refreshes resume_satp_changed busy_frame_refusals"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):before-contract"; done
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/c-ok.out 2>&1; ok=$?
cat /tmp/c-ok.out; echo "VM_EXIT:normal:$ok"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):after-ok"; done
./capstone-vexec contract.dom stale > /tmp/c-st.out 2>&1; st=$?
cat /tmp/c-st.out; echo "VM_EXIT:stale:$st"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):after-contract"; done
r=0
[ $ok -eq 0 ] || r=922
[ $r -eq 0 ] && { grep -q "^VIRTUAL_APPLICATION_OK " /tmp/c-ok.out || r=923; }
[ $r -eq 0 ] && { [ $st -eq 139 ] || r=924; }
[ $r -eq 0 ] && { grep -q "domain fault cause=25 " /tmp/c-st.out || r=925; }
echo "RESULT vm-contract retval=$r"
