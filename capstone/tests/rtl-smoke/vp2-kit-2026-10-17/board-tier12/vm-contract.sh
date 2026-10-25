#!/bin/sh
# vp2 Tier 1, the first virtual rung: contract.dom alone. Its two readings are VIRTUAL_APPLICATION_OK (exit 0), and a
# stale access denied with cause 25 (exit 139), which is also the identity reading that the run is virtual.
# RESULT vm-contract retval: 0 both as QEMU's run.py record; 921 capstone_vm.ko did not load; 922 `ok` exited non-zero;
# 923 no VIRTUAL_APPLICATION_OK line; 924 `stale` did not exit 139; 925 no `domain fault cause=25` line.
cd /test-domains/vm || { echo "RESULT vm-contract retval=921"; exit 1; }
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko || { echo "RESULT vm-contract retval=921"; exit 1; }
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/c-ok.out 2>&1; ok=$?
cat /tmp/c-ok.out; echo "VM_EXIT:normal:$ok"
./capstone-vexec contract.dom stale > /tmp/c-st.out 2>&1; st=$?
cat /tmp/c-st.out; echo "VM_EXIT:stale:$st"
r=0
[ $ok -eq 0 ] || r=922
[ $r -eq 0 ] && { grep -q "^VIRTUAL_APPLICATION_OK " /tmp/c-ok.out || r=923; }
[ $r -eq 0 ] && { [ $st -eq 139 ] || r=924; }
[ $r -eq 0 ] && { grep -q "domain fault cause=25 " /tmp/c-st.out || r=925; }
echo "RESULT vm-contract retval=$r"
