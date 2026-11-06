#!/bin/sh
set -eu
trap 'echo COPYABLE_GATE_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cd /mnt/vm
dmesg -n 1
unpinned=$(cat unpinned)
case "$unpinned" in 0|1) ;; *) exit 2;; esac
insmod capstone_vm.ko supervisor_slots=32 unpinned="$unpinned"
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_EXEC_DIAGNOSTICS=1
for mode in ok vm vm-large vm-mappings nested threads heap-threads map-churn stale bounds nested-stale retired vm-ro-write vm-none-read vm-guard vm-padding; do
    echo COPYABLE_BEGIN:$mode
    status=0
    ./capstone-vexec contract.dom "$mode" argument > /tmp/copyable-case.log 2>&1 || status=$?
    cat /tmp/copyable-case.log
    echo COPYABLE_RESULT:$mode:$status
done
./abi-probe
sleep 1
echo COPYABLE_TIMER_WAKE_PASS
echo COPYABLE_UNLOAD_BEGIN
rmmod capstone_vm
echo COPYABLE_SUITE_COMPLETE
