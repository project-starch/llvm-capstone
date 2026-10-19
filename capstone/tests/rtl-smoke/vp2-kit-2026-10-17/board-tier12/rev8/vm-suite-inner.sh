#!/bin/sh
# vp2 Tier 1 for module 39d8492d, rev8: the stopgap's positive control without sparse-churn. On rev7, sparse-churn wedged
# the core with S-25's face before the table ran out. dense-churn (contract-dense.dom, the compiler lane's) retires one
# arena, so collect() is reachable, then derives children inside ONE already-written 4096-byte block until the flat
# table runs out. That should give the must-fire "capstone_vm: collect skipped: ..." (KERN_EMERG) and a terminal end.
# It runs between two reads of the counters, with the module as vm-contract loaded it.
cd /test-domains/vm || exit 1
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko diag_emerg=1 node_flat_records=@NFR@ || exit 1
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
C="resume_sfences resume_satp_refreshes resume_satp_changed busy_frame_refusals"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):before-churn"; done
echo VM_BEGIN:dense-churn
./capstone-vexec contract-dense.dom dense-churn > /tmp/churn.out 2>&1
echo VM_EXIT:dense-churn:$?
tail -40 /tmp/churn.out
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):after-churn"; done
dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-CHURN: /'; echo KMSG-AFTER-CHURN-END lines=$(dmesg | wc -l)
