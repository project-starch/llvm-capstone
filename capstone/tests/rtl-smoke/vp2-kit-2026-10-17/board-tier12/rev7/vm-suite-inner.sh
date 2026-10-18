#!/bin/sh
# vp2 Tier 1 for the module revision 39d8492d: sparse-churn LAST, between two reads of the counters. sparse-churn is the
# blocker-6 stopgap's positive control: under node_flat_records=4096 it exhausts within a few thousand allocations, the
# module refuses the collect with -ENOMEM and the must-fire "capstone_vm: collect skipped: ..." (KERN_EMERG), and the
# program ends VM_CONTRACT_FAIL:recycling allocation, exit 1. It was held NOT RUN under S-25 (a first-touch capability
# store has wedged the core, R-57), so it runs last, behind VM_BEGIN, with every other reading already taken.
cd /test-domains/vm || exit 1
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko diag_emerg=1 node_flat_records=4096 || exit 1
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
C="resume_sfences resume_satp_refreshes resume_satp_changed busy_frame_refusals"
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):before-churn"; done
echo VM_BEGIN:sparse-churn
./capstone-vexec contract.dom sparse-churn > /tmp/churn.out 2>&1
echo VM_EXIT:sparse-churn:$?
cat /tmp/churn.out
for p in $C; do echo "@@PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo ABSENT):after-churn"; done
dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-CHURN: /'; echo KMSG-AFTER-CHURN-END lines=$(dmesg | wc -l)
