#!/bin/sh
# Guest side of the Capstone twin: one launch per scenario, the launcher's output (SCEN, ACCESS,
# or the domain fault line) folded into one @@PROBE line per scenario. run-staged.py mounts the
# stage read-only at /mnt/vm.
dmesg -n 1
cd /mnt/vm || { echo "@@NO_STAGE"; echo VIRTUAL_STAGED_DONE; exit 1; }
insmod /mnt/vm/capstone_vm.ko || { echo "@@INSMOD failed"; echo VIRTUAL_STAGED_DONE; exit 1; }
for mode in fixed freed shrink-small-same-class shrink-small-other-class grow-small-within-class \
            shrink-large-in-place grow-moved two-owners-in-place two-owners-moved \
            mruby-sort-bang-shrink perl-hsplit-aux-grow; do
  ./capstone-vexec probe.dom "$mode" > /tmp/o 2>&1
  echo "@@PROBE $mode rc=$? $(tr '\n' ' ' < /tmp/o)"
done
rmmod capstone_vm
echo "@@CLEANUP rc=$?"
echo VIRTUAL_STAGED_DONE
