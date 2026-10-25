#!/bin/sh
# Cross-check boot: kernel peak RSS via time -v and the launcher's own CAPSTONE_VM_STATS line.
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export HOME=/tmp TMPDIR=/tmp
echo PORT_BEGIN:stats-before
./capstone-vexec --stats
echo PORT_END:stats-before:$?
echo PORT_BEGIN:cpython
CAPSTONE_VM_STATS=1 PYTHONHOME=/mnt/vm/cpy /usr/bin/time -v ./capstone-vexec ./cpython.dom -S /mnt/vm/mem_driver.py > /tmp/py.out 2>&1
rc=$?
cat /tmp/py.out
echo PORT_END:cpython:$rc
echo PORT_BEGIN:stats-after
./capstone-vexec --stats
echo PORT_END:stats-after:$?
echo PORT_BEGIN:cleanup
rmmod capstone_vm
echo PORT_END:cleanup:$?
echo VIRTUAL_STAGED_DONE
