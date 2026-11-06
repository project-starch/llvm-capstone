#!/bin/sh
set -eu
trap 'echo NATIVE_TASK_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
cp /mnt/vm/probe.dom /mnt/vm/blocked.dom /mnt/vm/entry-driver /mnt/vm/gcc-test.dom /mnt/vm/t1.c /mnt/vm/compute.dom /mnt/vm/task-driver /tmp/
chmod +x /tmp/*.dom /tmp/entry-driver
cd /tmp
yes Q | head -c 131072 > input
./probe.dom <input >output
cat output
./task-driver ./compute.dom ./blocked.dom
./entry-driver ./blocked.dom
./gcc-test.dom t1.c -O3 -finline-limit=50000 -o result.s
cmp result.s /mnt/vm/reference.s
echo PASS:GCC_EXACT_OUTPUT
