#!/bin/sh
# Run as gate.sh through run-staged.py with the qualified native kernel flags.
set -eu
trap 'echo MODULE_FREE_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
# Built-in parameters exist, but the adapter must not be a loaded module.
test "$(cat /sys/module/capstone_vm/parameters/native_task)" = Y
test "$(cat /sys/module/capstone_vm/parameters/native_dispatch)" = Y
if grep -q '^capstone_vm ' /proc/modules; then
    echo 'FAIL: dynamically loaded Capstone module'
    exit 1
fi
# Inspect the root filesystem and staged resources, not proc/sys/device mounts.
artifacts=$(find / -xdev -name 'capstone_vm.ko*'; find /mnt/vm -name 'capstone_vm.ko*')
if test -n "$artifacts"; then
    echo "FAIL: obsolete module artifacts: $artifacts"
    exit 1
fi
if grep -r -E '(insmod|modprobe).*capstone_vm' /etc/init.d /etc/inittab; then
    echo 'FAIL: obsolete Capstone startup loader'
    exit 1
fi
echo MODULE_FREE_PASS
