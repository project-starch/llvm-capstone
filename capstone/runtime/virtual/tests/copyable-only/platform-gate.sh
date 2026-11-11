#!/bin/sh
set -eu
trap 'echo COPYABLE_PLATFORM_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cd /mnt/vm
dmesg -n 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
expected=$(cat expected)
case "$expected" in
    enabled)
        test "$(cat /sys/module/captag/parameters/enabled)" = Y
        insmod capstone_vm.ko supervisor_slots=32 unpinned=1
        for op in L S Q A; do
            echo COPYABLE_RAW_BEGIN:$op
            status=0
            ./capstone-vexec raw-op.dom "$op" >/tmp/raw-op.log 2>&1 || status=$?
            cat /tmp/raw-op.log
            test "$status" = 139
            test "$(grep -c 'domain fault cause=2 ' /tmp/raw-op.log)" = 1
            echo COPYABLE_RAW_REFUSED:$op
        done
        rmmod capstone_vm
        echo COPYABLE_PLATFORM_ENABLED
        ;;
    disabled)
        test "$(cat /sys/module/captag/parameters/enabled)" = N
        if insmod capstone_vm.ko supervisor_slots=32 unpinned=1 >/tmp/load.log 2>&1; then
            echo COPYABLE_UNEXPECTED_LOAD
            rmmod capstone_vm
            exit 1
        fi
        cat /tmp/load.log
        grep -q 'Operation not supported' /tmp/load.log
        test ! -e /dev/capstone-vm
        echo COPYABLE_PLATFORM_REFUSED
        ;;
    *) exit 2;;
esac
