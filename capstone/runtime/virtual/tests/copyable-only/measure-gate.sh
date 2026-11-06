#!/bin/sh
set -eu
trap 'echo COPYABLE_MEASURE_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cd /mnt/vm
dmesg -n 1
killall syslogd klogd 2>/dev/null || true
echo 0 > /proc/sys/kernel/randomize_va_space
export CAPSTONE_PROFILE_RECORD=/tmp/image-records.txt
for rep in 1 2 3; do
    arms='0 1'
    if [ "$rep" = 2 ]; then arms='1 0'; fi
    for unpinned in $arms; do
        echo COPYABLE_MEASURE_BEGIN:$rep:$unpinned
        insmod capstone_vm.ko supervisor_slots=32 unpinned="$unpinned"
        for f in /sys/module/capstone_vm/sections/.*; do
            echo SECTION $(basename "$f") $(cat "$f")
        done
        : > "$CAPSTONE_PROFILE_RECORD"
        ./capstone-vexec allocation.dom 16384
        cat "$CAPSTONE_PROFILE_RECORD"
        echo PAYLOAD_FAULTS:$(cat /sys/module/capstone_vm/parameters/unpinned_faults)
        rmmod capstone_vm
        echo COPYABLE_MEASURE_END:$rep:$unpinned
    done
done
