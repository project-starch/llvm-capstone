#!/bin/sh
set -eu
trap 'echo COPYABLE_MOVEMENT_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cd /mnt/vm
insmod capstone_vm.ko unpinned=1 supervisor_slots=32
export CAPSTONE_VM_TORTURE_FORK=2 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec torture.dom 4 1000 > /tmp/movement.out 2> /tmp/movement.err &
pid=$!
tries=0
while ! grep -q '^READY ' /tmp/movement.out; do
    kill -0 "$pid"
    tries=$((tries+1)); test "$tries" -lt 300
    sleep 0.1
done
line=$(grep -m1 '^READY ' /tmp/movement.out)
echo "$line"
heap=$(echo "$line" | sed 's/.* heap=\(0x[0-9a-f]*\).*/\1/')
hb=$(echo "$line" | sed 's/.*heap_bytes=\([0-9]*\).*/\1/')
bss=$(echo "$line" | sed 's/.* bss=\(0x[0-9a-f]*\).*/\1/')
bb=$(echo "$line" | sed 's/.*bss_bytes=\([0-9]*\).*/\1/')
while kill -0 "$pid" 2>/dev/null; do
    ./pfns "$pid" "$heap" "$hb" /tmp/heap-pfns > /tmp/heap-pfns.out 2>/dev/null &&
        echo COPYABLE_PFNS:heap:$(cat /tmp/heap-pfns.out) || true
    ./pfns "$pid" "$bss" "$bb" /tmp/bss-pfns > /tmp/bss-pfns.out 2>/dev/null &&
        echo COPYABLE_PFNS:bss:$(cat /tmp/bss-pfns.out) || true
    echo 1 > /proc/sys/vm/compact_memory
    sleep 0.2
done
wait "$pid"
cat /tmp/movement.out /tmp/movement.err
grep -q 'TORTURE PASS total_bad=0' /tmp/movement.out
sleep 3
status=0
./capstone-vexec torture.dom stale 2 > /tmp/stale.out 2> /tmp/stale.err || status=$?
cat /tmp/stale.out /tmp/stale.err
test "$status" -eq 139
grep -q 'TORTURE PASS total_bad=0' /tmp/stale.out
grep -q 'domain fault cause=25 ' /tmp/stale.err
sleep 3
rmmod capstone_vm
echo COPYABLE_MOVEMENT_COMPLETE
