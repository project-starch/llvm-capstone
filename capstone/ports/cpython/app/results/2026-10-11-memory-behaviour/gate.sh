#!/bin/sh
# One arm, one repetition: lifetime counters before and after one interpreter run.
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export HOME=/tmp TMPDIR=/tmp
echo PORT_BEGIN:stats-before
./capstone-vexec --stats
echo PORT_END:stats-before:$?
echo PORT_BEGIN:cpython
t0=$(date +%s)
PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S /mnt/vm/mem_driver.py > /tmp/py.out 2>&1 &
pid=$!
hwm=none; n=0
while kill -0 $pid 2>/dev/null; do
  h=$(sed -n 's/^VmHWM:[[:space:]]*//p' /proc/$pid/status 2>/dev/null)
  [ -n "$h" ] && hwm=$h && n=$((n+1))
  sleep 0.2 2>/dev/null || sleep 1
done
wait $pid; rc=$?
echo PORT_TIME:cpython:$(( $(date +%s) - t0 ))
cat /tmp/py.out
echo "SAMPLED_VMHWM:$hwm samples=$n"
echo PORT_END:cpython:$rc
echo PORT_BEGIN:stats-after
./capstone-vexec --stats
echo PORT_END:stats-after:$?
echo PORT_BEGIN:verbatim
PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S -c 'import json,gc; a=[{"n":n} for n in range(1000)]; assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")'
echo PORT_END:verbatim:$?
echo PORT_BEGIN:stats-final
./capstone-vexec --stats
echo PORT_END:stats-final:$?
echo PORT_BEGIN:cleanup
rmmod capstone_vm
echo PORT_END:cleanup:$?
echo VIRTUAL_STAGED_DONE
