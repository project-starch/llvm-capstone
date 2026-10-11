#!/bin/sh
# Run every corpus case with the domain mruby, one verdict line each.
# Each case gets a watchdog, because three of them loop forever at the pin and
# this guest has no timeout(1).
BIN=$1
# This busybox has no timeout(1), so each case gets a watchdog of its own:
# the case runs in the background and a sleeper kills it if it outlives 45 s.
echo "ARMS-BEGIN watchdog=45"
while read n; do
  cd /mnt/host
  "$BIN" /mnt/host/cases/$n.rb > /tmp/case.out 2>&1 &
  pid=$!
  ( sleep 45; kill -9 $pid 2>/dev/null ) > /dev/null 2>&1 &
  dog=$!
  wait $pid
  st=$?
  kill $dog 2>/dev/null
  out=$(cat /tmp/case.out)
  first=$(echo "$out" | grep -m1 '^\[')
  fault=$(echo "$out" | grep -m1 -o 'cause=[0-9]*')
  echo "CASE $n status=$st fault=${fault:-none} first=${first:-none}"
done < /mnt/host/cases.txt
echo "ARMS-DONE"
