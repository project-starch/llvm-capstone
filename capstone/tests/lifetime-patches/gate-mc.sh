#!/bin/sh
cd /mnt/vm || exit 1
for v in fixed orig; do
  for m in stale alloc newslab big cache; do
    ./capstone-vexec ./mc-test-$v.dom $m > /tmp/o 2>&1; rc=$?
    sed "s/^/TEST $v $m: /" /tmp/o | head -9
    echo "TEST $v $m: exit $rc"
  done
done
echo VIRTUAL_STAGED_DONE
