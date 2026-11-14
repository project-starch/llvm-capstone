#!/bin/sh
cd /mnt/vm || exit 1
export HOME=/tmp TMPDIR=/tmp PYTHONHOME=/mnt/vm/cpy
for v in fixed orig; do
  for m in stale alloc realloc; do
    ./capstone-vexec ./cpy-test-$v.dom $m > /tmp/o 2>&1; rc=$?
    sed "s/^/TEST $v $m: /" /tmp/o | grep -v '^TEST [a-z]* [a-z]*: $' | head -8
    echo "TEST $v $m: exit $rc"
  done
done
echo VIRTUAL_STAGED_DONE
