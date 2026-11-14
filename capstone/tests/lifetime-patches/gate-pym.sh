#!/bin/sh
cd /mnt/vm || exit 1
for v in B O; do
  for m in stale alloc realloc arena newarena; do
    ./capstone-vexec ./pym-test-$v.dom $m > /tmp/o 2>&1; rc=$?
    sed "s/^/TEST $v $m: /" /tmp/o | head -6
    echo "TEST $v $m: exit $rc"
  done
done
echo VIRTUAL_STAGED_DONE
