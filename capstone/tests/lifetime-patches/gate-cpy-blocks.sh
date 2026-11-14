#!/bin/sh
cd /mnt/vm || exit 1
export HOME=/tmp TMPDIR=/tmp PYTHONHOME=/mnt/vm/cpy
for img in new-stock new-protected old-stock old-protected; do
  ./capstone-vexec ./cpython-$img.dom -S -c 'import sys; n = sys.getallocatedblocks(); print("CPYBLOCKS", n > 0, n)' > /tmp/o 2>&1; rc=$?
  sed "s/^/TEST $img: /" /tmp/o | grep -v REUSE-GAP | head -5
  echo "TEST $img: exit $rc"
done
echo VIRTUAL_STAGED_DONE
