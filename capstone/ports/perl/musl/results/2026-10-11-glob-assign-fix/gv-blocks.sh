#!/usr/bin/env bash
# The two glob_assign blocks of op/gv.t as a standalone script, on the 0011 images and the
# recorded images without 0011 (one boot), and natively with and without 0011.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf R=$M/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=$M/qemu-gvb.lock
for v in without-0011 with-0011; do
  (cd $M && $M/native-$v/perl-5.36.3/perl gv-blocks.pl > $M/gvb-native-$v.out 2>&1); echo "NATIVE $v rc=$? sha256=$(sha256sum < $M/gvb-native-$v.out | cut -d' ' -f1)"; done
ST=$M/stage-gvb; rm -rf $ST; mkdir -p $ST
cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $M/gv-blocks.pl $ST/
cp $M/perl-stock/src/perl-5.36.3/perl $ST/new-stock.dom; cp $M/perl-protected/src/perl-5.36.3/perl $ST/new-protected.dom
cp $HOME/sa-perl/stage-smoke2k/perl-stock.dom $ST/old-stock.dom; cp $HOME/sa-perl/stage-smoke2k/perl-protected.dom $ST/old-protected.dom
sha256sum $ST/*.dom $ST/gv-blocks.pl
cat > $ST/gate.sh <<'G'
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
for img in new-stock new-protected old-stock old-protected; do
  mkdir -p /tmp/g-$img; cp /mnt/vm/gv-blocks.pl /tmp/g-$img/; cd /tmp/g-$img
  /mnt/vm/capstone-vexec /mnt/vm/$img.dom gv-blocks.pl > /tmp/g.out 2> /tmp/g.err
  rc=$?
  echo "GVB_BEGIN:$img"; cat /tmp/g.out; echo "GVB_STDERR:$img"; cat /tmp/g.err
  echo "GVB_END:$img:$rc:$(sha256sum /tmp/g.out | cut -d ' ' -f 1)"
done
cd /mnt/vm
rmmod capstone_vm
echo GVB_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
W=$M/guest-gvb-$(date +%Y%m%d-%H%M%S); s=$(date +%s)
python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage $ST --work $W --timeout 1200
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s work=$W"
echo GVB-DONE
