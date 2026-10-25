#!/usr/bin/env bash
# Positive control: the glob-assign one-liner on the recorded images WITHOUT 0011 (the same-answers
# images, stock e4c27891 and protected 20856abd), one boot.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf R=$M/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=$M/qemu-ctl.lock
ST=$M/stage-ctl; rm -rf $ST; mkdir -p $ST
cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $ST/
cp $HOME/sa-perl/stage-smoke2k/perl-stock.dom $ST/stock.dom; cp $HOME/sa-perl/stage-smoke2k/perl-protected.dom $ST/protected.dom
sha256sum $ST/*.dom
cat > $ST/gate.sh <<'G'
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
for arm in stock protected; do
  echo "CTL_IMAGE:$arm:$(sha256sum /mnt/vm/$arm.dom | cut -d ' ' -f 1)"
  mkdir -p /tmp/ctl-$arm; cd /tmp/ctl-$arm
  /mnt/vm/capstone-vexec /mnt/vm/$arm.dom -e '$x = *foo; *x = $x; print "done\n"' > /tmp/c.out 2> /tmp/c.err
  rc=$?
  echo "CTL_BEGIN:$arm"; cat /tmp/c.out; echo "CTL_STDERR:$arm"; cat /tmp/c.err; echo "CTL_END:$arm:$rc"
done
cd /mnt/vm
rmmod capstone_vm
echo CTL_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
W=$M/guest-ctl-$(date +%Y%m%d-%H%M%S); s=$(date +%s)
python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage $ST --work $W --timeout 1200
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s work=$W"
echo CTL-DONE
