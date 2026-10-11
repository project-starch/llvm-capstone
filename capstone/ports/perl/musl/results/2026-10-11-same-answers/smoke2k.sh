#!/usr/bin/env bash
# Follow-up: smoke.pl with CAPSTONE_PERL_SMOKE_OBJECTS=2000 on native and on the same two images, one boot.
set -u
K=$HOME/pg-virtual-kit SA=$HOME/sa-perl R=$SA/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=$SA/qemu.lock
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl NP=$SA/perl-stock/native/bin/perl
NR=$SA/native-smoke2k; rm -rf $NR; mkdir -p $NR/smoke; cp $SMOKE $NR/smoke/smoke.pl
s=$(date +%s.%N)
(cd $NR/smoke && CAPSTONE_PERL_SMOKE_OBJECTS=2000 $NP smoke.pl $NR/smoke > $NR/smoke.stdout 2> $NR/smoke.stderr); rc=$?
echo "NATIVE smoke2k rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc) sha256=$(sha256sum < $NR/smoke.stdout | cut -d" " -f1)"
ST=$SA/stage-smoke2k; rm -rf $ST; mkdir -p $ST
cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $SMOKE $ST/
for arm in stock protected; do cp $SA/stage/$arm/perl.dom $ST/perl-$arm.dom; done
sha256sum $ST/*.dom $ST/smoke.pl $ST/capstone-vexec $ST/capstone_vm.ko
cat > $ST/gate.sh <<'G'
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo $u; }
for arm in stock protected; do
  mkdir -p /tmp/smoke-$arm; cp /mnt/vm/smoke.pl /tmp/smoke-$arm/smoke.pl; cd /tmp/smoke-$arm
  s=$(up)
  CAPSTONE_PERL_SMOKE_OBJECTS=2000 /mnt/vm/capstone-vexec /mnt/vm/perl-$arm.dom smoke.pl /tmp/smoke-$arm > /tmp/sa.out 2> /tmp/sa.err
  rc=$?
  e=$(up)
  h=$(sha256sum /tmp/sa.out | cut -d " " -f 1)
  echo "SA_BEGIN:$arm:smoke"
  cat /tmp/sa.out
  echo "SA_STDERR:$arm:smoke"
  cat /tmp/sa.err
  echo "SA_END:$arm:smoke:$rc:$s:$e:$h"
done
cd /mnt/vm
rmmod capstone_vm
echo SA_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
W=$SA/guest-smoke2k-$(date +%Y%m%d-%H%M%S)
date -u +"BOOT start %Y-%m-%dT%H:%M:%SZ (host clock)"
s=$(date +%s)
python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage $ST --work $W --timeout 1800
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s work=$W"
echo RUN-DONE
