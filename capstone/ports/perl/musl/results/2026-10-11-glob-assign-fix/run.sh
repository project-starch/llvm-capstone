#!/usr/bin/env bash
# Glob-assign fix (0011): ONE boot running, per image (stock, then protected), the control
# `$x = *foo; *x = $x; print "done\n"`, t/base (9 files) and smoke.pl at 2,000 objects,
# as ports/perl/musl/results/2026-10-11-same-answers (its run.sh and smoke2k.sh).
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf R=$M/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=$M/qemu-sa.lock
TESTS="base/cond.t base/if.t base/lex.t base/num.t base/pat.t base/rs.t base/term.t base/translate.t base/while.t"
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl
ST=$M/stage; rm -rf $ST; mkdir -p $ST
cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $SMOKE $ST/
for arm in stock protected; do
  S=$M/perl-$arm/src/perl-5.36.3; mkdir -p $ST/$arm
  cp $S/perl $ST/$arm/perl.dom
  rsync -a --exclude="*.o" --exclude="*.a" $S/t/ $ST/$arm/t/
  rsync -a --exclude="*.o" --exclude="*.a" $S/lib/ $ST/$arm/lib/
done
sha256sum $ST/*/perl.dom $ST/smoke.pl $ST/capstone-vexec $ST/capstone_vm.ko
cat > $ST/gate.sh <<G
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo \$u; }
report() { echo "SA_BEGIN:\$1:\$2"; cat /tmp/sa.out; echo "SA_STDERR:\$1:\$2"; cat /tmp/sa.err
  echo "SA_END:\$1:\$2:\$3:\$4:\$5:\$(sha256sum /tmp/sa.out | cut -d ' ' -f 1)"; }
for arm in stock protected; do
  echo "SA_IMAGE:\$arm:\$(sha256sum /mnt/vm/\$arm/perl.dom | cut -d ' ' -f 1)"
  mkdir -p /tmp/ctl-\$arm; cd /tmp/ctl-\$arm
  s=\$(up)
  /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom -e '\$x = *foo; *x = \$x; print "done\\n"' > /tmp/sa.out 2> /tmp/sa.err
  rc=\$?; e=\$(up); report \$arm control \$rc \$s \$e
  mkdir -p /tmp/\$arm; cp -a /mnt/vm/\$arm/t /tmp/\$arm/t; ln -s /mnt/vm/\$arm/lib /tmp/\$arm/lib
  for t in $TESTS; do
    cd /tmp/\$arm/t
    s=\$(up)
    PERL5LIB=/tmp/\$arm/lib /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom \$t > /tmp/sa.out 2> /tmp/sa.err
    rc=\$?; e=\$(up); report \$arm \$t \$rc \$s \$e
  done
  mkdir -p /tmp/smoke-\$arm; cp /mnt/vm/smoke.pl /tmp/smoke-\$arm/smoke.pl; cd /tmp/smoke-\$arm
  s=\$(up)
  CAPSTONE_PERL_SMOKE_OBJECTS=2000 /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom smoke.pl /tmp/smoke-\$arm > /tmp/sa.out 2> /tmp/sa.err
  rc=\$?; e=\$(up); report \$arm smoke \$rc \$s \$e
done
cd /mnt/vm
rmmod capstone_vm
echo SA_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
W=$M/guest-sa-$(date +%Y%m%d-%H%M%S)
s=$(date +%s)
python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage $ST --work $W --timeout 3600 --disk-mib 2048
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s work=$W"
echo RUN-DONE
