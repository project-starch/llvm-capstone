#!/usr/bin/env bash
# Same-answers run for Perl 5.36.3, as ../2026-10-11-same-answers/run.sh and smoke2k.sh in one boot: the native
# reference on the host, then ONE staged boot running t/base (9 files), smoke.pl as it is, and smoke.pl with
# CAPSTONE_PERL_SMOKE_OBJECTS=2000, on the stock and the protected virtual image.
#   usage: run.sh STOCK_ROOT PROTECTED_ROOT KIT WORK   (the two PERLD_ROOTs of build.sh; KIT: qemu-system-riscv64,
#   images/, adapter/). CAPSTONE_REPO_ROOT, CAPSTONE_QEMU_LOCK and CAPSTONE_QEMU_CPU_EXTRA as the caller set them.
set -u
STOCK=$1 PROT=$2 K=$3 SA=$4
R=${CAPSTONE_REPO_ROOT:?} V=$R/capstone/runtime/virtual
TESTS="base/cond.t base/if.t base/lex.t base/num.t base/pat.t base/rs.t base/term.t base/translate.t base/while.t"
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl
mkdir -p $SA

# ---- native reference: installed native perl, a copy of its own build tree t/, ../lib = its build lib
NR=$SA/native-run; rm -rf $NR; mkdir -p $NR/out
NS=$STOCK/src/native-perl-5.36.3; NP=$STOCK/native/bin/perl
cp -a $NS/t $NR/t; ln -s $NS/lib $NR/lib
for t in $TESTS; do
  n=$(echo $t | tr / _)
  (cd $NR/t && PERL5LIB=$NR/lib $NP $t > $NR/out/$n.stdout 2> $NR/out/$n.stderr); rc=$?
  echo "NATIVE $t rc=$rc sha256=$(sha256sum < $NR/out/$n.stdout | cut -d' ' -f1)"
done
mkdir -p $NR/smoke; cp $SMOKE $NR/smoke/smoke.pl
(cd $NR/smoke && $NP smoke.pl $NR/smoke > $NR/out/smoke.stdout 2> $NR/out/smoke.stderr); rc=$?
echo "NATIVE smoke rc=$rc sha256=$(sha256sum < $NR/out/smoke.stdout | cut -d' ' -f1)"
(cd $NR/smoke && CAPSTONE_PERL_SMOKE_OBJECTS=2000 $NP smoke.pl $NR/smoke > $NR/out/smoke2k.stdout 2> $NR/out/smoke2k.stderr); rc=$?
echo "NATIVE smoke2k rc=$rc sha256=$(sha256sum < $NR/out/smoke2k.stdout | cut -d' ' -f1)"

# ---- stage for the boot
ST=$SA/stage; rm -rf $ST; mkdir -p $ST
cp $K/adapter/capstone-vexec $ST/; cp $K/adapter/module/capstone_vm.ko $ST/
cp $SMOKE $ST/smoke.pl
for arm in stock protected; do
  [ $arm = stock ] && S=$STOCK/src/perl-5.36.3 || S=$PROT/src/perl-5.36.3
  mkdir -p $ST/$arm
  cp $S/perl $ST/$arm/perl.dom
  rsync -a --exclude="*.o" --exclude="*.a" $S/t/ $ST/$arm/t/
  rsync -a --exclude="*.o" --exclude="*.a" $S/lib/ $ST/$arm/lib/
done
sha256sum $ST/*/perl.dom $ST/smoke.pl $ST/capstone-vexec $ST/capstone_vm.ko $K/qemu-system-riscv64 $K/images/*
cat > $ST/gate.sh <<G
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo \$u; }
for arm in stock protected; do
  mkdir -p /tmp/\$arm; cp -a /mnt/vm/\$arm/t /tmp/\$arm/t; ln -s /mnt/vm/\$arm/lib /tmp/\$arm/lib
  for t in $TESTS; do
    cd /tmp/\$arm/t
    s=\$(up)
    PERL5LIB=/tmp/\$arm/lib /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom \$t > /tmp/sa.out 2> /tmp/sa.err
    rc=\$?
    e=\$(up)
    h=\$(sha256sum /tmp/sa.out | cut -d " " -f 1)
    echo "SA_BEGIN:\$arm:\$t"
    cat /tmp/sa.out
    echo "SA_STDERR:\$arm:\$t"
    cat /tmp/sa.err
    echo "SA_END:\$arm:\$t:\$rc:\$s:\$e:\$h"
  done
  for smoke in smoke smoke2k; do
    mkdir -p /tmp/\$smoke-\$arm; cp /mnt/vm/smoke.pl /tmp/\$smoke-\$arm/smoke.pl; cd /tmp/\$smoke-\$arm
    s=\$(up)
    if [ \$smoke = smoke2k ]; then
      CAPSTONE_PERL_SMOKE_OBJECTS=2000 /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom smoke.pl /tmp/\$smoke-\$arm > /tmp/sa.out 2> /tmp/sa.err
    else
      /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom smoke.pl /tmp/\$smoke-\$arm > /tmp/sa.out 2> /tmp/sa.err
    fi
    rc=\$?
    e=\$(up)
    h=\$(sha256sum /tmp/sa.out | cut -d " " -f 1)
    echo "SA_BEGIN:\$arm:\$smoke"
    cat /tmp/sa.out
    echo "SA_STDERR:\$arm:\$smoke"
    cat /tmp/sa.err
    echo "SA_END:\$arm:\$smoke:\$rc:\$s:\$e:\$h"
  done
done
cd /mnt/vm
rmmod capstone_vm
echo SA_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
du -sm $ST
s=$(date +%s)
python3 $V/run-staged.py --qemu $K/qemu-system-riscv64 --images $K/images --stage $ST --work $SA/guest --timeout 3600 --disk-mib 2048
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s"
echo RUN-DONE
