#!/usr/bin/env bash
# Same-answers run for Perl 5.36.3: native reference on the host, then ONE boot running
# t/base (9 files) and smoke.pl on the stock and the protected virtual image.
set -u
K=$HOME/pg-virtual-kit SA=$HOME/sa-perl R=$SA/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=$SA/qemu.lock
TESTS="base/cond.t base/if.t base/lex.t base/num.t base/pat.t base/rs.t base/term.t base/translate.t base/while.t"
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl

# ---- native reference: installed native perl, a copy of its own build tree t/, ../lib = its build lib
NR=$SA/native-run; rm -rf $NR; mkdir -p $NR/out
NS=$SA/perl-stock/src/native-perl-5.36.3; NP=$SA/perl-stock/native/bin/perl
cp -a $NS/t $NR/t; ln -s $NS/lib $NR/lib
for t in $TESTS; do
  n=$(echo $t | tr / _); s=$(date +%s.%N)
  (cd $NR/t && PERL5LIB=$NR/lib $NP $t > $NR/out/$n.stdout 2> $NR/out/$n.stderr); rc=$?
  echo "NATIVE $t rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc)"
done
mkdir -p $NR/smoke; cp $SMOKE $NR/smoke/smoke.pl
s=$(date +%s.%N)
(cd $NR/smoke && $NP smoke.pl $NR/smoke > $NR/out/smoke.stdout 2> $NR/out/smoke.stderr); rc=$?
echo "NATIVE smoke rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc)"

# ---- stage for the boot
ST=$SA/stage; rm -rf $ST; mkdir -p $ST
cp $K/kit-sublet/adapter/capstone-vexec $ST/; cp $K/kit-sublet/adapter/module/capstone_vm.ko $ST/
cp $SMOKE $ST/smoke.pl
for arm in stock protected; do
  S=$SA/perl-$arm/src/perl-5.36.3; mkdir -p $ST/$arm
  cp $S/perl $ST/$arm/perl.dom
  rsync -a --exclude="*.o" --exclude="*.a" $S/t/ $ST/$arm/t/
  rsync -a --exclude="*.o" --exclude="*.a" $S/lib/ $ST/$arm/lib/
done
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
    echo "SA_BEGIN:\$arm:\$t"
    cat /tmp/sa.out
    echo "SA_STDERR:\$arm:\$t"
    cat /tmp/sa.err
    echo "SA_END:\$arm:\$t:\$rc:\$s:\$e:\$(sha256sum < /tmp/sa.out | cut -d\" \" -f1)"
  done
  mkdir -p /tmp/smoke-\$arm; cp /mnt/vm/smoke.pl /tmp/smoke-\$arm/smoke.pl; cd /tmp/smoke-\$arm
  s=\$(up)
  /mnt/vm/capstone-vexec /mnt/vm/\$arm/perl.dom smoke.pl /tmp/smoke-\$arm > /tmp/sa.out 2> /tmp/sa.err
  rc=\$?
  e=\$(up)
  echo "SA_BEGIN:\$arm:smoke"
  cat /tmp/sa.out
  echo "SA_STDERR:\$arm:smoke"
  cat /tmp/sa.err
  echo "SA_END:\$arm:smoke:\$rc:\$s:\$e:\$(sha256sum < /tmp/sa.out | cut -d\" \" -f1)"
done
cd /mnt/vm
rmmod capstone_vm
echo SA_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
chmod +x $ST/gate.sh
du -sm $ST
W=$SA/guest-$(date +%Y%m%d-%H%M%S)
s=$(date +%s)
python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage $ST --work $W --timeout 3600 --disk-mib 2048
echo "STAGED rc=$? wall=$(( $(date +%s) - s ))s work=$W"
echo RUN-DONE
