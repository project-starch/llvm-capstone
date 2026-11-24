#!/usr/bin/env bash
# smoke-sizes.sh STOCK_ROOT PROTECTED_ROOT KIT WORK: smoke.pl on the protected and the stock image in one staged boot: 2,000 objects, then 6,000,
# then the script's default 20,000, each bounded by BOUND seconds of guest time and timed.
R=${CAPSTONE_REPO_ROOT:?}; T=$R/capstone
STOCK=$1 PROT=$2 K=$3 SA=$4; ST=$SA/stage; rm -rf $SA; mkdir -p $ST
cp $K/adapter/capstone-vexec $K/adapter/module/capstone_vm.ko $T/ports/perl/musl/scripts/smoke.pl $ST/
cp $STOCK/src/perl-5.36.3/perl $ST/perl-stock.dom; cp $PROT/src/perl-5.36.3/perl $ST/perl-protected.dom
cat > $ST/gate.sh <<'G'
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo $u; }
for n in 2000 6000 20000; do
  for arm in protected stock; do
    d=/tmp/smoke-$arm-$n; mkdir -p $d; cp /mnt/vm/smoke.pl $d/smoke.pl; cd $d
    s=$(up)
    CAPSTONE_PERL_SMOKE_OBJECTS=$n /mnt/vm/capstone-vexec /mnt/vm/perl-$arm.dom smoke.pl $d > $d/out 2> $d/err & pid=$!
    ( sleep BOUND; kill $pid 2>/dev/null ) & dog=$!
    wait $pid; rc=$?; kill $dog 2>/dev/null
    e=$(up)
    echo "SA_BEGIN:$arm:smoke$n"; cat $d/out; echo "SA_STDERR:$arm:smoke$n"; cat $d/err
    echo "SA_END:$arm:smoke$n:$rc:$s:$e:$(sha256sum $d/out | cut -d ' ' -f 1)"
  done
done
cd /mnt/vm
rmmod capstone_vm
echo SA_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
G
sed -i "s/BOUND/${BOUND:-900}/" $ST/gate.sh; chmod +x $ST/gate.sh
python3 $T/runtime/virtual/run-staged.py --qemu $K/qemu-system-riscv64 --images $K/images --stage $ST --work $SA/guest --timeout 4200 --disk-mib 256
echo "STAGED rc=$?"
