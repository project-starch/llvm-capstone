#!/bin/sh
# vp2 boot r13 variant of run.py's guest script: S-24's two-sided diagnostic. `vm` runs twice: with resolve_dirty off
# (the loop, and the PTE printed at the bound) and on (the fix: the page faulted in for write before the pin), threads
# again between them as the post-flip window's must-fire. The other returning steps follow with resolve_dirty on. No
# sparse steps and no kit applications (S-25). The module's lines are read from the ring buffer with dmesg.
cd /test-domains/vm || exit 1
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko || exit 1   # board: the contract rung may have loaded it, and the board kernel cannot unload
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/ok.out 2>&1
echo VM_EXIT:normal:$?
cat /tmp/ok.out
./capstone-vexec contract.dom threads > /tmp/threads.out 2>&1
echo VM_EXIT:threads:$?
cat /tmp/threads.out
echo 1 > /sys/module/capstone_vm/parameters/resolve_pte; echo RESOLVE_PTE:$(cat /sys/module/capstone_vm/parameters/resolve_pte):before-vm-dirty0
echo RESOLVE_DIRTY:$(cat /sys/module/capstone_vm/parameters/resolve_dirty):before-vm-dirty0
./capstone-vexec contract.dom vm 2>&1
echo VM_EXIT:vm-dirty0:$?
dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-VM0: /'; echo KMSG-AFTER-VM0-END
echo 1 > /sys/module/capstone_vm/parameters/resolve_dirty; echo RESOLVE_DIRTY:$(cat /sys/module/capstone_vm/parameters/resolve_dirty):before-threads2
./capstone-vexec contract.dom threads > /tmp/threads2.out 2>&1
echo VM_EXIT:threads2:$?
./capstone-vexec contract.dom vm 2>&1
echo VM_EXIT:vm-dirty1:$?
dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-VM1: /'; echo KMSG-AFTER-VM1-END
for mode in nested nested-stale heap-threads stale bounds retired vm-ro-write vm-none-read vm-guard vm-padding; do
  ./capstone-vexec contract.dom "$mode" 2>&1
  echo VM_EXIT:$mode:$?
done
./capstone-vexec contract.dom spin > /tmp/spin.out 2>&1 & spin=$!
# QEMU startup includes Linux image loading and the first delegated rounds;
# allow the child to publish its readiness before testing signal termination.
sleep 5
kill -TERM "$spin" 2>/dev/null
wait "$spin"; echo VM_EXIT:preempt:$?
cat /tmp/spin.out
./capstone-vexec contract.dom ok argument > /tmp/one.out 2>&1 & one=$!
./capstone-vexec contract.dom ok argument > /tmp/two.out 2>&1 & two=$!
wait "$one"; echo VM_EXIT:parallel_one:$?
wait "$two"; echo VM_EXIT:parallel_two:$?
cat /tmp/one.out /tmp/two.out
# vp2 boot r12: the module's lines after the returning steps (the compiler lane predicts `vma` site lines for
# vm-ro-write and vm-none-read: a second positive control), then a count of the buffer as its window.
dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-RETURNING: /'; echo KMSG-AFTER-RETURNING-END lines=$(dmesg | wc -l)
# vp2 boot r12: mruby and sqlite are out (r10's sqlite_write and r11's mruby wedge the core alike).
if [ -f perl.dom ]; then
  ./capstone-vexec perl.dom -e 'my $sum = 0; $sum += $_ for 1..100; open my $w, ">", "/tmp/virtual-perl" or die $!; print $w "virtual"; close $w; open my $r, "<", "/tmp/virtual-perl" or die $!; my $text = <$r>; close $r; print "PERL_VIRTUAL_OK sum=$sum file=$text\n"' > /tmp/perl.out 2>&1
  echo VM_EXIT:perl:$?
  cat /tmp/perl.out
  if [ -f perl-smoke.pl ]; then
    mkdir -p /tmp/virtual-perl-files
    cp /test-domains/vm/perl-smoke.pl /tmp/virtual-perl-files/smoke.pl
    CAPSTONE_PERL_SMOKE_OBJECTS=2000 ./capstone-vexec perl.dom /test-domains/vm/perl-smoke.pl /tmp/virtual-perl-files > /tmp/perl-smoke.out 2>&1
    echo VM_EXIT:perl_smoke:$?
    cat /tmp/perl-smoke.out
  fi
fi
# vp2 boot r13: no sparse steps (S-25: a capability store as a page's first touch wedges the core on bf3db3175).
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
if [ -f legacy.dom ]; then
  ./capstone-vexec legacy.dom ok argument > /tmp/legacy.out 2>&1
  echo VM_EXIT:legacy:$?
  cat /tmp/legacy.out
fi
echo VIRTUAL_RUNTIME_DONE
