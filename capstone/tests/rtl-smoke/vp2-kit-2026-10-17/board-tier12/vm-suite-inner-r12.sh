#!/bin/sh
# vp2 boot r12 variant of run.py's guest script: mruby and sqlite out (both wedge the core: r10, r11); the module's lines
# read from the ring buffer with dmesg (the console shows only KERN_EMERG during rungs); the fence pair in the order
# fence=1 sparse-stale, fence=0 sparse-stale, then sparse-churn LAST.
cd /test-domains/vm || exit 1
grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko || exit 1   # board: the contract rung may have loaded it, and the board kernel cannot unload
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/ok.out 2>&1
echo VM_EXIT:normal:$?
cat /tmp/ok.out
./capstone-vexec contract.dom threads > /tmp/threads.out 2>&1
echo VM_EXIT:threads:$?
cat /tmp/threads.out
for mode in vm nested nested-stale heap-threads stale bounds retired vm-ro-write vm-none-read vm-guard vm-padding; do
  ./capstone-vexec contract.dom "$mode" 2>&1
  echo VM_EXIT:$mode:$?
  # vp2 boot r12: the module's lines so far, from the ring buffer (no -c: busybox's -c is untested here). After vm the
  # window's must-fire is threads' "thread create refused", read in the same call.
  if [ "$mode" = vm ]; then dmesg | grep -a "capstone_vm:" | sed 's/^/KMSG-AFTER-VM: /'; echo KMSG-AFTER-VM-END; fi
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
# vp2 boot r9 (2026-10-18): sparse-churn moved LAST, after every other step and before rmmod: it wedged the core on
# boot r8, and a wedge takes every later step with it (the ordering rule). run.py's own order is otherwise unchanged.
# vp2 boot r11: the sparse pair with fence_before_entry=1 (a fence rw,rw before every CSRUNV), then sparse-stale with
# fence_before_entry=0 LAST, the expected wedge and the control that the condition still exists in this boot. The
# parameter is flipped through sysfs (the board kernel cannot rmmod); each flip echoes the value read back.
# vp2 boot r12: the compiler lane's order. The matched pair first (fence=1 then fence=0 sparse-stale), sparse-churn
# LAST: it is a second expected wedge, and in r11's order it sat between the two halves of the pair.
echo 1 > /sys/module/capstone_vm/parameters/fence_before_entry; echo FENCE_BEFORE_ENTRY:$(cat /sys/module/capstone_vm/parameters/fence_before_entry):before-sparse-stale-fence1
./capstone-vexec contract.dom sparse-stale 2>&1
echo VM_EXIT:sparse-stale-fence1:$?
echo 0 > /sys/module/capstone_vm/parameters/fence_before_entry; echo FENCE_BEFORE_ENTRY:$(cat /sys/module/capstone_vm/parameters/fence_before_entry):before-sparse-stale-fence0
./capstone-vexec contract.dom sparse-stale 2>&1
echo VM_EXIT:sparse-stale-fence0:$?
./capstone-vexec contract.dom sparse-churn 2>&1
echo VM_EXIT:sparse-churn-fence0:$?
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
if [ -f legacy.dom ]; then
  ./capstone-vexec legacy.dom ok argument > /tmp/legacy.out 2>&1
  echo VM_EXIT:legacy:$?
  cat /tmp/legacy.out
fi
echo VIRTUAL_RUNTIME_DONE
