#!/bin/sh
# vp2 boot r11 variant of run.py's guest script: sqlite deferred; the sparse pair with the entry fence on, then off (see below).
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
# vp2 boot r11: mruby before sqlite (sqlite_write wedged the core on boot r10).
./capstone-vexec mruby.dom -e 'puts "MRUBY_SUM=#{(1..100).inject(0){|s,n| s+n}}"; File.open("/tmp/virtual-ruby", "w"){|f| f.write("virtual")}; puts "MRUBY_FILE=#{File.read("/tmp/virtual-ruby")}"' > /tmp/mruby.out 2>&1
echo VM_EXIT:mruby:$?
cat /tmp/mruby.out
# vp2 boot r11: sqlite deferred to r12 (r10's sqlite_write wedge is a second unknown; it would cost the fence=0 control).
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
echo 1 > /sys/module/capstone_vm/parameters/fence_before_entry; echo FENCE_BEFORE_ENTRY:$(cat /sys/module/capstone_vm/parameters/fence_before_entry)
./capstone-vexec contract.dom sparse-stale 2>&1
echo VM_EXIT:sparse-stale-fence1:$?
./capstone-vexec contract.dom sparse-churn 2>&1
echo VM_EXIT:sparse-churn-fence1:$?
echo 0 > /sys/module/capstone_vm/parameters/fence_before_entry; echo FENCE_BEFORE_ENTRY:$(cat /sys/module/capstone_vm/parameters/fence_before_entry)
./capstone-vexec contract.dom sparse-stale 2>&1
echo VM_EXIT:sparse-stale-fence0:$?
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
if [ -f legacy.dom ]; then
  ./capstone-vexec legacy.dom ok argument > /tmp/legacy.out 2>&1
  echo VM_EXIT:legacy:$?
  cat /tmp/legacy.out
fi
echo VIRTUAL_RUNTIME_DONE
