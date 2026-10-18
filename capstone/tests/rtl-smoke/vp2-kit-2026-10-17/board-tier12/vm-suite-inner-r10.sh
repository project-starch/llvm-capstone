#!/bin/sh
# vp2 boot r10 variant of run.py's guest script: sparse-stale and sparse-churn last (see below). Everything else as run.py.
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
./capstone-vexec sqlite3.dom /tmp/virtual.db "CREATE TABLE t(n); WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<100) INSERT INTO t SELECT n FROM x; SELECT 'SQLITE_SUM='||sum(n) FROM t;" > /tmp/sqlite.out 2>&1
echo VM_EXIT:sqlite_write:$?
cat /tmp/sqlite.out
./capstone-vexec sqlite3.dom /tmp/virtual.db "SELECT 'SQLITE_PERSIST='||count(*) FROM t;" > /tmp/sqlite.out 2>&1
echo VM_EXIT:sqlite_read:$?
cat /tmp/sqlite.out
./capstone-vexec mruby.dom -e 'puts "MRUBY_SUM=#{(1..100).inject(0){|s,n| s+n}}"; File.open("/tmp/virtual-ruby", "w"){|f| f.write("virtual")}; puts "MRUBY_FILE=#{File.read("/tmp/virtual-ruby")}"' > /tmp/mruby.out 2>&1
echo VM_EXIT:mruby:$?
cat /tmp/mruby.out
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
# vp2 boot r10: sparse-stale joins sparse-churn at the end; it wedged the core on boot r9 at the same point as
# sparse-churn on r8 (the shared sparse path), so both go after every other step.
./capstone-vexec contract.dom sparse-stale 2>&1
echo VM_EXIT:sparse-stale:$?
./capstone-vexec contract.dom sparse-churn 2>&1
echo VM_EXIT:sparse-churn:$?
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
if [ -f legacy.dom ]; then
  ./capstone-vexec legacy.dom ok argument > /tmp/legacy.out 2>&1
  echo VM_EXIT:legacy:$?
  cat /tmp/legacy.out
fi
echo VIRTUAL_RUNTIME_DONE
