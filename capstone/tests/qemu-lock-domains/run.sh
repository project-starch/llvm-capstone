#!/usr/bin/env bash
# The QEMU lock-domain controls (2026-10-21), on a qkit7-style KIT:  run.sh KIT OUT
# Holds the REAL global lock ($CAPSTONE_QEMU_LOCK) EXCLUSIVE for the whole test, so no other QEMU session runs beside
# it and its timings are its own; inside that hold the reader-writer semantics are exercised on PRIVATE lock files.
#  (a) two snapshot-class run-staged.py sessions, domains A and B at once: their grant-to-end intervals must OVERLAP,
#      each serial log must carry only its own marker, and the shared rootfs.ext2 must hash the same before and after.
#  (b) the positive control for (a)'s detector: the same pair on a WRITABLE scratch copy of the rootfs (run-staged.py
#      with -snapshot removed and root mounted rw), each writing /root/marker-<id>: the shared file must end up holding
#      BOTH markers (or fail e2fsck) -- if it cannot show sharing, (a)'s clean reading proves nothing.
#  (c) an EXCLUSIVE holder of the global lock must block a domain session (its grant waits about the hold), and a
#      running domain session must block an exclusive taker (flock -x -w 10 times out), timed and printed.
#  (d) the turnstile: an exclusive waiter queued behind a running domain session is granted when it ends and BEFORE a
#      domain session that arrived after the waiter; (d') without the turnstile that later session gets in first.
# Exit 0 only when all three read as required. RESULT in OUT.
set -u
main() {
  K=${1:?kit}; O=${2:?out}; H=$(cd "$(dirname "$0")/../../runtime/virtual" && pwd); RS=$H/run-staged.py
  [ -e "$O" ] && { echo "$O exists"; return 2; }; mkdir -p "$O"; R=$O/RESULT
  : "${CAPSTONE_QEMU_LOCK:?}"; exec 9>>"$CAPSTONE_QEMU_LOCK"; flock -x 9; echo "real global lock held $(date +%s)" > $R
  export CAPSTONE_QEMU_LOCK=$O/locks/global.lock; mkdir -p $O/locks
  dom() { case $1 in A) echo "$O/locks/domain-A.lock 28-29,60-61";; B) echo "$O/locks/domain-B.lock 30-31,62-63";; esac; }
  stage() {  # dir id body
    mkdir -p $1; printf '#!/bin/sh\necho MARK-%s-BEGIN\n%s\necho MARK-%s-END\necho VIRTUAL_STAGED_DONE\n' "$2" "$3" "$2" > $1/gate.sh; chmod +x $1/gate.sh; }
  session() {  # id domain runner work stage
    set -- $1 $(dom $2) $3 $4 $5
    CAPSTONE_QEMU_DOMAIN_LOCK=$2 taskset -c $3 python3 $4 --qemu $K/qemu-system-riscv64 --images ${IMG:-$K/images} --stage $6 --work $5 --timeout 300 > $5.out 2>&1
    echo "$1 rc=$? end $(date +%s.%N | cut -c1-12)" >> $O/ends; }
  granted() { sed -n 's/.*granted \([0-9.]*\) (waited \([0-9.]*\) s).*/\1 \2/p' $1 | head -1; }
  rf=$K/images/rootfs.ext2; h0=$(sha256sum $rf | cut -c1-16)

  # (a)
  stage $O/sa A 'sleep 40'; stage $O/sb B 'sleep 40'
  session A A $RS $O/a-A $O/sa & session B B $RS $O/a-B $O/sb & wait
  read ga wa <<<"$(granted $O/a-A.out)"; read gb wb <<<"$(granted $O/a-B.out)"
  ea=$(awk '$1=="A"{print $4}' $O/ends); eb=$(awk '$1=="B"{print $4}' $O/ends)
  ov=$(python3 -c "print(int($ga < $eb and $gb < $ea))" 2>/dev/null || echo 0)
  own=$( grep -c 'MARK-A-' $O/a-A/serial.log ):$( grep -c 'MARK-B-' $O/a-A/serial.log ):$( grep -c 'MARK-B-' $O/a-B/serial.log ):$( grep -c 'MARK-A-' $O/a-B/serial.log )
  h1=$(sha256sum $rf | cut -c1-16)
  aok=0; [ "$ov" = 1 ] && [ "$own" = "2:0:2:0" ] && [ "$h0" = "$h1" ] && aok=1
  echo "(a) granted A $ga B $gb, ends A $ea B $eb, overlap=$ov; markers own:foreign A=${own%%:*}... ($own as A-own:A-foreign:B-own:B-foreign); rootfs $h0 -> $h1; PASS=$aok" >> $R

  # (b)
  cp --reflink=auto $rf $O/scratch.ext2; chmod u+w $O/scratch.ext2; mkdir -p $O/bimg; for f in Image fw_jump.elf; do ln -s $K/images/$f $O/bimg/$f; done; ln -s $O/scratch.ext2 $O/bimg/rootfs.ext2
  python3 - $RS $O/run-staged-writable.py <<'PY'
import sys
t = open(sys.argv[1]).read(); n = 0
for o, r in (("'-serial', 'stdio', '-snapshot',", "'-serial', 'stdio',"), ("'root=/dev/vda ro'", "'root=/dev/vda rw'")):
    assert t.count(o) == 1, o; t = t.replace(o, r); n += 1
open(sys.argv[2], "w").write(t); print("writable variant: substitutions", n)
PY
  cp $H/qemu_lock.py $O/; e0=$(/sbin/e2fsck -fn $O/scratch.ext2 > /dev/null 2>&1; echo $?)
  stage $O/wa A 'echo A > /root/marker-A; sync; sleep 20; ls /root; sync'; stage $O/wb B 'echo B > /root/marker-B; sync; sleep 20; ls /root; sync'
  IMG=$O/bimg session A A $O/run-staged-writable.py $O/b-A $O/wa & IMG=$O/bimg session B B $O/run-staged-writable.py $O/b-B $O/wb & wait
  ls_root=$(/sbin/debugfs -R 'ls /root' $O/scratch.ext2 2>/dev/null | tr -s ' \n' ' '); e1=$(/sbin/e2fsck -fn $O/scratch.ext2 > $O/e2fsck-b.txt 2>&1; echo $?)
  both=0; [[ "$ls_root" == *marker-A* && "$ls_root" == *marker-B* ]] && both=1
  bok=0; { [ $both = 1 ] || [ "$e1" != 0 ]; } && bok=1
  echo "(b) scratch rootfs after: /root holds [$ls_root]; both markers in the shared file=$both; e2fsck -fn before rc=$e0 after rc=$e1; FIRES=$bok" >> $R

  # (c)
  stage $O/sc1 C1 'true'; ( flock -x $CAPSTONE_QEMU_LOCK sleep 30 ) & hp=$!; sleep 2
  session C1 A $RS $O/c-1 $O/sc1; wait $hp; read g1 w1 <<<"$(granted $O/c-1.out)"
  c1=0; python3 -c "import sys; sys.exit(0 if $w1 >= 25 else 1)" && c1=1
  echo "(c1) exclusive holder for 30 s (from 2 s before): the domain session's grant waited $w1 s; BLOCKED=$c1" >> $R
  stage $O/sc2 C2 'sleep 40'; session C2 B $RS $O/c-2 $O/sc2 & sp=$!
  until grep -q granted $O/c-2.out 2>/dev/null; do sleep 1; done; sleep 2
  t0=$(date +%s.%N); flock -x -w 10 $CAPSTONE_QEMU_LOCK true; xr=$?; t1=$(date +%s.%N); wait $sp
  t2=$(date +%s.%N); flock -x -w 60 $CAPSTONE_QEMU_LOCK true; xr2=$?; t3=$(date +%s.%N)
  c2=0; [ $xr = 1 ] && [ $xr2 = 0 ] && c2=1
  echo "(c2) during a domain session: flock -x -w 10 rc=$xr after $(python3 -c "print(round($t1-$t0,1))") s; after it ended: rc=$xr2 after $(python3 -c "print(round($t3-$t2,1))") s; BLOCKED=$c2" >> $R
  # (d) the turnstile: a domain session D1 runs; an exclusive waiter X queues behind it THROUGH the turnstile; a second
  # domain session D2 (the other domain) starts after X queued. X must be granted when D1 ends and BEFORE D2.
  # (d') its negative twin: the same with a waiter that takes the global lock WITHOUT the turnstile (as un-migrated
  # callers do): D2 then passes the free turnstile and is granted while D1 still runs, before X -- if it is not, (d)
  # cannot tell the turnstile from luck.
  turn() {  # tag D1-stage D2-stage with-turnstile(1|0)
    local tag=$1; stage $O/$tag-s1 ${tag}1 'sleep 30'; stage $O/$tag-s2 ${tag}2 'true'
    session ${tag}1 A $RS $O/$tag-1 $O/$tag-s1 & local p1=$!
    until grep -q granted $O/$tag-1.out 2>/dev/null; do sleep 1; done
    if [ $4 = 1 ]; then ( exec 8>>"$CAPSTONE_QEMU_LOCK.turnstile" 9>>"$CAPSTONE_QEMU_LOCK"; flock -x 8; flock -x 9; date +%s.%N | cut -c1-14 > $O/$tag-x; flock -u 8; exec 8>&-; sleep 5 ) & local px=$!
    else ( exec 9>>"$CAPSTONE_QEMU_LOCK"; flock -x 9; date +%s.%N | cut -c1-14 > $O/$tag-x; sleep 5 ) & local px=$!; fi
    sleep 3; session ${tag}2 B $RS $O/$tag-2 $O/$tag-s2 & local p2=$!
    wait $p1 $px $p2
    read g1 w <<<"$(granted $O/$tag-1.out)"; read g2 w <<<"$(granted $O/$tag-2.out)"; gx=$(cat $O/$tag-x)
    e1=$(awk -v t=${tag}1 '$1==t{print $4}' $O/ends)
    echo "$g1 $gx $g2 $e1"
  }
  read g1 gx g2 e1 <<<"$(turn d 0 0 1)"
  d=0; python3 -c "import sys; sys.exit(0 if $gx >= $e1 - 2 and $gx < $g2 else 1)" && d=1
  echo "(d) with the turnstile: D1 granted $g1 ended $e1; exclusive X granted $gx; D2 granted $g2 -> X after D1 and before D2=$d" >> $R
  read h1 hx h2 f1 <<<"$(turn dn 0 0 0)"
  dn=0; python3 -c "import sys; sys.exit(0 if $h2 < $hx else 1)" && dn=1
  echo "(d') negative twin, waiter WITHOUT the turnstile: D1 granted $h1 ended $f1; X granted $hx; D2 granted $h2 -> D2 before X (the starvation the turnstile prevents)=$dn" >> $R
  ok=0; [ $aok = 1 ] && [ $bok = 1 ] && [ $c1 = 1 ] && [ $c2 = 1 ] && [ $d = 1 ] && [ $dn = 1 ] && ok=1
  echo "ALL=$ok $(date +%s)" >> $R; cat $R; [ $ok = 1 ]
}
main "$@"; exit
