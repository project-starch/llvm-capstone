#!/usr/bin/env bash
# Control (b) on its own, the hazard FORCED: QEMU's image locking refuses a second writer of one image ("Failed to get
# 'write' lock"), so the shared-writable pair is run with file.locking=off on the scratch rootfs drive. Interference is
# any of: a session listing the OTHER session's marker; a session's own marker missing from the final file although its
# console listed it (a lost update); e2fsck -fn failing afterwards. Must FIRE. Holds the real global lock throughout.
#   run-b.sh KIT OUT
set -u
main() {
  K=${1:?kit}; O=${2:?out}; H=$(cd "$(dirname "$0")/../../runtime/virtual" && pwd); RS=$H/run-staged.py
  [ -e "$O" ] && { echo "$O exists"; return 2; }; mkdir -p "$O/locks"; R=$O/RESULT
  : "${CAPSTONE_QEMU_LOCK:?}"; exec 9>>"$CAPSTONE_QEMU_LOCK"; flock -x 9; echo "real global lock held $(date +%s)" > $R
  export CAPSTONE_QEMU_LOCK=$O/locks/global.lock
  cp --reflink=auto $K/images/rootfs.ext2 $O/scratch.ext2; chmod u+w $O/scratch.ext2; mkdir -p $O/bimg
  for f in Image fw_jump.elf; do ln -s $K/images/$f $O/bimg/$f; done; ln -s $O/scratch.ext2 $O/bimg/rootfs.ext2
  python3 - $RS $O/run-staged-writable.py <<'PY'
import sys
t = open(sys.argv[1]).read()
for o, r in (("'-serial', 'stdio', '-snapshot',", "'-serial', 'stdio',"), ("'root=/dev/vda ro'", "'root=/dev/vda rw'"),
             ("rootfs.ext2\"},format=raw,id=hd0'", "rootfs.ext2\"},format=raw,id=hd0,file.locking=off'")):
    assert t.count(o) == 1, o; t = t.replace(o, r)
open(sys.argv[2], "w").write(t); print("writable variant, locking off: 3 substitutions")
PY
  cp $H/qemu_lock.py $O/
  for x in A B; do mkdir -p $O/s$x; printf '#!/bin/sh\necho MARK-%s-BEGIN\necho %s > /root/marker-%s; sync\nsleep 20\necho LS-%s: $(ls /root | tr "\\n" " ")\nsync\necho MARK-%s-END\necho VIRTUAL_STAGED_DONE\n' $x $x $x $x $x > $O/s$x/gate.sh; chmod +x $O/s$x/gate.sh; done
  for x in A B; do cpus=28-29,60-61; [ $x = B ] && cpus=30-31,62-63
    ( CAPSTONE_QEMU_DOMAIN_LOCK=$O/locks/domain-$x.lock taskset -c $cpus python3 $O/run-staged-writable.py --qemu $K/qemu-system-riscv64 --images $O/bimg --stage $O/s$x --work $O/b-$x --timeout 300 > $O/b-$x.out 2>&1 ) & done; wait
  la=$(tr -d '\r' < $O/b-A/serial.log | grep -a '^LS-A:'); lb=$(tr -d '\r' < $O/b-B/serial.log | grep -a '^LS-B:')
  fin=$(/sbin/debugfs -R 'ls /root' $O/scratch.ext2 2>/dev/null | tr -s ' \n' ' '); e=$(/sbin/e2fsck -fn $O/scratch.ext2 > $O/e2fsck.txt 2>&1; echo $?)
  foreign=0; [[ "$la" == *marker-B* || "$lb" == *marker-A* ]] && foreign=1
  lost=0; { [[ "$la" == *marker-A* && "$fin" != *marker-A* ]] || [[ "$lb" == *marker-B* && "$fin" != *marker-B* ]]; } && lost=1
  fires=0; { [ $foreign = 1 ] || [ $lost = 1 ] || [ "$e" != 0 ]; } && fires=1
  ran=0; [ -n "$la" ] && [ -n "$lb" ] && ran=1
  echo "(b) forced shared-writable pair: both ran=$ran; A listed [$la]; B listed [$lb]; final /root [$fin]; foreign-visible=$foreign lost-update=$lost e2fsck rc=$e; FIRES=$fires" >> $R
  cat $R; [ $ran = 1 ] && [ $fires = 1 ]
}
main "$@"; exit
