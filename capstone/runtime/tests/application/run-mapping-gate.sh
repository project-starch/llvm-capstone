#!/usr/bin/env bash
# The translated-mapping contract in the QEMU gate (docs/plans/mapping-transport-m2.md, section 4).
#
# Two boots of the images under $CAPSTONE_BUILDROOT_DIR/build/images with run-domain-smoke.py. The
# share directory must hold capstone.ko, capstone-exec, capstone-exec-adversary and
# mapping-contract.dom; this script writes the two guest scripts next to them.
#
#   boot 1: every mode under the honest launcher, then the lying launcher's cases (release claimed
#           but not done, read-only widened to read-write, a signal between grant and resume, a
#           second grant before the domain was entered, a grant to a preempted domain);
#   boot 2: CAPSTONE_REV_NODES=512, a 2 MiB grant must be refused with an error and small grants
#           must keep working (the monitor must not fault).
#
# Each case prints "gate <case>: OK" when its exit status is the expected one; every OK line and the
# adversary's own report lines are required markers, so any other outcome fails the gate.
#
#   CAPSTONE_BUILDROOT_DIR=<dir with build/images> CAPSTONE_QEMU_BINARY=<qemu-system-riscv64> \
#     run-mapping-gate.sh <share-dir> <log-prefix>
#
# Take the QEMU lock first (CAPSTONE_QEMU_LOCK_HELD=1 flock -x ~/.capstone-locks/qemu.lock ...).
set -u
share=${1:?share directory}
log=${2:?log prefix}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$here/../../../tests/runtime-qemu/run-domain-smoke.py"
for f in capstone.ko capstone-exec capstone-exec-adversary mapping-contract.dom; do
  [ -f "$share/$f" ] || { echo "run-mapping-gate: $share/$f missing" >&2; exit 2; }
done

prologue='rmmod capstone 2>/dev/null
insmod /mnt/host/capstone.ko || exit 1
cp /mnt/host/capstone-exec /mnt/host/capstone-exec-adversary /tmp/ || exit 1
chmod 0755 /tmp/capstone-exec /tmp/capstone-exec-adversary
D=/mnt/host/mapping-contract.dom
run() {
  expect=$1; label=$2; shift 2
  "$@"; rc=$?
  if [ "$rc" -eq "$expect" ]; then echo "gate $label: OK"; else echo "gate $label: BAD exit=$rc expected=$expect"; fi
}'
cat > "$share/mapping-gate-1.sh" <<GUEST
$prologue
for m in basic refresh two errors cycle large; do run 0 \$m /tmp/capstone-exec \$D \$m; done
run 0 signal-none /tmp/capstone-exec \$D signal 0
run 139 ro-store /tmp/capstone-exec \$D ro-store
run 139 alias-fault /tmp/capstone-exec \$D alias-fault
run 139 alias-fault-release-noop env CAPSTONE_ADVERSARY=release-noop /tmp/capstone-exec-adversary \$D alias-fault
run 0 ro-widened env CAPSTONE_ADVERSARY=widen /tmp/capstone-exec-adversary \$D ro-refused
run 0 signal-injected env CAPSTONE_ADVERSARY=signal /tmp/capstone-exec-adversary \$D signal 1
run 0 double-grant env CAPSTONE_ADVERSARY=double /tmp/capstone-exec-adversary \$D basic
run 0 preempted-grant env CAPSTONE_ADVERSARY=preempt /tmp/capstone-exec-adversary \$D spin
echo "gate boot 1 done"
GUEST
cat > "$share/mapping-gate-2.sh" <<GUEST
$prologue
run 0 budget /tmp/capstone-exec \$D budget
echo "gate boot 2 done"
GUEST

markers1=()
for c in basic refresh two errors cycle large signal-none ro-store alias-fault alias-fault-release-noop \
         ro-widened signal-injected double-grant preempted-grant; do
  markers1+=(--success-marker "gate $c: OK")
done
markers1+=(--success-marker 'second grant before entry refused'
           --success-marker 'grant to a preempted domain refused'
           --success-marker 'gate boot 1 done')

python3 "$runner" --share-dir "$share" --log-file "$log-1.log" \
  --timeout-multiplier "${CAPSTONE_TIMEOUT_MULTIPLIER:-6}" \
  --guest-command 'sh /mnt/host/mapping-gate-1.sh' "${markers1[@]}"
rc1=$?
CAPSTONE_REV_NODES=512 python3 "$runner" --share-dir "$share" --log-file "$log-2.log" \
  --timeout-multiplier "${CAPSTONE_TIMEOUT_MULTIPLIER:-6}" \
  --guest-command 'sh /mnt/host/mapping-gate-2.sh' \
  --success-marker 'mapping-contract budget: large=refused' --success-marker 'gate budget: OK' \
  --success-marker 'gate boot 2 done'
rc2=$?
echo "run-mapping-gate: boot 1 rc=$rc1, boot 2 rc=$rc2"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
