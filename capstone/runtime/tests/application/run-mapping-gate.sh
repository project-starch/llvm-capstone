#!/usr/bin/env bash
# The translated-mapping contract in the QEMU gate (docs/plans/mapping-transport-m2.md, section 4).
#
# Boots the images under $CAPSTONE_BUILDROOT_DIR/build/images with run-domain-smoke.py, loads the
# driver and the launcher from the share, and runs mapping-contract.dom in every mode. The share
# directory must hold capstone.ko, capstone-exec and mapping-contract.dom. Every mode's marker is
# required, so a mode that fails or halts fails the gate; alias-fault passes by ending in the
# reported domain fault (the launcher exits with SIGSEGV, 139).
#
#   CAPSTONE_BUILDROOT_DIR=<dir with build/images> CAPSTONE_QEMU_BINARY=<qemu-system-riscv64> \
#     run-mapping-gate.sh <share-dir> <log-file>
#
# Take the QEMU lock first (CAPSTONE_QEMU_LOCK_HELD=1 flock -x ~/.capstone-locks/qemu.lock ...).
set -u
share=${1:?share directory}
log=${2:?log file}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for f in capstone.ko capstone-exec mapping-contract.dom; do
  [ -f "$share/$f" ] || { echo "run-mapping-gate: $share/$f missing" >&2; exit 2; }
done
cmd='rmmod capstone; insmod /mnt/host/capstone.ko && cp /mnt/host/capstone-exec /tmp/capstone-exec && chmod 0755 /tmp/capstone-exec && for m in basic refresh two errors; do /tmp/capstone-exec /mnt/host/mapping-contract.dom $m || break; done; CAPSTONE_EXEC_DIAGNOSTICS=1 /tmp/capstone-exec /mnt/host/mapping-contract.dom alias-fault; echo "mapping-contract alias-fault: exit=$?"'
exec python3 "$here/../../../tests/runtime-qemu/run-domain-smoke.py" --share-dir "$share" --log-file "$log" \
  --timeout-multiplier "${CAPSTONE_TIMEOUT_MULTIPLIER:-6}" --guest-command "$cmd" \
  --success-marker 'mapping-contract basic: PASS' --success-marker 'mapping-contract refresh: PASS' \
  --success-marker 'mapping-contract two: PASS' --success-marker 'mapping-contract errors: PASS' \
  --success-marker 'mapping-contract alias-fault: exit=139'
