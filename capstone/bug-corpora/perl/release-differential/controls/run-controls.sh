#!/usr/bin/env bash
# The SV-head controls on one arm, one boot: each mode is its own process under capstone-vexec,
# in the order parent, sibling, successor, stale (the stale read last, as its own run).
#
#   CAPSTONE_QEMU_LOCK=<own lock> bash run-controls.sh ARM IMAGE.dom KIT WORK
#
# KIT holds the registered platform: qemu-system-riscv64 (af37cc32), images/{fw_jump.elf,Image,
# rootfs.ext2}, adapter/capstone-vexec (3975314a), adapter/module/capstone_vm.ko (a1c6cb6b).
# runtime/virtual/run-staged.py boots it with one hart and exact bounds on the virtual profile.
# The serial log is WORK/serial.log; collect.py reads it.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
ARM=$1 IMAGE=$2 KIT=$3 WORK=$4
ST=$WORK.stage
rm -rf "$ST"; mkdir -p "$ST"
cp "$KIT/adapter/capstone-vexec" "$KIT/adapter/module/capstone_vm.ko" "$ST/"
cp "$IMAGE" "$ST/ctl.dom"
sha256sum "$ST/ctl.dom" "$ST/capstone-vexec" "$ST/capstone_vm.ko" "$KIT/qemu-system-riscv64" \
  "$KIT"/images/{fw_jump.elf,Image,rootfs.ext2} "$REPO/capstone/runtime/virtual/run-staged.py"
cat > "$ST/gate.sh" <<G
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo \$u; }
echo "CTL_IMAGE:$ARM:\$(sha256sum /mnt/vm/ctl.dom | cut -d ' ' -f 1)"
for mode in parent sibling successor stale; do
  mkdir -p /tmp/ctl-\$mode; cd /tmp/ctl-\$mode
  s=\$(up)
  /mnt/vm/capstone-vexec /mnt/vm/ctl.dom \$mode > /tmp/c.out 2> /tmp/c.err
  rc=\$?; e=\$(up)
  echo "CTL_BEGIN:$ARM:\$mode"; cat /tmp/c.out; echo "CTL_STDERR:$ARM:\$mode"; cat /tmp/c.err
  echo "CTL_END:$ARM:\$mode:\$rc:\$s:\$e"
done
cd /mnt/vm
rmmod capstone_vm
echo CTL_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
chmod +x "$ST/gate.sh"
s=$(date +%s)
rc=0
python3 "$REPO/capstone/runtime/virtual/run-staged.py" --qemu "$KIT/qemu-system-riscv64" \
  --images "$KIT/images" --stage "$ST" --work "$WORK" --timeout 900 || rc=$?
echo "STAGED $ARM rc=$rc wall=$(( $(date +%s) - s ))s work=$WORK"
exit $rc
