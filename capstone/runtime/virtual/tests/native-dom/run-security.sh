#!/bin/bash
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/laneAG2}
cd "$W/src"
export CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_QEMU_LOCK=$W/probe.lock
source capstone/tests/capstone-test-env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src/capstone/runtime/host
PL=${NATIVE_PLATFORM:-$W/platform-native}
MODULE=${NATIVE_MODULE:-$PL/adapter/module/capstone_vm.ko}
OUT=${SECURITY_OUT:-$W/security}; mkdir -p "$OUT"
slots=${NATIVE_SLOTS:-32}
wide=${NATIVE_WIDE_GRANTS:-0}
[[ $slots =~ ^[0-9]+$ && $wide =~ ^[01]$ ]] || exit 2
sha256sum "$MODULE" "$PL/qemu-system-riscv64" \
 "$PL/images/Image" "$W/contracts/scalar-memory.dom" "$W/contracts/resource-probe.dom" \
 "$W/contracts/dac-driver" > "$OUT/inputs.sha256"
vm() { python3 -m capstone_vm --state "$OUT/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel \
 --cpu-prop x-capstone-supervisor-slots="$slots" --cpu-prop x-capstone-test-csrs=true \
 --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
 --rootfs "$PL/images/rootfs.ext2" --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
 --module "$MODULE" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/up.log" 2>&1
trap 'vm down >/dev/null 2>&1 || true' EXIT
vm exec -- sh -c 'set -e
rmmod capstone_vm
insmod /mnt/control/module dom_binfmt=1 supervisor_slots="$1" dom_wide_grants="$2"
echo 0 >/proc/sys/fs/binfmt_misc/status
echo 0 >/proc/sys/kernel/randomize_va_space
cp /mnt/host/contracts/scalar-memory.dom /mnt/host/contracts/resource-probe.dom /mnt/host/contracts/dac-driver /tmp/
chmod +x /tmp/scalar-memory.dom /tmp/resource-probe.dom /tmp/dac-driver
mkdir -p /tmp/resource-root/nested /tmp/resource-root/proc-mount
mount -t proc proc /tmp/resource-root/proc-mount
ln -s /proc/self/mem /tmp/resource-root/mem-link
ln -s /proc /tmp/resource-root/proc-link
ln -s /proc/self/fd/9 /tmp/resource-root/fd-link
ln -s / /tmp/resource-root/root-link
mknod /tmp/resource-root/memory-device c 1 1
mknod /tmp/resource-root/null-device c 1 3
: >/tmp/resource-outside
echo private >/tmp/resource-root/no-permission
echo file >/tmp/resource-root/fd-input
chmod 000 /tmp/resource-root/no-permission' sh "$slots" "$wide" > "$OUT/setup.log" 2>&1
if [[ ${TEST_WIDE_GRANTS:-0} == 1 ]]; then
 vm exec -- sh -c 'set -e
cp /mnt/host/contracts/wide-grants.dom /tmp/
chmod +x /tmp/wide-grants.dom
printf granted >/usr/n2-wide-input
mkdir /tmp/n2-proc
mount -t proc proc /tmp/n2-proc
mknod /tmp/n2-null-alias c 1 3
mknod /tmp/n2-memory c 1 1
ln -s /dev/null /tmp/n2-null-link
cd /tmp/resource-root
/tmp/wide-grants.dom; echo wide_rc=$?' > "$OUT/wide.log" 2>&1 || true
fi
vm exec -- sh -c 'cd /tmp; ./scalar-memory.dom; echo scalar_rc=$?' > "$OUT/scalar.log" 2>&1
# Two identical resource-probe execs with ASLR disabled: the first records the
# image placement; the second probes the scalar trampoline's actual address.
vm exec -- sh -c 'cd /tmp/resource-root; exec 8</proc; exec 9<>/proc/self/mem; /tmp/resource-probe.dom 1' > "$OUT/seed.log" 2>&1
vm exec -- sh -c 'cd /tmp/resource-root; exec 8</proc; exec 9<>/proc/self/mem
base=$(cat /sys/module/capstone_vm/parameters/dom_last_trampoline)
echo trampoline=$base
/tmp/resource-probe.dom "$base"
echo resource_rc=$?
echo trampoline_after=$(cat /sys/module/capstone_vm/parameters/dom_last_trampoline)' > "$OUT/resource.log" 2>&1
vm exec -- sh -c 'cd /tmp/resource-root; exec 0</proc/self/mem; /tmp/resource-probe.dom stdio; echo stdio_rc=$?' > "$OUT/stdio.log" 2>&1
vm exec -- sh -c 'cd /tmp/resource-root; /tmp/dac-driver /tmp/resource-probe.dom; echo dac_rc=$?' > "$OUT/dac.log" 2>&1
vm exec -- sh -c 'cd /tmp/resource-root; /tmp/dac-driver /tmp/resource-probe.dom many-fds; echo fds_rc=$?' > "$OUT/fds.log" 2>&1
vm exec -- sh -c 'dmesg | tail -20; rmmod capstone_vm; echo unload_rc=$?' > "$OUT/after.log" 2>&1
cat "$OUT/scalar.log" "$OUT/resource.log" "$OUT/stdio.log" "$OUT/dac.log" "$OUT/fds.log"
python3 - "$OUT" <<'CHECK'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
if (p / 'wide.log').exists():
    text = (p / 'wide.log').read_text()
    assert 'FAIL' not in text and 'wide_rc=0' in text, text
for name, key in [('scalar','scalar'), ('resource','resource'), ('stdio','stdio'), ('dac','dac'), ('fds','fds')]:
    text = (p / (name + '.log')).read_text()
    assert 'FAIL' not in text and key + '_rc=0' in text, text
text = (p / 'resource.log').read_text()
assert re.search(r'^trampoline=(\d+)$', text, re.M)[1] == re.search(r'^trampoline_after=(\d+)$', text, re.M)[1]
assert 'unload_rc=0' in (p / 'after.log').read_text()
print('PASS security logs and stable trampoline address')
CHECK
