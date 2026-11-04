#!/bin/bash
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/laneAG2}
cd "$W/src"
export CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_QEMU_LOCK=$W/contracts.lock CAPSTONE_TOOLCHAIN_CHECKED=1
source capstone/tests/capstone-test-env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src/capstone/runtime/host
PL=${NATIVE_PLATFORM:-$W/platform-native}
OUT=${CONTRACT_OUT:-$W/contracts-review}
slots=${NATIVE_SLOTS:-32}
wide=${NATIVE_WIDE_GRANTS:-0}
[[ $slots =~ ^[0-9]+$ && $wide =~ ^[01]$ ]] || exit 2
mkdir -p "$OUT"
vm() { python3 -m capstone_vm --state "$OUT/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel \
 --cpu-prop x-capstone-supervisor-slots="$slots" --cpu-prop x-capstone-test-csrs=true \
 --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
 --rootfs "$PL/images/rootfs.ext2" --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
 --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/up.log" 2>&1
trap 'vm down >/dev/null 2>&1 || true' EXIT
vm exec -- sh -c 'rmmod capstone_vm && insmod /mnt/control/module dom_binfmt=1 supervisor_slots="$1" dom_wide_grants="$2" && echo 0 >/proc/sys/fs/binfmt_misc/status && cp /mnt/host/contracts/contract.dom* /mnt/host/contracts/driver /tmp/ && chmod +x /tmp/contract.dom* /tmp/driver && chmod 4755 /tmp/contract.dom.setuid' sh "$slots" "$wide" > "$OUT/setup.log" 2>&1
set +e
vm exec -- /tmp/driver /tmp/contract.dom > "$OUT/contract.log" 2>&1
rc=$?
vm exec -- sh -c 'dmesg | tail -40; rmmod capstone_vm; echo unload_rc=$?' > "$OUT/after.log" 2>&1
set -e
cat "$OUT/contract.log"
cat "$OUT/after.log"
exit "$rc"
