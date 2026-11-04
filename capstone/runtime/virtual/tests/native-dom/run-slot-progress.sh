#!/bin/bash
# Queue this progress probe; it deliberately uses ONE slot, unlike landing gates.
set -euo pipefail
W=${LANE_ROOT:?private lane directory}
P=${NATIVE_PLATFORM:?platform directory}
O=${CONTRACT_OUT:?fresh output directory}
cd "$W/src"
export CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT=$W/tmp
source capstone/tests/capstone-test-env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src/capstone/runtime/host
[[ ! -e $O ]] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$O"
sha256sum "$P/adapter/module/capstone_vm.ko" "$P/qemu-system-riscv64" \
 "$P/images/Image" "$W/contracts/slot-progress" "$W/contracts/contract.dom" > "$O/inputs.sha256"
vm() { python3 -m capstone_vm --state "$O/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel \
 --cpu-prop x-capstone-context-transport=true --cpu-prop x-capstone-supervisor-slots=1 --cpu-prop x-capstone-test-csrs=true \
 --qemu "$P/qemu-system-riscv64" --kernel "$P/images/Image" --firmware "$P/images/fw_jump.elf" \
 --rootfs "$P/images/rootfs.ext2" --launcher "$P/adapter/capstone-vexec" --job-helper "$P/adapter/capstone-job" \
 --module "$P/adapter/module/capstone_vm.ko" --ssh-server "$P/ssh-server/dropbearmulti" > "$O/up.log" 2>&1
trap 'vm down >"$O/down.log" 2>&1 || true' EXIT
vm exec -- sh -c "set -e; rmmod capstone_vm; insmod /mnt/control/module dom_binfmt=1 supervisor_slots=1 native_slot_transport=${SLOT_TRANSPORT:-1}; echo 0 >/proc/sys/fs/binfmt_misc/status; cp /mnt/host/contracts/slot-progress /mnt/host/contracts/contract.dom /tmp/; chmod +x /tmp/slot-progress /tmp/contract.dom" > "$O/setup.log" 2>&1
rc=0
vm exec -- /tmp/slot-progress /tmp/contract.dom > "$O/result.log" 2>&1 || rc=$?
vm exec -- sh -c 'rmmod capstone_vm; echo unload_rc=$?; dmesg' > "$O/after.log" 2>&1
if grep -aEq 'do not call blocking ops|WARNING:|BUG:|thread create refused|Exec format error' "$O/after.log"; then
 echo 'FAIL kernel diagnostics' >> "$O/result.log"
 rc=1
fi
if ! grep -aq 'unload_rc=0' "$O/after.log" ||
   [ "$(grep -ac '^PASS slot-progress' "$O/result.log")" != 8 ]; then
 echo 'FAIL incomplete progress probe or unload' >> "$O/result.log"
 rc=1
fi
echo "probe_rc=$rc" >> "$O/result.log"
cat "$O/result.log"
exit "$rc"
