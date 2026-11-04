#!/bin/bash
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/progI}
P=$W/platform
O=${NATIVE_OUT:-$W/guest-safe}
MODULE=${NATIVE_MODULE:-$P/adapter/module/capstone_vm.ko}
cd "$W/src"
export TMPDIR=$W/tmp CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_TOOLCHAIN_CHECKED=1
source capstone/tests/capstone-test-env.sh
export PYTHONPATH=$W/src/capstone/runtime/host PYTHONDONTWRITEBYTECODE=1
test ! -e "$O"
mkdir -p "$O"
sha256sum "$MODULE" "$P/qemu-system-riscv64" "$P/images/Image" \
 "$P/images/fw_jump.elf" "$P/images/rootfs.ext2" "$W/probes/probe.dom" > "$O/inputs.sha256"
vm() { python3 -m capstone_vm --state "$O/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel \
 --cpu-prop x-capstone-test-csrs=true --cpu-prop x-capstone-raw-moves=true --cpu-prop x-capstone-checked-copy=true \
 --cpu-prop x-capstone-context-transport=true --cpu-prop x-capstone-supervisor-slots=32 \
 --qemu "$P/qemu-system-riscv64" --kernel "$P/images/Image" --firmware "$P/images/fw_jump.elf" \
 --rootfs "$P/images/rootfs.ext2" --launcher "$P/adapter/capstone-vexec" --job-helper "$P/adapter/capstone-job" \
 --module "$MODULE" --ssh-server "$P/ssh-server/dropbearmulti" > "$O/up.log" 2>&1
trap 'vm down >"$O/down.log" 2>&1 || true' EXIT
vm exec -- sh -c 'set -e; rmmod capstone_vm; insmod /mnt/control/module dom_binfmt=1 dom_wide_grants=1 native_slot_transport=1 checked_copy=1 native_dispatch=1 supervisor_slots=32 unpinned=1; echo 0 >/proc/sys/fs/binfmt_misc/status; cp /mnt/host/probes/probe.dom /tmp/; chmod +x /tmp/probe.dom; yes Q | head -c 131072 >/tmp/input' > "$O/setup.log" 2>&1
rc=0
vm exec -- sh -c 'cd /tmp; ./probe.dom <input >output; r=$?; cat output; exit $r' > "$O/result.log" 2>&1 || rc=$?
vm exec -- sh -c 'rmmod capstone_vm; echo unload_rc=$?; dmesg' > "$O/after.log" 2>&1
cat "$O/result.log"
if grep -aEq 'WARNING:|BUG:|thread create refused|Exec format error' "$O/after.log"; then rc=1; fi
grep -aq 'unload_rc=0' "$O/after.log" || rc=1
if [[ ${EXPECT_BOUNDS_FAILURE:-0} == 1 ]]; then
    test "$rc" -eq 1
    grep -aq '^FAIL tight bound refused' "$O/result.log"
    grep -aq '^FAIL bound sentinel intact' "$O/result.log"
    grep -aq '^PASS forged scalar buffer' "$O/result.log"
    grep -aq '^PASS revoked input refused' "$O/result.log"
    echo 'unsafe bounds control detected sentinel corruption'
else
    grep -aq 'native integration checks=24 failures=0' "$O/result.log" || rc=1
    exit "$rc"
fi
