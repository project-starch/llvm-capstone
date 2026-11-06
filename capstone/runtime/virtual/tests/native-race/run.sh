#!/bin/bash
# usage: run.sh OUTDIR [SCENARIOS]   one VM boot, every scenario of the file in turn.
# Start through the host gate queue. Then: python3 judge.py OUTDIR
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/progRT}
P=$W/platform
O=$1; SC=${2:-$W/src/capstone/runtime/virtual/tests/native-race/scenarios.txt}
MODULE=${RT_MODULE:-$W/adapter-race/module/capstone_vm.ko}
cd "$W/src"
export TMPDIR=$W/tmp CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_TOOLCHAIN_CHECKED=1
source capstone/tests/capstone-test-env.sh
export PYTHONPATH=$W/src/capstone/runtime/host PYTHONDONTWRITEBYTECODE=1
test ! -e "$O"
mkdir -p "$O"
cp "$SC" "$O/scenarios.txt"
sha256sum "$MODULE" "$P/qemu-system-riscv64" "$P/images/Image" "$P/images/fw_jump.elf" "$P/images/rootfs.ext2" \
  "$W/probes/race.dom" "$W/probes/pipe-fill" "$SC" > "$O/inputs.sha256"
vm() { python3 -m capstone_vm --state "$O/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel \
 --cpu-prop x-capstone-test-csrs=true --cpu-prop x-capstone-raw-moves=true --cpu-prop x-capstone-checked-copy=true \
 --cpu-prop x-capstone-context-transport=true --cpu-prop x-capstone-supervisor-slots=32 \
 --qemu "$P/qemu-system-riscv64" --kernel "$P/images/Image" --firmware "$P/images/fw_jump.elf" \
 --rootfs "$P/images/rootfs.ext2" --launcher "$W/adapter-race/capstone-vexec" --job-helper "$W/adapter-race/capstone-job" \
 --module "$MODULE" --ssh-server "$P/ssh-server/dropbearmulti" > "$O/up.log" 2>&1
trap 'vm down >"$O/down.log" 2>&1 || true' EXIT
vm exec -- sh -c 'set -e; rmmod capstone_vm; insmod /mnt/control/module dom_binfmt=1 dom_wide_grants=1 native_slot_transport=1 checked_copy=1 native_dispatch=1 supervisor_slots=32 unpinned=1; echo 0 >/proc/sys/fs/binfmt_misc/status; echo 0 >/proc/sys/kernel/randomize_va_space; cp /mnt/host/probes/race.dom /mnt/host/probes/pipe-fill /tmp/; chmod +x /tmp/race.dom /tmp/pipe-fill' > "$O/setup.log" 2>&1
while IFS='|' read -r id params args; do
  [[ -z $id || $id == \#* ]] && continue
  setp=""
  for n in rt_norecheck rt_noexcl rt_dummy rt_nohold; do
    v=0; [[ " $params " == *" $n "* ]] && v=1
    setp="$setp echo $v >/sys/module/capstone_vm/parameters/$n;"
  done
  rc=0
  vm exec -- sh -c "$setp dmesg -c >/dev/null; cd /tmp; ./pipe-fill ./race.dom $args; r=\$?; echo RC=\$r; dmesg | grep -a capstone; exit 0" \
    > "$O/$id.log" 2>&1 < /dev/null || rc=$?
  echo "$id rc=$rc $(grep -a '^RACE' "$O/$id.log" | head -1)"
done < "$SC"
vm exec -- sh -c 'rmmod capstone_vm; echo unload_rc=$?; dmesg' > "$O/after.log" 2>&1 || true
! grep -aEq 'WARNING:|BUG:|Oops|Unable to handle' "$O/after.log"
grep -aq 'unload_rc=0' "$O/after.log"
