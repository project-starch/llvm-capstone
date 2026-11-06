#!/bin/bash
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/progI}
P=$W/platform
O=${BUDGET_OUT:-$W/budget}
MODULE=${NATIVE_MODULE:-$P/adapter/module/capstone_vm.ko}
cd "$W/src"
export TMPDIR=$W/tmp CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_TOOLCHAIN_CHECKED=1
source capstone/tests/capstone-test-env.sh
export PYTHONPATH=$W/src/capstone/runtime/host PYTHONDONTWRITEBYTECODE=1
test ! -e "$O"
mkdir -p "$O/capprof"
sha256sum "$MODULE" "$P/qemu-system-riscv64" "$P/images/Image" \
 "$P/images/fw_jump.elf" "$P/images/rootfs.ext2" "$P/plugins/libcapprof.so" \
 "$P/adapter/capstone-vexec" "$W/probes/pipe-exec" "$W"/probes/budget-* \
 > "$O/inputs.sha256"
vm() { python3 -m capstone_vm --state "$O/vm" "$@"; }
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$W" --no-pcrel --icount 0 --plugin "$P/plugins/libcapprof.so" --plugin-arg out="$O/capprof" \
 --cpu-prop x-capstone-test-csrs=true --cpu-prop x-capstone-raw-moves=true --cpu-prop x-capstone-checked-copy=true \
 --cpu-prop x-capstone-context-transport=true --cpu-prop x-capstone-supervisor-slots=32 \
 --qemu "$P/qemu-system-riscv64" --kernel "$P/images/Image" --firmware "$P/images/fw_jump.elf" \
 --rootfs "$P/images/rootfs.ext2" --launcher "$P/adapter/capstone-vexec" --job-helper "$P/adapter/capstone-job" \
 --module "$MODULE" --ssh-server "$P/ssh-server/dropbearmulti" > "$O/up.log" 2>&1
trap 'vm down >"$O/down.log" 2>&1 || true' EXIT
vm exec -- sh -c 'set -e; rmmod capstone_vm; insmod /mnt/control/module dom_binfmt=1 dom_wide_grants=1 native_slot_transport=1 checked_copy=1 native_dispatch=1 supervisor_slots=32 unpinned=1 '"${MODULE_ARGS:-}"'; echo 0 >/proc/sys/fs/binfmt_misc/status; echo 0 >/proc/sys/kernel/randomize_va_space; killall syslogd klogd 2>/dev/null || true; cp /mnt/host/probes/* /tmp/; cp /mnt/control/launcher /tmp/launcher; chmod +x /tmp/launcher; chmod +x /tmp/budget-* /tmp/pipe-exec' > "$O/setup.log" 2>&1
# Load addresses of the module's sections, for per-function attribution of the windows.
vm exec -- sh -c 'cd /sys/module/capstone_vm/sections && for f in .[a-z]* ; do echo "$f $(cat $f)"; done' > "$O/sections.txt" 2>&1
index=0
for arm in ${BUDGET_ARMS:-plain launcher native}; do
 for bytes in ${BUDGET_BYTES:-1 65536 0}; do
  for rep in ${BUDGET_REPS:-1 2 3}; do
   cmd=/tmp/budget-plain
   [[ $arm != launcher ]] || cmd='/tmp/launcher /tmp/budget-legacy.dom'
   [[ $arm != native ]] || cmd=/tmp/budget-native.dom
   printf '%s\t%s\t%s\t%s\n' "$index" "$arm" "$bytes" "$rep" >>"$O/arms.tsv"
   vm exec -- sh -c "cd /tmp; ./pipe-exec $cmd $bytes >budget.out 2>budget.err; r=\$?; cat budget.out budget.err; exit \$r" >"$O/$index.log" 2>&1
   index=$((index+1))
  done
 done
done
vm exec -- sh -c 'rmmod capstone_vm; echo unload_rc=$?; dmesg' >"$O/after.log" 2>&1
! grep -aEq 'WARNING:|BUG:|thread create refused|Exec format error' "$O/after.log"
grep -aq 'unload_rc=0' "$O/after.log"
