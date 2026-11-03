#!/bin/bash
# vt-run2.sh TAG: capprof validation on the profiling platform (~/prof-dev/platform): one VM with
# the plugin, -icount, no-pcrel and the test CSRs, the validation programs in the fixed order of
# vtcheck.py (one window each), results in ~/prof-dev/vt/TAG/. Then vtcheck.py.
set -euo pipefail
TAG=$1
D=~/prof-dev; PL=$D/platform; V=$D/vt; OUT=$V/$TAG
[ -e "$OUT" ] && { echo "exists $OUT"; exit 2; }
mkdir -p "$OUT/capprof"
vm() { env CAPSTONE_QEMU_LOCK=$OUT/qemu.lock PYTHONPATH=$D/host python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mkdir -m 700 -p ~/.capstone-spec-vm/vt-$TAG; ln -sfn ~/.capstone-spec-vm/vt-$TAG "$OUT/vm"
vm up --profile virtual --exact-bounds --memory 8G --share-model mapped-xattr --no-pcrel --icount 0 \
  --cpu-prop x-capstone-test-csrs=true \
  --plugin $PL/plugins/libcapprof.so --plugin-arg out=$OUT/capprof \
  --qemu $PL/qemu-system-riscv64 --kernel $PL/images/Image --firmware $PL/images/fw_jump.elf \
  --rootfs $PL/images/rootfs.ext2 --share $V \
  --launcher $PL/adapter/capstone-vexec --job-helper $PL/adapter/capstone-job \
  --module $PL/adapter/module/capstone_vm.ko --ssh-server $PL/ssh-server/dropbearmulti > "$OUT/vm-up.log" 2>&1
trap 'vm down > /dev/null 2>&1 || true' EXIT
vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space; killall syslogd klogd 2>/dev/null; cat /proc/sys/kernel/randomize_va_space" > "$OUT/guest-setup.txt"
n=0
cap() { echo "window $n: vt-cap $1" >> "$OUT/order.txt"; n=$((n+1))
  vm run --cwd /mnt/host -e CAPSTONE_VM_STATS=1 -e CAPSTONE_PROFILE_RECORD=/mnt/host/$TAG/image-records.txt \
    --result "$OUT/run-$n.json" -- ./vt-cap "$1" > "$OUT/run-$n.out" 2> "$OUT/run-$n.err" || echo "rc=$? window $((n-1))" >> "$OUT/order.txt"; }
plain() { echo "window $n: $*" >> "$OUT/order.txt"; n=$((n+1))
  vm exec -- sh -c "cp /mnt/host/$1 /tmp/$1 && chmod 0755 /tmp/$1 && cd /mnt/host && /tmp/$*" > "$OUT/run-$n.out" 2> "$OUT/run-$n.err" || echo "rc=$? window $((n-1))" >> "$OUT/order.txt"; }
cap window; cap window; cap long; cap prefault; cap fault
plain vt-plain window; plain alias
vm down; trap - EXIT
ls "$OUT/capprof"
python3 $V/vtcheck.py "$OUT" | tee "$OUT/vtcheck.txt" | tail -15
