#!/bin/bash
# run-arm.sh <arm>: boot once, run all 165 cases with that arm's domain mruby
set -u
# REPO is the llvm-capstone tree this runs from, KIT the scratch directory
# holding the arm images and the staged share. Both default off this script's
# own location and $CAPSTONE_TMP_ROOT, so nothing here names a home directory.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
KIT=${KIT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/mruby-arms}
W=$REPO; K=$KIT; RT=$W/capstone/runtime
arm=$1; OUT=$K/results; mkdir -p $OUT
export PYTHONPATH=$RT/host CAPSTONE_QEMU_LOCK=$K/qemu.lock
case $arm in level0) ;; *) echo "run-arm.sh: arm $arm? (level0; the Sublet-heap arms were removed 2026-10-11)" >&2; exit 2 ;; esac
ROOT=/tmp/capstone/step-one-ecall; S=$K/vm-$arm
vm() { python3 -m capstone_vm --state $S "$@"; }
vm down > /dev/null 2>&1; rm -rf $S
vm up --qemu $ROOT/qemu-build/qemu-system-riscv64 --kernel /tmp/capstone/delegation-binfmt-linux/arch/riscv/boot/Image \
  --firmware $ROOT/new2/opensbi/platform/generic/firmware/fw_jump.elf --rootfs /tmp/capstone/cheap/rootfs.ext2 --share $K/share \
  --module /tmp/capstone/launch-cost/module/module/capstone.ko --launcher /tmp/capstone/v0-removal-stack/guest/capstone-exec \
  --cma-mib 1024 --process-cache-mib 768 > $OUT/up-$arm.log 2>&1 || { echo "up FAILED"; tail -5 $OUT/up-$arm.log; exit 1; }
# the arm's own control first: an arm whose control fails cannot report a catch
timeout 600 python3 -m capstone_vm --state $S exec /mnt/host/mruby-$arm.dom /mnt/host/files/smoke.rb > $OUT/$arm-smoke.txt 2>&1
echo "smoke rc=$? $(grep -c SMOKE_DONE $OUT/$arm-smoke.txt 2>/dev/null)"
timeout 7200 python3 -m capstone_vm --state $S exec sh /mnt/host/run-all.sh /mnt/host/mruby-$arm.dom > $OUT/$arm.txt 2>&1
echo "run rc=$?"
vm down > /dev/null 2>&1
echo "ARM-DONE $arm"
