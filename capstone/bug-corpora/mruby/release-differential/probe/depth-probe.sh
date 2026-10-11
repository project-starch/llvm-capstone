#!/bin/bash
# Localise the revoking arms' control failure: one recursion depth per run, all three arms.
set -u
# REPO is the llvm-capstone tree this runs from, KIT the scratch directory
# holding the arm images and the staged share. Both default off this script's
# own location and $CAPSTONE_TMP_ROOT, so nothing here names a home directory.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
KIT=${KIT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/mruby-arms}
W=$REPO; K=$KIT; RT=$W/capstone/runtime
OUT=$K/results; mkdir -p $OUT
export PYTHONPATH=$RT/host CAPSTONE_QEMU_LOCK=$K/qemu.lock CAPSTONE_REV_NODES=16777216
ROOT=/tmp/capstone/step-one-ecall; S=$K/vm-depth
python3 -m capstone_vm --state $S down >/dev/null 2>&1; rm -rf $S
python3 -m capstone_vm --state $S up --qemu $ROOT/qemu-build/qemu-system-riscv64 \
  --kernel /tmp/capstone/delegation-binfmt-linux/arch/riscv/boot/Image \
  --firmware $ROOT/new2/opensbi/platform/generic/firmware/fw_jump.elf --rootfs /tmp/capstone/cheap/rootfs.ext2 \
  --share $K/share --module /tmp/capstone/launch-cost/module/module/capstone.ko \
  --launcher /tmp/capstone/v0-removal-stack/guest/capstone-exec --cma-mib 1024 --process-cache-mib 768 \
  > $OUT/up-depth.log 2>&1 || { echo "up FAILED"; exit 1; }
for arm in level0 sublet sublet-gc; do
  for d in 20 40 60 100 200 500; do
    o=$(timeout 180 python3 -m capstone_vm --state $S exec /mnt/host/mruby-$arm.dom /mnt/host/depth/d$d.rb 2>&1)
    echo "$arm d=$d rc=$? :: $(echo "$o" | grep -E 'DEEP|cause=|Segmentation' | head -2 | tr '\n' ' ')"
  done
done
python3 -m capstone_vm --state $S down >/dev/null 2>&1
echo DEPTH-DONE
