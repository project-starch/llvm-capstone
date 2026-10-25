#!/usr/bin/env bash
# One persistent virtual VM for the corpus, as pg-virtual-kit/vm-up-sublet.sh (QEMU af37cc32,
# launcher 3975314a, module a1c6cb6b), state and share under ~/sa-glob-pf, own QEMU lock.
set -eu
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf name=$1
mkdir -p $M/share-$name
cd $M/repo/capstone/runtime/host
PYTHONDONTWRITEBYTECODE=1 CAPSTONE_QEMU_LOCK=$M/qemu-$name.lock CAPSTONE_REV_NODES=262144 python3 -m capstone_vm --state $M/$name up \
  --profile virtual --exact-bounds --qemu $K/qemu-src-sublet/build/qemu-system-riscv64 \
  --kernel $K/platform/images/Image --firmware $K/platform/images/fw_jump.elf --rootfs $K/platform/images/rootfs.ext2 \
  --share $M/share-$name --launcher $K/adapter-sublet/capstone-vexec --module $K/adapter-sublet/module/capstone_vm.ko \
  --ssh-server $K/platform/adapter/dropbearmulti --cma-mib 1024 --process-cache-mib 768
