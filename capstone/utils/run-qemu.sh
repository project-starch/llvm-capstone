#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUILDROOT_DIR="$REPO_ROOT/capstone/caplifive-buildroot"
QEMU_BINARY="$REPO_ROOT/capstone/capstone-qemu/build/qemu-system-riscv64"

exec "$QEMU_BINARY" \
  -M virt-capstone -m 8G -nographic \
  -bios "$BUILDROOT_DIR/build/images/fw_jump.elf" \
  -kernel "$BUILDROOT_DIR/build/images/Image" \
  -append 'root=/dev/vda ro' \
  -drive "file=$BUILDROOT_DIR/build/images/rootfs.ext2,format=raw,id=hd0" \
  -device virtio-blk-device,drive=hd0 \
  -chardev stdio,mux=on,id=ch0,signal=on \
  -mon chardev=ch0,mode=readline \
  -serial chardev:ch0 \
  -cpu rv64,sstc=false,h=false
# -netdev user,id=net0,hostfwd=tcp::60022-:22 \
# -device e1000,netdev=net0

