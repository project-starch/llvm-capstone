#!/usr/bin/env bash
# A PRIVATE guest rootfs for this port's QEMU runs, built fresh from buildroot's pristine
# rootfs.tar, with a chosen kernel module as /capstone.ko.
#
#   make-rootfs.sh <out-dir> <capstone.ko>
#
# <out-dir>/build/images/ then holds rootfs.ext2 (fresh), plus fw_jump.elf and Image as symlinks
# to the shared build: what the runner's --buildroot-dir expects.
#
# Why not copy the shared build/images/rootfs.ext2:
# - Every lane boots that image, and it has been found ext4-corrupt twice: 2026-09-23 (see the
#   FFmpeg port's run-qemu.sh) and 2026-09-24, when a suite booted it read-write without
#   -snapshot and `e2fsck -n` exited 4.
# - A copy inherits the corruption. rootfs.tar is buildroot's own output from the same build, and
#   this script only reads it.
# Same geometry as the shared image: ext2 rev 1, 4 KiB blocks, 131072 inodes, 2 GiB, label rootfs.
# Needs fakeroot (the host's, or buildroot's build/host/bin/fakeroot).
# Checks: e2fsck -n is clean, and /capstone.ko reads back byte-identical.
set -euo pipefail
OUT=${1:?out dir}; KO=${2:?capstone.ko}
IMG=${CAPSTONE_BUILDROOT_DIR:?set CAPSTONE_BUILDROOT_DIR}/build/images
TAR=$IMG/rootfs.tar
[ -f "$TAR" ] && [ -f "$KO" ] || { echo "make-rootfs: need $TAR and $KO" >&2; exit 2; }
D=$OUT/build/images; mkdir -p "$D"
ln -sf "$IMG/fw_jump.elf" "$D/fw_jump.elf"; ln -sf "$IMG/Image" "$D/Image"
rm -f "$D/rootfs.ext2"
# Extracted under fakeroot, so the image keeps the tarball's owners (0/0, and one 33/33); built by
# mke2fs -d from that directory under the same fakeroot state. (e2fsprogs 1.47.0's mke2fs cannot
# read a tarball directly.)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir "$W/root"
fakeroot -s "$W/state" -- tar -xpf "$TAR" -C "$W/root"
fakeroot -i "$W/state" -- mke2fs -q -t ext2 -r 1 -L rootfs -b 4096 -N 131072 -d "$W/root" "$D/rootfs.ext2" 2G
debugfs -w -R "rm /capstone.ko" "$D/rootfs.ext2" > /dev/null 2>&1 || true
debugfs -w -R "write $KO /capstone.ko" "$D/rootfs.ext2" > /dev/null 2>&1
debugfs -w -R "sif /capstone.ko mode 0100755" "$D/rootfs.ext2" > /dev/null 2>&1
chk=$(mktemp); debugfs -R "dump /capstone.ko $chk" "$D/rootfs.ext2" > /dev/null 2>&1
cmp -s "$KO" "$chk" || { rm -f "$chk"; echo "make-rootfs: /capstone.ko did not read back" >&2; exit 1; }
rm -f "$chk"
e2fsck -n -f "$D/rootfs.ext2" > "$OUT/e2fsck.log" 2>&1 || { echo "make-rootfs: e2fsck -n not clean ($OUT/e2fsck.log)" >&2; exit 1; }
echo "make-rootfs: $D/rootfs.ext2 fresh from $(basename "$TAR"), /capstone.ko = $(sha256sum < "$KO" | cut -c1-16), e2fsck clean"
