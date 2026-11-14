#!/usr/bin/env bash
# boot.sh BINARIES BASE_INITRAMFS QEMU FIRMWARE KERNEL OUTPUT
set -euo pipefail
BIN=$(realpath "${1:?test binaries}")
BASE=$(realpath "${2:?busybox initramfs}")
QEMU=$(realpath "${3:?QEMU}")
FW=$(realpath "${4:?firmware}")
KERNEL=$(realpath "${5:?kernel}")
OUT=$(realpath -m "${6:?output directory}")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"
mkdir -p "$OUT/root"
(cd "$OUT/root" && gzip -dc "$BASE" | cpio -idu --quiet)
cp "$BIN/"*.dom "$OUT/root/"
install -m 0755 "$HERE/init" "$OUT/root/init"
(cd "$OUT/root" && find . | cpio -o -H newc --quiet | gzip -1) > "$OUT/initramfs.cpio.gz"
flock "$CAPSTONE_QEMU_LOCK" flock "$OUT/domain.lock" \
    python3 "$HERE/boot.py" "$QEMU" "$FW" "$KERNEL" "$OUT/initramfs.cpio.gz" "$OUT/boot.log"
