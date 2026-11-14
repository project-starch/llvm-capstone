#!/usr/bin/env bash
# Run the six unit images in one short trap-route boot.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"
OUT=${1:?output directory containing fixed/ and unfixed/ images}
QEMU=${SUBLET_QEMU:?QEMU binary}
FW=${SUBLET_FIRMWARE:?firmware}
IMAGE=${SUBLET_KERNEL:?kernel Image}
BASE=${TRAPTEST_BASE_INITRAMFS:?busybox initramfs}
mkdir -p "$OUT/root"
OUT=$(cd "$OUT" && pwd)
(cd "$OUT/root" && gzip -dc "$BASE" | cpio -idu --quiet)
for arm in fixed unfixed; do
  for port in postgres mruby perl; do
    cp "$OUT/$arm/$port.dom" "$OUT/root/$arm-$port.dom"
  done
done
cp "$HERE/init" "$OUT/root/init"
chmod +x "$OUT/root/init"
(cd "$OUT/root" && find . | cpio -o -H newc --quiet | gzip -1) > "$OUT/initramfs.cpio.gz"
exec 8>>"$CAPSTONE_QEMU_LOCK.turnstile" 9>>"$CAPSTONE_QEMU_LOCK" 7>>"$OUT/domain.lock"
flock -w 45 -x 8
flock -w 45 -s 9
flock -u 8
flock -w 45 -x 7
python3 "$HERE/boot.py" "$QEMU" "$FW" "$IMAGE" "$OUT/initramfs.cpio.gz" "$OUT/boot.log" off
python3 "$HERE/check.py" "$OUT/boot.log"
