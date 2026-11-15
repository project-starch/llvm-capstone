#!/usr/bin/env bash
# The unit test of the PostgreSQL port's layout fix, in one short trap-route boot.
#
#   run.sh OUT
#
# OUT holds the two images of ports/common/tests/lifetime-failures/build.py postgres
# --baseline 5060011844e: fixed/postgres.dom with this tree's patch 0003 and
# unfixed/postgres.dom with the patch of 5060011844e, which kept the block's reference in
# the chunk header (the negative control). check.py states what each image must show.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../../.." && pwd)
LIFE=$REPO/capstone/ports/common/tests/lifetime-failures
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
  cp "$OUT/$arm/postgres.dom" "$OUT/root/$arm-postgres.dom"
done
cp "$HERE/init" "$OUT/root/init"
chmod +x "$OUT/root/init"
(cd "$OUT/root" && find . | cpio -o -H newc --quiet | gzip -1) > "$OUT/initramfs.cpio.gz"
exec 8>>"$CAPSTONE_QEMU_LOCK.turnstile" 9>>"$CAPSTONE_QEMU_LOCK" 7>>"$OUT/domain.lock"
flock -w 45 -x 8
flock -w 45 -s 9
flock -u 8
flock -w 45 -x 7
python3 "$LIFE/boot.py" "$QEMU" "$FW" "$IMAGE" "$OUT/initramfs.cpio.gz" "$OUT/boot.log" off
python3 "$HERE/check.py" "$OUT/boot.log"
