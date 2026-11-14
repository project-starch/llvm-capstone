#!/usr/bin/env bash
# run.sh QEMU FW IMAGE NAME [SSTC]: build the test program, pack it with this /init into the
# busybox initramfs at $TRAPTEST_BASE_INITRAMFS, boot IMAGE and print the verdict lines.
#   CROSS_COMPILE  RISC-V Linux tool prefix
set -euo pipefail
q=${1:?qemu} fw=${2:?firmware} image=${3:?kernel Image} name=${4:?name} sstc=${5:-off}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=${TRAPTEST_BASE_INITRAMFS:?busybox initramfs.cpio.gz}
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
"${CROSS_COMPILE:?}gcc" -nostdlib -static -march=rv64gc -mabi=lp64d -o "$work/regs" "$here/regs.S"
mkdir "$work/root"
(cd "$work/root" && zcat "$base" | cpio -id 2>/dev/null)
install -m 0755 "$work/regs" "$work/root/regs"
install -m 0755 "$here/init" "$work/root/init"
(cd "$work/root" && find . | cpio -o -H newc 2>/dev/null | gzip -9) > "$work/initramfs.cpio.gz"
python3 "$here/boot.py" "$q" "$fw" "$image" "$work/initramfs.cpio.gz" "${TRAPTEST_LOGS:-$work}/$name.log" "$sstc"
