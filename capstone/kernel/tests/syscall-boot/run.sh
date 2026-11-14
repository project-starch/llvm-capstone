#!/usr/bin/env bash
# run.sh QEMU FW IMAGE NAME: build s5.S, pack it with this /init into the busybox initramfs at
# $S5TEST_BASE_INITRAMFS, boot IMAGE and print the verdict lines. The status line
# "S5TEST: s5 status 0" means every check passed; n names the first failed check of s5.S.
#   CROSS_COMPILE  RISC-V Linux tool prefix
set -euo pipefail
q=${1:?qemu} fw=${2:?firmware} image=${3:?kernel Image} name=${4:?name}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=${S5TEST_BASE_INITRAMFS:?busybox initramfs.cpio.gz}
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
"${CROSS_COMPILE:?}gcc" -nostdlib -static -march=rv64gc -mabi=lp64d -o "$work/s5" "$here/s5.S"
mkdir "$work/root"
(cd "$work/root" && zcat "$base" | cpio -id 2>/dev/null)
install -m 0755 "$work/s5" "$work/root/s5"
install -m 0755 "$here/init" "$work/root/init"
(cd "$work/root" && find . | cpio -o -H newc 2>/dev/null | gzip -9) > "$work/initramfs.cpio.gz"
python3 "$here/boot.py" "$q" "$fw" "$image" "$work/initramfs.cpio.gz" "${S5TEST_LOGS:-$work}/$name.log"
