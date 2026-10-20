#!/bin/bash
# Assemble the platform directory tools/run.sh boots, from the kit's artifacts and the QEMU build,
# and check it against manifest.lock.json.
# usage: platform-dir.sh KIT QEMU_WORKDIR OUT
set -euo pipefail
KIT=$(cd "$1" && pwd) Q=$(cd "$2" && pwd) OUT=$3
[ -e "$OUT" ] && { echo "platform-dir: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT/images"
cp "$Q/src/capstone-qemu/build/qemu-system-riscv64" "$OUT/"
cp "$KIT/artifacts/platform/images/Image" "$KIT/artifacts/platform/images/fw_jump.elf" "$OUT/images/"
zstd -dc "$KIT/artifacts/platform/images/rootfs.ext2.zst" > "$OUT/images/rootfs.ext2"
cp -r "$KIT/artifacts/platform/adapter" "$OUT/adapter"
mkdir -p "$OUT/ssh-server"; cp "$KIT/artifacts/platform/ssh-server/dropbearmulti" "$OUT/ssh-server/"
python3 "$KIT/tools/check-platform.py" "$KIT/manifest.lock.json" "$OUT"
