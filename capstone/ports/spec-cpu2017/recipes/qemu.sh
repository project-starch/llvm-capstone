#!/bin/bash
# Build capstone-qemu from the source archive that manifest.lock.json names.
# usage: qemu.sh KIT WORKDIR   (WORKDIR empty or absent; gets src/, qemu-record.txt)
# meson fetches the subprojects that the source's subprojects/*.wrap files pin.
set -euo pipefail
KIT=$(cd "$1" && pwd) W=$2
[ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ] && { echo "qemu: $W is not empty" >&2; exit 2; }
mkdir -p "$W"; W=$(cd "$W" && pwd)
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$KIT/manifest.lock.json" "$1"; }
R=$W/qemu-record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
SRC=$KIT/artifacts/sources/$(m platform.qemu.archive.file)
got=$(sha256sum "$SRC" | cut -c1-64); [ "$got" = "$(m platform.qemu.archive.sha256)" ] || { echo "qemu: archive is $got" >&2; exit 1; }
note "capstone-qemu commit $(m platform.qemu.commit) archive $got"
mkdir -p "$W/src"; zstd -dc "$SRC" | tar -C "$W/src" -xf -
cd "$W/src/capstone-qemu"; mkdir build; cd build
conf=$(python3 -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['platform']['qemu']['configure']))" "$KIT/manifest.lock.json")
../configure $conf > configure.log 2>&1
make -j "${JOBS:-$(nproc)}" > make.log 2>&1
note "host $(uname -srm) cc $(cc --version | head -1)"
note "configure $conf"
for w in ../subprojects/*.wrap; do note "wrap $(basename "$w") $(sed -n 's/^revision *= *//p;s/^source_hash *= *//p' "$w" | tr '\n' ' ')"; done
note "qemu-system-riscv64 $(sha256sum qemu-system-riscv64 | cut -c1-64)"
note QEMU_DONE
