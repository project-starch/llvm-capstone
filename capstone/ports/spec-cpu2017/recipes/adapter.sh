#!/bin/bash
# Build the virtual adapter -- kernel module capstone_vm.ko, launcher capstone-vexec, capstone-job --
# from the toolchain source archive, against the kernel build tree of the guest Image.
# usage: adapter.sh KIT KERNEL_BUILD CROSS_COMPILE OUT
#   KERNEL_BUILD   the Linux build tree the guest Image came from (its arch/riscv/boot/Image must
#                  match the manifest's images/Image)
#   CROSS_COMPILE  riscv64 Linux cross-compiler prefix (buildroot host gcc)
#   OUT            new directory: capstone-vexec, capstone-job, module/capstone_vm.ko, adapter-record.txt
set -euo pipefail
KIT=$(cd "$1" && pwd) KB=$(cd "$2" && pwd) CROSS=$3 OUT=$4
[ -e "$OUT" ] && { echo "adapter: $OUT exists" >&2; exit 2; }
M=$KIT/manifest.lock.json
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$M" "$1"; }
mkdir -p "$OUT/module"; OUT=$(cd "$OUT" && pwd)
R=$OUT/adapter-record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
img=$(sha256sum "$KB/arch/riscv/boot/Image" | cut -c1-64)
[ "$img" = "$(m 'platform.files.images/Image')" ] || { echo "adapter: $KB's Image is $img, not the manifest's" >&2; exit 1; }
SRC=$KIT/artifacts/sources/$(m toolchain.archive.file)
QSRC=$KIT/artifacts/sources/$(m platform.qemu.archive.file)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
zstd -dc "$SRC" | tar -C "$W" -xf - llvm-capstone/capstone/runtime
zstd -dc "$QSRC" | tar -C "$W" -xf - capstone-qemu/target/riscv/cap_rev_table_abi.h
V=$W/llvm-capstone/capstone/runtime/virtual
cp "$V/module/Makefile" "$V/module/capstone_vm.c" "$V/wire.h" "$V/vm-abi.h" "$OUT/module/"
cp "$W/capstone-qemu/target/riscv/cap_rev_table_abi.h" "$OUT/module/"
# Build paths are mapped away and every object gets a fixed name, so the same inputs give the
# same bytes wherever this runs.
make -C "$KB" ARCH=riscv CROSS_COMPILE="$CROSS" M="$OUT/module" KCFLAGS="-ffile-prefix-map=$OUT/module/=" \
  -j8 modules > "$OUT/module.log" 2>&1
mkdir "$W/obj"
objs=()
for s in "$V/exec.c" "$V/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park,domain-fault}.c \
         "$V/../linux/stub.S" "$V/../common/"{launch,delegate,spawn,msghdr}.c; do
  o=$W/obj/$(basename "${s%.*}").o
  "${CROSS}gcc" -O2 -pthread -Wall -Wextra -Werror -ffile-prefix-map="$W/=" -I"$V/../include" -c "$s" -o "$o"
  objs+=("$o")
done
( cd "$W/obj" && "${CROSS}gcc" -static -pthread $(for o in "${objs[@]}"; do basename "$o"; done) -o "$OUT/capstone-vexec" )
( cd "$V/../linux" && "${CROSS}gcc" -O2 -static -Wall -Wextra -Werror -ffile-prefix-map="$W/=" job.c -o "$OUT/capstone-job" )
note "source llvm-capstone $(m toolchain.commit) capstone-qemu $(m platform.qemu.commit)"
note "kernel-build $KB Image $img config $(sha256sum "$KB/.config" | cut -c1-64) Module.symvers $(sha256sum "$KB/Module.symvers" | cut -c1-64)"
note "cross $("${CROSS}gcc" --version | head -1)"
for f in capstone-vexec capstone-job module/capstone_vm.ko; do note "$f $(sha256sum "$OUT/$f" | cut -c1-64)"; done
