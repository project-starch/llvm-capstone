#!/usr/bin/env bash
# Build the M-infra gate: the domains, the guest loader, and the three kernel modules under test.
#   build.sh [OUT]      (default $CAPSTONE_TMP_ROOT/cma-domain-block)
# Modules are built OUT OF TREE from commits of the caplifive-buildroot repository, against the
# guest kernel tree of the main clone's buildroot build. Control: the d04bd83 module built this
# way is byte-identical to the shipped build/target/capstone.ko.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../../../tests/capstone-test-env.sh"
OUT=${1:-$CAPSTONE_TMP_ROOT/cma-domain-block}; S=$OUT/share; mkdir -p "$S"
BR=${CAPSTONE_BUILDROOT_DIR:?}; REPO=$CAPSTONE_REPO_ROOT
BIN=$CAPSTONE_LLVM_BIN; T=(-target capstone64-unknown-elf -ffreestanding)

"$BIN/clang" "${T[@]}" -c "$REPO/capstone/my_first_domain/start.S" -o "$S/start.o"
"$BIN/clang" "${T[@]}" -O0 -c "$REPO/capstone/my_first_domain/main.c" -o "$S/main42.o"
"$BIN/ld.lld" -T "$REPO/capstone/my_first_domain/link.ld" -o "$S/small.dom" "$S/start.o" "$S/main42.o"
dom() {  # name declared-data frame
  "$BIN/clang" "${T[@]}" -O0 -DFRAME="$3" -c "$HERE/touch.c" -o "$S/$1.o"
  "$BIN/clang" "${T[@]}" -O0 -DCAPSTONE_DOMREQ_DATA="$2" -DCAPSTONE_DOMREQ_STACK="$2" \
    -c "$REPO/capstone/tests/runtime-qemu/domreq.S" -o "$S/$1-req.o"
  "$BIN/ld.lld" -T "$REPO/capstone/my_first_domain/link.ld" -o "$S/$1.dom" "$S/start.o" "$S/$1.o" "$S/$1-req.o"
}
dom smalldecl $((256 * 1024)) $((192 * 1024))
dom big128    $((100 << 20)) $((96 << 20))
dom corruptwrap 0xffffffffffffffff 4096
dom corrupthuge $((1 << 40)) 4096
# edge64: code_len + 8 KiB + data lands 4 KiB under 64 MiB. code_len from a first link.
dom edge64 $((1 << 20)) 4096
CODE=$("$BIN/llvm-readelf" -lW "$S/edge64.dom" | awk '$1=="LOAD" && $7 ~ /E/ {print strtonum($6)}' | head -1)
DATA=$(( (64 << 20) - CODE - 8192 - 4096 ))
dom edge64 "$DATA" $(( DATA - 16384 ))
echo "edge64 code_len=$CODE declared=$DATA frame=$(( DATA - 16384 ))" | tee "$S/edge64.params"

"$BR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" -O2 -Werror=implicit-function-declaration \
  -I"$BR/package/modcapstone/include" -I"$BR/package/modcapstone/userspace" \
  -o "$S/gateload.user" "$HERE/gateload.c" "$BR/package/modcapstone/userspace/lib/libcapstone.c"

for spec in "orig d04bd83" "cma 7440cfc" "fix a74a856"; do
  set -- $spec; W=$OUT/kmod-$1; rm -rf "$W"; mkdir -p "$W"
  git -C "$BR" archive "$2" package/modcapstone | tar -x -C "$W"
  make -s -C "$BR/build/build/linux-custom" M="$W/package/modcapstone/module" ARCH=riscv \
    CROSS_COMPILE="$BR/build/host/bin/riscv64-buildroot-linux-gnu-" modules > "$W.log" 2>&1
  cp "$W/package/modcapstone/module/capstone.ko" "$S/$1.ko"
done
( cd "$S" && sha256sum *.dom gateload.user *.ko )
