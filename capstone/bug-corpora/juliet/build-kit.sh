#!/usr/bin/env bash
# Build one consistent virtual-profile QEMU kit and SDK from ONE llvm-capstone commit, in the layout
# tools/virtualvm.py expects (qemu-system-riscv64 + libslirp, images/, adapter/), with SHA256SUMS and PROVENANCE.
#
#   build-kit.sh <work-dir> <llvm-capstone commit> [first step]
#
# Trap route (14 Nov 2026): QEMU, firmware and the kernel Image come from the trap-route build, not from this
# commit's gitlinks, so the default steps skip qemu and buildroot. The kit step writes the kit with
# capstone/runtime/virtual/trap_platform.py make-kit (QEMU, Image, fw_jump.elf from SUBLET_QEMU, SUBLET_KERNEL,
# SUBLET_FIRMWARE, defaults in trap_platform.py) and the adapter step builds only juliet-timeout: there is no module.
# CROSS_COMPILE_TRAP names the guest's Linux cross prefix (default: the caplifive buildroot host tools).
#
# Steps: clone llvm qemu buildroot sdk adapter kit (qemu and buildroot only when asked as the first step). Every step is timed into <work-dir>/timings.tsv and stops the
# script on failure; every step checks its ARTIFACTS, because the buildroot Makefile can exit 0 with a failed inner
# build. QEMU is the commit's own capstone/capstone-qemu gitlink: the runtime, module and SDK of a commit target that
# QEMU, not a newer one.
#
# Host notes: buildroot 2023.08's host-m4 does not build with GCC 15; set BR_HOSTCC/BR_HOSTCXX to an older compiler
# (e.g. gcc-12/g++-12). Rust (cargo) must be on PATH for the monitor's capstone-c step. Measured on an 80-thread host
# with -j64: clone 18 min, LLVM 7.5 min, QEMU 1 min, buildroot 22 min, SDK + adapter + kit under a minute.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:?work dir}; DEV=${2:?llvm-capstone commit}; START=${3:-clone}
STEPS="clone llvm qemu buildroot sdk adapter kit"
TRAP_SKIP=${TRAP_SKIP:-"qemu buildroot"}
TRAP_CROSS=${CROSS_COMPILE_TRAP:-$HOME/caplifive-system/sw/buildroot/build/host/bin/riscv64-buildroot-linux-gnu-}
known=0; for s in $STEPS; do [ "$s" = "$START" ] && known=1; done
[ $known = 1 ] || { echo "unknown first step '$START' (one of: $STEPS)" >&2; exit 2; }
mkdir -p "$ROOT" && ROOT=$(cd "$ROOT" && pwd)
SRC=$ROOT/src; BLLVM=$ROOT/build-llvm; OUT=$ROOT/virtual-runtime; KIT=$ROOT/kit; LOG=$ROOT/logs
J=${JOBS:-64}
mkdir -p "$LOG" "$ROOT/tmp"
TIMES=$ROOT/timings.tsv
[ -f "$TIMES" ] || printf 'step\tstart_utc\tend_utc\tseconds\trc\n' > "$TIMES"
for v in $(env | sed -n 's/^\(CAPSTONE_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CROSS_COMPILE KERNEL_BUILD MUSL_CACHE_ROOT CCC_OVERRIDE_OPTIONS
export CAPSTONE_TMP_ROOT=$ROOT/tmp

need() { local f; for f in "$@"; do [ -s "$f" ] || { echo "MISSING ARTIFACT: $f"; return 1; }; done; }
step() {
  local name=$1; shift
  local t0 s0; t0=$(date -u +%s); s0=$(date -u +%FT%TZ)
  echo "=== $name start $s0"
  "$@" > "$LOG/$name.log" 2>&1; local rc=$?
  local t1; t1=$(date -u +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$s0" "$(date -u +%FT%TZ)" "$((t1 - t0))" "$rc" >> "$TIMES"
  echo "=== $name end rc=$rc $((t1 - t0)) s"
  [ "$rc" -eq 0 ] || { echo "STOPPED at $name (rc=$rc), see $LOG/$name.log"; exit "$rc"; }
}

do_clone() {
  [ -d "$SRC/.git" ] || git clone --filter=blob:none --no-checkout https://github.com/project-starch/llvm-capstone.git "$SRC" || return 1
  git -C "$SRC" -c fetch.writeCommitGraph=false fetch origin "$DEV" || return 1
  git -C "$SRC" -c submodule.recurse=false checkout --detach "$DEV" || return 1
  git -C "$SRC" submodule update --init --filter=blob:none capstone/capstone-qemu capstone/caplifive-buildroot capstone/capstone-c || return 1
  local BR=$SRC/capstone/caplifive-buildroot
  git -C "$BR" submodule update --init --filter=blob:none buildroot components/linux components/opensbi package/capstone-sbi-domain/capstone-sbi || return 1
  git -C "$BR/components/opensbi" submodule update --init --filter=blob:none
}

do_llvm() {
  mkdir -p "$BLLVM" && cd "$BLLVM" || return 1
  cmake -G Ninja "$SRC/llvm" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=Capstone -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_PARALLEL_LINK_JOBS=4 \
    -DLLVM_INCLUDE_BENCHMARKS=OFF || return 1
  ninja -j "$J" clang lld llvm-ar llvm-ranlib llvm-objdump llvm-nm llvm-objcopy llvm-readobj llvm-strip llvm-config || return 1
  [ -e "$BLLVM/bin/llvm-readelf" ] || ln -s llvm-readobj "$BLLVM/bin/llvm-readelf"   # the corpus judge needs both names
  need "$BLLVM/bin/clang" "$BLLVM/bin/ld.lld" "$BLLVM/bin/llvm-nm" "$BLLVM/bin/llvm-readelf"
}

do_qemu() {
  local Q=$SRC/capstone/capstone-qemu
  mkdir -p "$Q/build" && cd "$Q/build" || return 1
  ../configure --target-list=riscv64-softmmu --disable-docs --disable-werror --enable-slirp || return 1
  ninja -j "$J" || return 1
  need "$Q/build/qemu-system-riscv64"
}

do_buildroot() {
  cd "$SRC/capstone/caplifive-buildroot" || return 1
  export CAPSTONE_CC_PATH=$SRC/capstone/capstone-c
  export HOSTCC=${BR_HOSTCC:-gcc} HOSTCXX=${BR_HOSTCXX:-g++}
  rm -rf build-qemu
  make TARGET=qemu JOBS="$J" setup && make TARGET=qemu JOBS="$J" build || return 1
  local BRO=$PWD/build-qemu
  need "$BRO/images/Image" "$BRO/images/fw_jump.elf" "$BRO/images/rootfs.ext2" "$BRO/build/linux-custom/Module.symvers" || return 1
  ls "$BRO"/host/bin/riscv64-*linux-gcc > /dev/null || { echo "MISSING ARTIFACT: cross gcc"; return 1; }
}

do_sdk() {
  cd "$SRC" || return 1
  export CAPSTONE_LLVM_BUILD_DIR=$BLLVM
  source capstone/tests/capstone-test-env.sh || return 1
  export MUSL_CACHE_ROOT=$OUT/musl-source
  local prep musl
  prep=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh) || return 1   # status first, then the path
  musl=${prep##*$'\n'}
  bash capstone/runtime/virtual/build-sdk.sh "$OUT" "$musl" || return 1
  need "$OUT/sdk/sdk.json" "$OUT/sdk/capstone-cc"
}

do_adapter() {
  mkdir -p "$OUT" || return 1
  "${TRAP_CROSS}gcc" -O2 -static -Wall -Wextra -Werror "$HERE/juliet-timeout.c" -o "$OUT/juliet-timeout" || return 1
  need "$OUT/juliet-timeout"
}

do_kit() {
  rm -rf "$KIT" || return 1
  python3 "$SRC/capstone/runtime/virtual/trap_platform.py" make-kit "$KIT" || return 1
  local head clang cc helper
  head=$(git -C "$SRC" rev-parse HEAD) || return 1
  clang=$(sha256sum "$BLLVM/bin/clang") && cc=$(sha256sum "$OUT/sdk/capstone-cc") \
    && helper=$(sha256sum "$OUT/juliet-timeout") || return 1
  printf 'llvm-capstone %s\ntrap route kit (trap_platform.py make-kit, SHA256SUMS)\nclang %s\ncapstone-cc %s\njuliet-timeout %s\n' \
    "$head" "${clang%% *}" "${cc%% *}" "${helper%% *}" > "$KIT/PROVENANCE"
}

echo "build-kit.sh dev=$DEV start $(date -u +%FT%TZ)"
go=0
for s in $STEPS; do
  [ "$s" = "$START" ] && go=1
  if [ $go = 1 ] && [ "$s" != "$START" ] && [[ " $TRAP_SKIP " == *" $s "* ]]; then
    echo "=== $s skipped (trap route: TRAP_SKIP)"
  elif [ $go = 1 ]; then step "$s" "do_$s"; else echo "=== $s skipped (START=$START)"; fi
done
echo "ALL DONE $(date -u +%FT%TZ)"
