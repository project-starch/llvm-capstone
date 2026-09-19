#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
source "$REPO/capstone/tests/capstone-test-env.sh"
WORK=${1:?usage: platform.sh WORK llvm|qemu|cheribsd|image}
STAGE=${2:?select a build stage}
JOBS=${BUILD_JOBS:-3}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
case "$WORK/" in "$REPO/"*) echo "Use a build directory outside the repository" >&2; exit 2;; esac
case "$STAGE" in
llvm)
  cmake -G Ninja -S "$WORK/source/llvm-project/llvm" -B "$WORK/build/llvm" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
    -DCMAKE_INSTALL_PREFIX="$WORK/sdk" '-DLLVM_ENABLE_PROJECTS=clang;lld' \
    '-DLLVM_TARGETS_TO_BUILD=RISCV;X86' -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_USE_LINKER=lld
  cmake --build "$WORK/build/llvm" -j "$JOBS" --target clang lld llvm-ar llvm-ranlib \
    llvm-objdump llvm-readelf llvm-nm llvm-strip llvm-objcopy llvm-size llvm-strings \
    llvm-link clang-resource-headers
  mkdir -p "$WORK/sdk/bin" "$WORK/sdk/lib"
  for tool in clang clang++ clang-cpp ld.lld lld llvm-ar llvm-ranlib llvm-objdump \
      llvm-readelf llvm-nm llvm-strip llvm-objcopy llvm-size llvm-strings llvm-link; do
    test -e "$WORK/sdk/bin/$tool" || ln -s "$WORK/build/llvm/bin/$tool" "$WORK/sdk/bin/$tool"
  done
  test -e "$WORK/sdk/lib/clang" || ln -s "$WORK/build/llvm/lib/clang" "$WORK/sdk/lib/clang"
  ;;
qemu)
  mkdir -p "$WORK/build/qemu"
  cd "$WORK/build/qemu"
  "$WORK/source/qemu/configure" --target-list=riscv64cheri-softmmu \
    --prefix="$WORK/sdk" --cc=clang-18 --cxx=clang++-18 --disable-werror \
    --disable-docs --disable-capstone --disable-gtk --disable-sdl --disable-opengl \
    --enable-slirp=git --disable-linux-user --disable-bsd-user --with-git-submodules=ignore
  ninja -j "$JOBS" qemu-system-riscv64xcheri qemu-img
  mkdir -p "$WORK/sdk/bin"
  test -e "$WORK/sdk/bin/qemu-system-riscv64cheri" || \
    ln -s "$WORK/build/qemu/qemu-system-riscv64xcheri" "$WORK/sdk/bin/qemu-system-riscv64cheri"
  test -e "$WORK/sdk/bin/qemu-img" || ln -s "$WORK/build/qemu/qemu-img" "$WORK/sdk/bin/qemu-img"
  if [[ -n "${CHERI_BOOT_FIRMWARE:-}" ]]; then
    mkdir -p "$WORK/sdk/share/qemu"
    cp "$CHERI_BOOT_FIRMWARE" "$WORK/sdk/share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"
  fi
  ;;
cheribsd|image)
  : "${CHERIBUILD:?set CHERIBUILD to the cheribuild.py entry point}"
  common=(--skip-update --source-root "$WORK/source" --build-root "$WORK/build" --tools-root "$WORK"
    --output-root "$WORK/output" --cheribsd/source-directory "$WORK/source/cheribsd"
    --cheribsd/toolchain custom --cheribsd/toolchain-path "$WORK/sdk" --make-jobs "$JOBS")
  if [[ "$STAGE" == cheribsd ]]; then
    "${PYTHON:-python3}" "$CHERIBUILD" cheribsd-riscv64-purecap "${common[@]}" \
      --kernel-config CHERI-PURECAP-QEMU-POISON --cheribsd/no-build-tests \
      --cheribsd/build-options "WITHOUT_MAN=1 WITHOUT_ZFS=1 WITHOUT_CDDL=1 WITHOUT_MAIL=1 WITHOUT_SENDMAIL=1 WITHOUT_EXAMPLES=1 WITHOUT_LOCALES=1 WITHOUT_NLS=1"
  else
    "${PYTHON:-python3}" "$CHERIBUILD" disk-image-riscv64-purecap "${common[@]}"
  fi
  ;;
*) echo "Unknown stage: $STAGE" >&2; exit 2;;
esac
