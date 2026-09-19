#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD=${1:?usage: build.sh BUILD [extra CMake options]}
shift
: "${CHERI_SDK:?set CHERI_SDK to the PoisonCap SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to its matching rootfs}"
cmake -S "$HERE/../../.." -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/../../../../../common/cmake/toolchains/cheribsd.cmake" \
  -DCHERI_SDK="$CHERI_SDK" -DCHERI_SYSROOT="$CHERI_SYSROOT" \
  -DCMAKE_BUILD_TYPE=Debug -DPYMALLOC_POISONCAP=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DBUILD_TESTING=OFF "$@"
cmake --build "$BUILD" -j "${BUILD_JOBS:-2}"
