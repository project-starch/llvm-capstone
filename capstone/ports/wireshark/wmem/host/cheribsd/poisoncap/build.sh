#!/usr/bin/env bash
# Configure and build the wmem port for CheriBSD purecap. With --poisoncap the
# PoisonCap backend replaces the backing policy; with --corpus DIR every case
# of that corpus becomes one program beside supervise.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd -- "$HERE/../../.." && pwd)
BUILD=${1:?usage: build.sh BUILD [--poisoncap] [--corpus DIR] [--jobs N]}
shift
POISON=OFF; CORPUS=""; JOBS=2
while (( $# )); do
  case $1 in
    --poisoncap) POISON=ON ;;
    --corpus) CORPUS=$2; shift ;;
    --jobs) JOBS=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
: "${CHERI_SDK:?set CHERI_SDK}" "${CHERI_SYSROOT:?set CHERI_SYSROOT}"
cmake -S "$PORT" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PORT/../../common/cmake/toolchains/cheribsd.cmake" \
  -DCHERI_SDK="$CHERI_SDK" -DCHERI_SYSROOT="$CHERI_SYSROOT" \
  -DCMAKE_BUILD_TYPE=Debug -DWM_POISONCAP=$POISON -DBUILD_TESTING=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ${CORPUS:+-DWM_CORPUS_DIR=$CORPUS}
cmake --build "$BUILD" -j "$JOBS"
