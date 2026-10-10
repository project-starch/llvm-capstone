#!/usr/bin/env bash
# Build one program per defect through the port's one-source seam.
#
#   shared/build-cases.sh native OUT [cmake options]
#       one port build, then each case.c linked with the native driver:
#       OUT/bin/defect-NN, run as `defect-NN buggy|fixed NN`
#   shared/build-cases.sh cheribsd OUT [cmake options]
#       the seam once per case for CheriBSD purecap: OUT/bin/defect-NN, plus
#       the platform's ABI probe, the revocation control and the supervisor
#       that observes each program from outside. CHERI_SDK and CHERI_SYSROOT
#       must name a matching pair.
#   shared/build-cases.sh poisoncap OUT [cmake options]
#       the same, against the port's PoisonCap adapter instead of the
#       platform's malloc: OUT/bin/defect-NN choose spatial or protected from
#       the mode argument, and the platform's own instruction probe comes
#       along as the control. CHERI_SDK and CHERI_SYSROOT must be the
#       published PoisonCap pair.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
TARGET=${1:?usage: build-cases.sh native|cheribsd|poisoncap|capstone-application OUT [cmake options]}
OUT=${2:?select an output directory}
shift 2
PORT="$REPO/capstone/ports/memcached/allocators"
mkdir -p "$OUT/bin" "$OUT/work"
built=0
case "$TARGET" in
native)
  work="$OUT/work/port"
  cmake --preset native -S "$PORT" -B "$work" "$@" >"$OUT/work/port.log" 2>&1
  cmake --build "$work" >>"$OUT/work/port.log" 2>&1
  SRC=$(ls -d "$work"/source/memcached-*)
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    # The same two knobs the port builds with, so the class table and the
    # header sizes agree across the seam (see the port's cmake/Allocators.cmake).
    "${CC:-cc}" -std=c11 -O1 -g -Wall -Wextra -pthread -DNDEBUG -DCHUNK_ALIGN_BYTES=16 \
      -I"$CORPUS/shared" -I"$PORT/src/shared" -I"$SRC" -I"$PORT/../adapted" \
      -I"$REPO/capstone/runtime/include" \
      -o "$OUT/bin/defect-$number" "$dir/case.c" "$CORPUS/shared/driver.c" \
      "$work/libmemcached-allocators.a"
    built=$((built + 1))
  done
  ;;
cheribsd)
  : "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
  : "${CHERI_SYSROOT:?set CHERI_SYSROOT to its matching rootfs}"
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    work="$OUT/work/$number"
    cmake --preset cheribsd -S "$PORT" -B "$work" \
      -DMCP_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    # The runner's platform controls come from the same build.
    cp -f "$work/bin/cheribsd-abi-probe" "$work/bin/revocation-control" "$OUT/bin/"
    # The supervisor that observes each program's fault from outside it: the
    # pymalloc corpus's, built with this corpus's label. Referenced, not copied.
    if [ ! -x "$OUT/bin/supervise" ]; then
      "$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 \
        -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin" \
        --sysroot="$CHERI_SYSROOT" -std=gnu11 -O1 -Wall -Wextra -fuse-ld=lld \
        -DPROBE_SYMBOL='"mc_defect_read"' \
        "$REPO/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" \
        -lutil -o "$OUT/bin/supervise"
    fi
    built=$((built + 1))
  done
  ;;
poisoncap)
  : "${CHERI_SDK:?set CHERI_SDK to the published PoisonCap SDK}"
  : "${CHERI_SYSROOT:?set CHERI_SYSROOT to its matching rootfs}"
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    work="$OUT/work/$number"
    bash "$PORT/host/cheribsd/poisoncap/build.sh" "$work" \
      -DMCP_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    # The runner's controls come from the same build: the shared ABI probe and
    # the platform's own poison/sweep instruction probe.
    cp -f "$work/bin/cheribsd-abi-probe" "$work/bin/poisoncap-probe" "$OUT/bin/"
    if [ ! -x "$OUT/bin/supervise" ]; then
      "$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 \
        -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin" \
        --sysroot="$CHERI_SYSROOT" -std=gnu11 -O1 -Wall -Wextra -fuse-ld=lld \
        -DPROBE_SYMBOL='"mc_defect_read"' \
        "$REPO/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" \
        -lutil -o "$OUT/bin/supervise"
    fi
    built=$((built + 1))
  done
  ;;
capstone-application)
  # The virtual Capstone process build, for the virtual-malloc and virtual-nested-pools arms: the
  # port's capstone-application preset on the virtual SDK (CAPSTONE_SDK, CAPSTONE_LLVM_BUILD_DIR),
  # then each case linked as the native arms are, with the SDK's capstone-cc. Built with
  # -DMCP_SUBLET=ON the allocators carry the application's patch 0006 and no ledger, the cases are
  # compiled with MC_CAPSTONE_SUBLET (the item layout gains its parent slot) and the driver is
  # shared/driver-virtual.c. Otherwise it is driver.c, unchanged. The Sublet build also takes the
  # carve remedy (MC_CARVE_BOUNDS, the identity for every case but 06 and 07), because the physical
  # column 3 read `sublet-carve` for those two: the nested column keeps that remedy.
  : "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual Capstone application SDK}"
  work="$OUT/work/port"
  cmake --preset capstone-application -S "$PORT" -B "$work" "$@" >"$OUT/work/port.log" 2>&1
  cmake --build "$work" >>"$OUT/work/port.log" 2>&1
  SRC=$(ls -d "$work"/source/memcached-*)
  driver="$CORPUS/shared/driver.c"
  carve=()
  if grep -q '^MCP_SUBLET:BOOL=ON$' "$work/CMakeCache.txt"; then
    driver="$CORPUS/shared/driver-virtual.c"; carve=(-DMC_CARVE_BOUNDS -DMC_CAPSTONE_SUBLET)
  fi
  for dir in "$CORPUS"/[0-9][0-9]_*/; do
    number=$(basename "$dir" | cut -c1-2)
    "$CAPSTONE_SDK/capstone-cc" -std=c11 -O1 -g -Wall -Wextra -DNDEBUG -DCHUNK_ALIGN_BYTES=16 "${carve[@]}" \
      -I"$CORPUS/shared" -I"$PORT/src/shared" -I"$SRC" -I"$PORT/../adapted" \
      -I"$REPO/capstone/runtime/include" \
      -o "$OUT/bin/defect-$number" "$dir/case.c" "$driver" "$work/libmemcached-allocators.a"
    built=$((built + 1))
  done
  ;;
*) echo "Unknown target: $TARGET" >&2; exit 2;;
esac
# No cases found must be an error, never an empty success.
if [ "$built" -eq 0 ]; then
  echo "build-cases.sh: no case directories under $CORPUS" >&2
  exit 2
fi
echo "built $built programs into $OUT/bin"
