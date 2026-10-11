#!/usr/bin/env bash
# Build one program per defect through the port's one-source seam.
#
# The seam takes a single corpus source and produces a single target, so it is
# invoked once per case. The programs land side by side in OUT/bin as
# defect-NN, which is what both runners expect.
#
#   shared/build-cases.sh cheribsd OUT [extra cmake options]
#       stock CheriBSD purecap, the platform as it ships. This is the build the
#       `cheribsd` ARM of the study needs, and until 2026-10-08 it did not exist
#       here: the target of that name was the PoisonCap one below, which is our
#       adapter for a competitor's platform and which arms.json excludes on
#       purpose. The name said otherwise, so the arm looked measurable when the
#       only CheriBSD images were of something else.
#   shared/build-cases.sh poisoncap OUT [extra cmake options]
#       the PoisonCap pair, against the published platform's own SDK.
#   shared/build-cases.sh capstone-domain OUT [extra cmake options]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
TARGET=${1:?usage: build-cases.sh cheribsd|poisoncap|capstone-domain|capstone-application OUT [cmake options]}
OUT=${2:?select an output directory}
shift 2
PORT="$REPO/capstone/ports/cpython/pymalloc"
# PYC_CASES=<dir> builds another directory of cases in the same format -- controls/.
CASES=${PYC_CASES:-$CORPUS}
mkdir -p "$OUT/bin" "$OUT/work"
built=0
for dir in "$CASES"/[0-9][0-9]_*/; do
  number=$(basename "$dir" | cut -c1-2)
  work="$OUT/work/$number"
  case "$TARGET" in
  cheribsd)
    : "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
    : "${CHERI_SYSROOT:?set CHERI_SYSROOT to its matching rootfs}"
    cmake --preset cheribsd -S "$PORT" -B "$work" \
      -DCHERI_SDK="$CHERI_SDK" -DCHERI_SYSROOT="$CHERI_SYSROOT" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    cp -f "$work/bin/cheribsd-abi-probe" "$OUT/bin/" 2>/dev/null || true
    if [ ! -x "$OUT/bin/supervise" ]; then
      "$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 \
        -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin" \
        --sysroot="$CHERI_SYSROOT" -std=gnu11 -O1 -Wall -Wextra -fuse-ld=lld \
        "$CORPUS/observe/supervise.c" -lutil -o "$OUT/bin/supervise"
    fi
    ;;
  poisoncap)
    bash "$PORT/host/cheribsd/poisoncap/build.sh" "$work" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    # The runner's platform control comes from the same build.
    cp -f "$work/bin/cheribsd-abi-probe" "$OUT/bin/" 2>/dev/null || true
    # The supervisor that observes each case's fault from outside it.
    if [ ! -x "$OUT/bin/supervise" ]; then
      "$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 \
        -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin" \
        --sysroot="$CHERI_SYSROOT" -std=gnu11 -O1 -Wall -Wextra -fuse-ld=lld \
        "$CORPUS/observe/supervise.c" -lutil -o "$OUT/bin/supervise"
    fi
    ;;
  capstone-domain)
    cmake --preset capstone-domain -S "$PORT" -B "$work" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects.dom" "$OUT/bin/defect-$number.dom"
    ;;
  capstone-application)
    # The virtual profile: the hosted replay as a Capstone process on the SDK in
    # CAPSTONE_SDK. Add -DPYMALLOC_SUBLET=ON for the nested arm, where the
    # lifetime adapter owns an arena the virtual heap lends linear.
    : "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual application SDK}"
    cmake --preset capstone-application -S "$PORT" -B "$work" -DCAPSTONE_SDK="$CAPSTONE_SDK" \
      -DPY_CORPUS_SRC="$dir/case.c" "$@" >"$OUT/work/$number.log" 2>&1
    cmake --build "$work" >>"$OUT/work/$number.log" 2>&1
    cp "$work/bin/defects" "$OUT/bin/defect-$number"
    ;;
  *) echo "Unknown target: $TARGET" >&2; exit 2;;
  esac
  built=$((built + 1))
done
# No cases found must be an error, never an empty success.
if [ "$built" -eq 0 ]; then
  echo "build-cases.sh: no case directories under $CORPUS" >&2
  exit 2
fi
echo "built $built programs into $OUT/bin"
