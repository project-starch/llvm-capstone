#!/usr/bin/env bash
# Build one arm of the virtual FFmpeg 9.0.1 decoder image with the
# reuse-distance hooks (docs/plans/2026-10-11-nested-reuse-distance.md),
# without editing the port:
#
#   build.sh stock|protected OUT FFMPEG_TARBALL
#
# Environment as for ports/common/application/build-virtual.sh
# (CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_REPO_ROOT, CAPSTONE_TMP_ROOT, FFAPP_JOBS).
#
# The steps are build-virtual.sh's ffmpeg branch, with the hook patch applied
# between source preparation and the build:
# 1. prepare-source.sh (stock) or prepare-source.sh --sublet (protected; the
#    buffer-pool port's 0001 and 0003) into OUT/source, from FFMPEG_TARBALL
#    (prepare-source.sh checks it against upstream.json);
# 2. pools-reuse-gap-ARM.patch at fuzz 0 and the two headers into libavutil/;
# 3. build-domain.sh with FFAPP_SUBLET=0|1 and
#    FFAPP_EXTRA_CONFIGURE=--extra-cflags=-DEXP_REUSE_GAP (the only build
#    input that differs from the memory-behaviour images besides the patch),
#    FFAPP_EXTRA_FIXTURES="11 12 13 14 15" as there;
# 4. build.py --app ffmpeg --nested none|ffmpeg, as build-virtual.sh calls it.
# Checks that fail the build: the define is in the libraries' CFLAGS, and
# buffer.o and refstruct.o in libavutil.a carry the hook's tables.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# == 3 ]] || { echo "usage: build.sh stock|protected OUT FFMPEG_TARBALL" >&2; exit 2; }
ARM=$1 OUT=$(realpath -m "$2") TARBALL=$(realpath "$3")
case $ARM in
  stock) SUB=0 NESTED=none PREP=() ;;
  protected) SUB=1 NESTED=ffmpeg PREP=(--sublet) ;;
  *) echo "arm: stock or protected" >&2; exit 2 ;;
esac
R=${CAPSTONE_REPO_ROOT:?}
APP=$R/capstone/ports/ffmpeg/app
[[ ! -e $OUT ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
source "$R/capstone/tests/capstone-test-env.sh" >/dev/null
export CAPSTONE_APPLICATION_PROFILE=virtual FFAPP_WORK=$OUT/source FFAPP_SUBLET=$SUB
export FFAPP_EXTRA_CONFIGURE=--extra-cflags=-DEXP_REUSE_GAP FFAPP_EXTRA_FIXTURES="11 12 13 14 15"
mkdir -p "$FFAPP_WORK"
cp "$TARBALL" "$FFAPP_WORK/ffmpeg-9.0.1.tar.xz"

SRC=$(bash "$APP/host/prepare-source.sh" "${PREP[@]}" | tail -1)
cp "$R/capstone/experiments/study/reuse-gap-observer.h" "$HERE/reuse-gap-hook.h" "$SRC/libavutil/"
patch -d "$SRC" -p1 --batch --forward --fuzz=0 < "$HERE/pools-reuse-gap-$ARM.patch" > "$OUT/hook-patch.log"
echo "source $SRC"

bash "$APP/host/build-domain.sh" > "$OUT/build-domain.log" 2>&1
XB=$FFAPP_WORK/domain/ffmpeg-build
grep -q -- '-DEXP_REUSE_GAP' "$XB/ffbuild/config.mak" \
  || { echo "EXP_REUSE_GAP is not in the libraries' CFLAGS" >&2; exit 1; }
# nm to a file first: under pipefail, `nm | grep -q` fails when grep exits early.
for obj in buffer.o:buffer_pool_reuse_gap_slots refstruct.o:refstruct_pool_reuse_gap_slots; do
  "$CAPSTONE_LLVM_BIN/llvm-nm" "$XB/libavutil/${obj%%:*}" > "$OUT/${obj%%:*}.nm"
  grep -q " ${obj#*:}\$" "$OUT/${obj%%:*}.nm" \
    || { echo "libavutil/${obj%%:*} lacks ${obj#*:}" >&2; exit 1; }
done

python3 "$R/capstone/ports/common/application/build.py" --app ffmpeg --root "$FFAPP_WORK" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 1048576 --out "$OUT/image" > "$OUT/build-image.log" 2>&1
grep -q 'REUSE-GAP' "$OUT/image/ffmpeg.dom" || { echo "image lacks the report" >&2; exit 1; }
echo "image $OUT/image/ffmpeg.dom sha256=$(sha256sum "$OUT/image/ffmpeg.dom" | cut -d' ' -f1)"
