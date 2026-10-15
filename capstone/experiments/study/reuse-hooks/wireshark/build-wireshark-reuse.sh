#!/usr/bin/env bash
# One tshark (Wireshark 4.6.8) image with the reuse-distance observer at wmem's block and
# block_fast allocators (pre-registration capstone/docs/plans/2026-10-11-nested-reuse-distance.md,
# 16d2171a3edc).
#
#   build-wireshark-reuse.sh stock|protected NEW_OUTPUT
#
# The tshark recipe of ports/common/application/build-virtual.sh, unchanged, with one step inserted
# between the cross build and the domain link: wmem-reuse-gap.patch is applied to the cross build's
# source at fuzz 0, the two hook headers are copied beside the allocators, and the two allocator
# objects are recompiled with ninja's own compile command plus -DEXP_REUSE_GAP and replaced in
# libwsutil.a. (A second ninja run would re-run CMake and recompile far more: cross-build.sh.)
# protected = TSAPP_SUBLET=1 (wmem patch 0001); the same hook patch serves both arms.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../../.." && pwd)
source "$CAP/tests/capstone-test-env.sh" >/dev/null
[[ $# == 2 ]] || { echo 'usage: build-wireshark-reuse.sh stock|protected NEW_OUTPUT' >&2; exit 2; }
ARM=$1
OUT=$(realpath -m "$2")
case "$ARM" in
  stock) unset TSAPP_SUBLET; NESTED=none LABEL= ;;
  protected) export TSAPP_SUBLET=1; NESTED=wireshark LABEL=-DEXP_REUSE_GAP_PROTECTED ;;
  *) echo "arm: stock or protected, not $ARM" >&2; exit 2 ;;
esac
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT/source"
export CAPSTONE_APPLICATION_PROFILE=virtual TS_WORK=$OUT/source
PORTS=$CAPSTONE_REPO_ROOT/capstone/ports
PATCH=$HERE/wmem-reuse-gap.patch

for lib in zlib pcre2 c-ares libgpg-error libgcrypt libxml2 glib; do
  bash "$PORTS/wireshark/app/deps/build-$lib.sh"
done
bash "$PORTS/wireshark/app/host/cross-build.sh"

source "$PORTS/wireshark/app/deps/env.sh"
SRC=$TS_WORK/xsrc B=$TS_WORK/xbuild
patch -d "$SRC" --batch --forward --fuzz=0 -p1 < "$PATCH"
cp "$HERE/wmem-reuse-gap.h" "$CAP/experiments/study/reuse-gap-observer.h" "$SRC/wsutil/wmem/"
for unit in wmem_allocator_block wmem_allocator_block_fast; do
  obj=wsutil/CMakeFiles/wsutil.dir/wmem/$unit.c.o
  cmd=$(ninja -C "$B" -t commands "$obj" | tail -1)
  [[ $cmd == *" -o $obj -c $SRC/wsutil/wmem/$unit.c" ]] \
    || { echo "unexpected compile command for $obj: ${cmd: -200}" >&2; exit 2; }
  ( cd "$B" && eval "${cmd/ -o $obj / -DEXP_REUSE_GAP $LABEL -o $obj }" )
  "$CAPSTONE_LLVM_BIN/llvm-ar" r "$B/run/libwsutil.a" "$B/$obj"
done
"$CAPSTONE_LLVM_BIN/llvm-nm" "$B/run/libwsutil.a" 2>/dev/null | grep -c ' wmem_rg_report$' > /dev/null \
  || { echo "libwsutil.a has no reuse hooks" >&2; exit 2; }
echo "[reuse] $ARM: applied $(basename "$PATCH"), EXP_REUSE_GAP in the two wmem allocators" >&2

bash "$PORTS/wireshark/app/host/build-domain.sh"
python3 "$PORTS/common/application/build.py" --app tshark --root "$TS_WORK" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 1048576 --musl "$TS_MUSL" --libc "$TS_LIBC_ARCHIVE" \
  --out "$OUT/image"
# The hooks must be in the image: their report string is.
grep -c "REUSE-GAP %s arm=%s" "$OUT/image/tshark.dom" >/dev/null \
  || { echo "the image has no REUSE-GAP report" >&2; exit 2; }
sha256sum "$PATCH" "$HERE/wmem-reuse-gap.h" "$SRC/wsutil/wmem/wmem_allocator_block.c" \
  "$SRC/wsutil/wmem/wmem_allocator_block_fast.c" "$OUT/image/tshark.dom" > "$OUT/reuse-build.sha256"
echo "[reuse] $ARM image $OUT/image/tshark.dom" >&2
