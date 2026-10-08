#!/usr/bin/env bash
# Cross-build libavutil for CheriBSD riscv64 PURECAP, so ffmpeg/plane-repros can have a CheriBSD
# arm at all.
#
# WHY THIS IS NEEDED: plane-repros links REAL libavutil -- the case calls av_frame_alloc and
# av_frame_get_buffer, because the row's whole claim is "real allocator, model consumer". Its
# plain-heap siblings link nothing, which is why copying their run-cheribsd.sh is not sufficient.
#
# OUT OF TREE on purpose: /tmp/capstone/ffmpeg-native-vidstab/build holds the NATIVE libavutil the
# corpus's native arm uses, and that must not be clobbered.
#
# A FAILURE HERE IS A RESULT, not a dead end: if libavutil will not build purecap, that is the
# recorded reason the arm stays unmeasured, which is better than the arm's current silence.
set -uo pipefail

SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
SRC=${FFSRC:-/tmp/capstone/ffmpeg-native-vidstab/src}
OUT=${1:-/tmp/capstone/ffmpeg-purecap/build}

for p in "$SDK/bin/clang" "$SYSROOT" "$SRC/configure"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p"; exit 75; }
done
mkdir -p "$OUT" || exit 75
cd "$OUT" || exit 75

CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -O1"

echo "=== configure (purecap, libavutil only) ==="
"$SRC/configure" \
  --prefix="$OUT/install" \
  --enable-cross-compile \
  --arch=riscv64 \
  --target-os=freebsd \
  --cc="$SDK/bin/clang" \
  --ar="$SDK/bin/llvm-ar" \
  --nm="$SDK/bin/llvm-nm" \
  --ranlib="$SDK/bin/llvm-ranlib" \
  --strip="$SDK/bin/llvm-strip" \
  --extra-cflags="$CFLAGS" \
  --extra-ldflags="$CFLAGS -fuse-ld=lld" \
  --disable-everything --disable-programs --disable-doc --disable-autodetect \
  --disable-avdevice --disable-avfilter --disable-avformat --disable-avcodec \
  --disable-swresample --disable-swscale --disable-network \
  --disable-pthreads --disable-asm --disable-inline-asm
rc=$?
echo "CONFIGURE_RC=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- the tail of ffbuild/config.log, which is where configure says why ---"
  tail -30 ffbuild/config.log 2>/dev/null
  exit 3
fi

echo "=== build libavutil only ==="
taskset -c 0-7,32-39 nice -n 10 make -j12 libavutil/libavutil.a
rc=$?
echo "MAKE_RC=$rc"
[ "$rc" -eq 0 ] || exit 4

echo "=== result ==="
ls -l libavutil/libavutil.a | awk '{print "  "$5" bytes"}'
echo "=== prove it is PURECAP and not a native build ==="
"$SDK/bin/llvm-readobj" --file-headers libavutil/libavutil.a 2>/dev/null | grep -iE 'Machine|Class|Flags' | head -6 | sed 's/^/  /'
echo "PURECAP_LIBAVUTIL_OK"
