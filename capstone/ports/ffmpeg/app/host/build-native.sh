#!/usr/bin/env bash
# The native half: the ORACLE the domain is judged against, and the check that the
# oracle can fail.
#
#   1. A stock ffmpeg, configured exactly as the buffer-pool recordings' native binary
#      (ports/ffmpeg/buffer-pool/src/native/ffmpeg/CMakeLists.txt:20-28), produces the
#      workload and the reference with the recording's own command lines
#      (ports/ffmpeg/buffer-pool/host/record.sh:23-34). Only the "short" workload:
#      1 s, 320x180, 30 frames -- the one whose image fits a domain (see the plan).
#   2. The minimal library set the domain uses is built for the host, and the same decode
#      core the domain runs (src/shared/ffapp_decode.c) is linked against it.
#   3. That program's hash column must equal the reference: ORACLE CHECK.
#   4. A one-byte-flipped input must CHANGE hashes: POSITIVE CONTROL. A comparison that
#      cannot fail proves nothing.
# Exit status is non-zero unless both hold.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
JOBS=${FFAPP_JOBS:-48}
SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" | tail -1)
OUT="$WORK/native"
mkdir -p "$OUT/stock" "$OUT/minimal"

# --- 1. stock ffmpeg and the workload -------------------------------------------------
if [ ! -x "$OUT/stock/ffmpeg" ]; then
  ( cd "$OUT/stock" && "$SRC/configure" \
      --disable-everything --disable-autodetect --disable-doc --disable-network \
      --disable-x86asm --disable-debug --disable-ffplay --disable-ffprobe --enable-ffmpeg \
      --enable-protocol=file,pipe --enable-indev=lavfi \
      --enable-filter=testsrc2,scale,hflip,format,null,anull \
      --enable-encoder=mpeg4,rawvideo --enable-decoder=mpeg4,wrapped_avframe \
      --enable-parser=mpeg4video --enable-muxer=matroska,framemd5,md5,null \
      --enable-demuxer=matroska --enable-static --disable-shared > configure.log 2>&1 \
    && make -j"$JOBS" ffmpeg > build.log 2>&1 )
fi
COMMON=(-nostdin -hide_banner -loglevel warning -threads 1 -filter_threads 1 -filter_complex_threads 1)
"$OUT/stock/ffmpeg" "${COMMON[@]}" -f lavfi -i "testsrc2=size=320x180:rate=30:duration=1" \
  -c:v mpeg4 -q:v 3 -y "$WORK/input.mkv" 2>/dev/null
"$OUT/stock/ffmpeg" "${COMMON[@]}" -i "$WORK/input.mkv" -f framemd5 -y "$WORK/stock.framemd5"

# The positive-control input: one byte flipped in the middle of the payload.
python3 - "$WORK/input.mkv" "$WORK/input.flip.mkv" <<'PY'
import sys
b = bytearray(open(sys.argv[1], 'rb').read()); b[len(b) // 2] ^= 0x01
open(sys.argv[2], 'wb').write(b)
PY

# --- 2. the minimal library set, native ------------------------------------------------
if [ ! -f "$OUT/minimal/libavformat/libavformat.a" ]; then
  ( cd "$OUT/minimal" && "$SRC/configure" \
      --disable-everything --disable-autodetect --disable-doc --disable-network --disable-asm \
      --disable-pthreads --disable-programs --disable-debug --disable-iconv \
      --disable-swresample --disable-swscale --disable-avfilter --disable-avdevice \
      --enable-demuxer=matroska --enable-decoder=mpeg4 --enable-parser=mpeg4video \
      --enable-protocol=file --enable-static --disable-shared > configure.log 2>&1 \
    && make -j"$JOBS" > build.log 2>&1 )
fi
cc -O1 -I"$OUT/minimal" -I"$SRC" -I"$APP_DIR/src/shared" \
  "$APP_DIR/src/native/ffapp_native.c" "$APP_DIR/src/shared/ffapp_decode.c" \
  "$OUT/minimal/libavformat/libavformat.a" "$OUT/minimal/libavcodec/libavcodec.a" \
  "$OUT/minimal/libavutil/libavutil.a" -lm -o "$OUT/ffapp_native"

# --- 3 and 4. oracle check and positive control ----------------------------------------
"$OUT/ffapp_native" "$WORK/input.mkv"      > "$WORK/native.out"
"$OUT/ffapp_native" "$WORK/input.flip.mkv" > "$WORK/native.flip.out" || true
python3 "$SCRIPT_DIR/compare-md5.py" "$WORK/stock.framemd5" "$WORK/native.out" \
  --control "$WORK/native.flip.out"
printf 'native oracle ready in %s\n' "$WORK"
