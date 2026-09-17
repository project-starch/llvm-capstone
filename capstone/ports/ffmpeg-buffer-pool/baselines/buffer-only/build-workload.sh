#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../runtime/prepare.sh" native
ARM=${1:?usage: build-workload.sh stock|traced}
case "$ARM" in stock|traced) ;; *) exit 2;; esac
SRC="$WORK/workload-src-$ARM/ffmpeg-9.0.1"
OUT="$WORK/workload-$ARM"
ARCHIVE="$WORK/download/ffmpeg-9.0.1.tar.xz"
printf '%s  %s\n' cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635 "$ARCHIVE" | sha256sum -c -
if [[ ! -d "$SRC" ]]; then
    mkdir -p "$(dirname "$SRC")"
    tar -xf "$ARCHIVE" -C "$(dirname "$SRC")"
    if [[ "$ARM" == traced ]]; then python3 "$HERE/instrument.py" "$SRC"; fi
fi
mkdir -p "$OUT"
cd "$OUT"
if [[ ! -f config.h ]] || [[ "$HERE/build-workload.sh" -nt config.h ]]; then
    "$SRC/configure" --disable-everything --disable-autodetect \
        --disable-doc --disable-network --disable-x86asm --disable-debug \
        --disable-ffplay --disable-ffprobe --enable-ffmpeg \
        --enable-protocol=file,pipe --enable-indev=lavfi \
        --enable-filter=testsrc2,scale,hflip,format,null,anull \
        --enable-encoder=mpeg4,rawvideo --enable-decoder=mpeg4,wrapped_avframe \
        --enable-parser=mpeg4video --enable-muxer=matroska,framemd5,md5,null \
        --enable-demuxer=matroska --enable-static --disable-shared > configure.log 2>&1
fi
make -j"${JOBS:-4}" ffmpeg > build.log 2>&1
./ffmpeg -version > version.txt
sha256sum ffmpeg "$SRC/libavutil/buffer.c" > identity.sha256
echo "built $OUT/ffmpeg"
