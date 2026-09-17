#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../build/ffmpeg-source.sh"

# Keep stock and instrumented decoder sources and builds separate.
DECODER_VARIANT=${1:-traced}
case "$DECODER_VARIANT" in
    stock)
        DECODER_SOURCE_DIR="$WORK_DIR/workload-src-stock/ffmpeg-$FFMPEG_VERSION"
        DECODER_BUILD_DIR="$WORK_DIR/workload-stock"
        ;;
    traced)
        DECODER_SOURCE_DIR="$TRACED_SOURCE_DIR"
        DECODER_BUILD_DIR="$WORK_DIR/combined-workload"
        ;;
    *)
        echo "usage: $0 stock|traced" >&2
        exit 2
        ;;
esac

# The full decoder uses FFmpeg's configure script, not the replay configuration.
fetch_ffmpeg_source
if [[ ! -d "$DECODER_SOURCE_DIR" ]]; then
    mkdir -p "$(dirname "$DECODER_SOURCE_DIR")"
    tar -xf "$FFMPEG_ARCHIVE" -C "$(dirname "$DECODER_SOURCE_DIR")"
    if [[ "$DECODER_VARIANT" == traced ]]; then
        python3 "$HERE/instrument-pools.py" "$DECODER_SOURCE_DIR"
    fi
fi

# Configure the codecs and filters needed by the recording workload.
mkdir -p "$DECODER_BUILD_DIR"
cd "$DECODER_BUILD_DIR"
if [[ ! -f config.h ]]; then
    "$DECODER_SOURCE_DIR/configure" --disable-everything --disable-autodetect \
        --disable-doc --disable-network --disable-x86asm --disable-debug \
        --disable-ffplay --disable-ffprobe --enable-ffmpeg \
        --enable-protocol=file,pipe --enable-indev=lavfi \
        --enable-filter=testsrc2,scale,hflip,format,null,anull \
        --enable-encoder=mpeg4,rawvideo --enable-decoder=mpeg4,wrapped_avframe \
        --enable-parser=mpeg4video --enable-muxer=matroska,framemd5,md5,null \
        --enable-demuxer=matroska --enable-static --disable-shared > configure.log 2>&1
fi

make -j"${JOBS:-4}" ffmpeg > build.log 2>&1

# Keep binary and source identities beside the build output.
./ffmpeg -version > version.txt
sha256sum ffmpeg "$DECODER_SOURCE_DIR/libavutil/buffer.c" \
    "$DECODER_SOURCE_DIR/libavutil/refstruct.c" > identity.sha256
echo "built $DECODER_BUILD_DIR/ffmpeg"
