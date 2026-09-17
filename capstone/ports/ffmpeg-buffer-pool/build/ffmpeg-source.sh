#!/usr/bin/env bash
# Shared source paths. Call fetch_ffmpeg_source before building or extracting copies.

FFPOOL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$FFPOOL_ROOT/../../tests/capstone-test-env.sh"

WORK_DIR=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
FFMPEG_VERSION=9.0.1
FFMPEG_SHA256=cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635

FFMPEG_ARCHIVE="$WORK_DIR/download/ffmpeg-$FFMPEG_VERSION.tar.xz"
UPSTREAM_SOURCE_DIR="$WORK_DIR/ffmpeg-$FFMPEG_VERSION"
TRACED_SOURCE_DIR="$WORK_DIR/combined-src/ffmpeg-$FFMPEG_VERSION"

fetch_ffmpeg_source() {
    # Verify cached downloads too: every build starts from the pinned release.
    mkdir -p "$WORK_DIR/download"
    if [[ ! -f "$FFMPEG_ARCHIVE" ]]; then
        curl --fail --location --silent --show-error \
            "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" \
            -o "$FFMPEG_ARCHIVE"
    fi
    printf '%s  %s\n' "$FFMPEG_SHA256" "$FFMPEG_ARCHIVE" | sha256sum -c -

    if [[ ! -d "$UPSTREAM_SOURCE_DIR" ]]; then
        tar -xf "$FFMPEG_ARCHIVE" -C "$WORK_DIR"
    fi

    # The upstream copy must stay unmodified; instrumentation uses its own tree.
    local file
    for file in buffer.c buffer.h buffer_internal.h; do
        tar -xOf "$FFMPEG_ARCHIVE" "ffmpeg-$FFMPEG_VERSION/libavutil/$file" |
            cmp - "$UPSTREAM_SOURCE_DIR/libavutil/$file"
    done
}
