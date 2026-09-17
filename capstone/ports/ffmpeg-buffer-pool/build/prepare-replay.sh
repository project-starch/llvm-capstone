#!/usr/bin/env bash
# Sourced by build-replay.sh: prepare headers, compiler flags and support paths.

BUILD_TARGET=${1:?usage: build-replay.sh native|capstone}
case "$BUILD_TARGET" in
    native|capstone)
        ;;
    *)
        echo "expected build target: native or capstone" >&2
        exit 2
        ;;
esac

source "$(dirname -- "${BASH_SOURCE[0]}")/ffmpeg-source.sh"
fetch_ffmpeg_source

SUPPORT_BUILD_DIR="$WORK_DIR/$BUILD_TARGET"
mkdir -p "$SUPPORT_BUILD_DIR/include/libavutil" "$SUPPORT_BUILD_DIR/obj"

# Replay runs on one thread. FFmpeg's thread abstraction therefore needs no
# thread library; reference counts still use C11 atomics in both targets.
cat > "$SUPPORT_BUILD_DIR/include/config.h" <<'EOF'
#define HAVE_PTHREADS 0
#define HAVE_W32THREADS 0
#define HAVE_OS2THREADS 0
#define HAVE_PRCTL 0
#define HAVE_PTHREAD_SETNAME_NP 0
#define HAVE_PTHREAD_SET_NAME_NP 0
#define HAVE_PTHREAD_NP_H 0
EOF

cat > "$SUPPORT_BUILD_DIR/include/libavutil/avconfig.h" <<'EOF'
#define AV_HAVE_BIGENDIAN 0
#define AV_HAVE_FAST_UNALIGNED 0
EOF

COMPILE_FLAGS=(
    -std=c11 -O0 -g
    -ffunction-sections -fdata-sections
    -I"$SUPPORT_BUILD_DIR/include"
    -I"$UPSTREAM_SOURCE_DIR"
)

if [[ "$BUILD_TARGET" == native ]]; then
    CC=${CC:-cc}
else
    CC=$CAPSTONE_CLANG

    # The freestanding domain uses MUSL headers, but does not link MUSL libc.
    MUSL_SOURCE_DIR=${FFPOOL_MUSL:-$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5}
    if [[ ! -f "$MUSL_SOURCE_DIR/obj/include/bits/alltypes.h" ]]; then
        echo "set FFPOOL_MUSL to the prepared musl headers" >&2
        exit 2
    fi
    CLANG_INCLUDE_DIR="$("$CC" -print-resource-dir)/include"

    COMPILE_FLAGS+=(
        -target capstone64-unknown-elf
        # Integer multiply/divide and integer atomic instructions.
        -Xclang -target-feature -Xclang +m
        -Xclang -target-feature -Xclang +a
        -ffreestanding -fno-builtin -nostdinc -DFFPOOL_DOMAIN
        -isystem "$MUSL_SOURCE_DIR/arch/capstone64"
        -isystem "$MUSL_SOURCE_DIR/arch/generic"
        -isystem "$MUSL_SOURCE_DIR/obj/include"
        -isystem "$MUSL_SOURCE_DIR/include"
        -isystem "$CLANG_INCLUDE_DIR"
    )
fi

# Capstone execution requires the AS200 atomic compiler and QEMU fixes.
echo "ffpool target=$BUILD_TARGET atomics=c11 (single-threaded replay)"
