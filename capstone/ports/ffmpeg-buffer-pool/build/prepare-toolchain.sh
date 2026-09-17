#!/usr/bin/env bash
# Source from a build script with native|capstone as its argument.
# Exports WORK, SRC, OUT, MODE, CC and FLAGS in the calling shell; preserves HERE.
# Generated paths stay stable so archived commands and reports remain usable.

FFPOOL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$FFPOOL_ROOT/../../tests/capstone-test-env.sh"

MODE=${1:-native}
case "$MODE" in native|capstone) ;; *) echo "usage: $0 native|capstone" >&2; exit 2;; esac

WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
VERSION=9.0.1
SHA=cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635
ARCHIVE="$WORK/download/ffmpeg-$VERSION.tar.xz"
SRC="$WORK/ffmpeg-$VERSION"

# Fetch the pinned archive and verify cached downloads before extraction.
mkdir -p "$WORK/download"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --silent --show-error \
        "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz" -o "$ARCHIVE"
fi
printf '%s  %s\n' "$SHA" "$ARCHIVE" | sha256sum -c -
if [[ ! -d "$SRC" ]]; then
    tar -xf "$ARCHIVE" -C "$WORK"
fi

# Refuse a modified pool source rather than silently calling it upstream.
for file in buffer.c buffer.h buffer_internal.h; do
    tar -xOf "$ARCHIVE" "ffmpeg-$VERSION/libavutil/$file" |
        cmp - "$SRC/libavutil/$file"
done

OUT="$WORK/$MODE"
mkdir -p "$OUT/include/libavutil" "$OUT/obj"
# Only these configuration choices are used through buffer.c's public headers
# and its thread abstraction. Both isolated arms select upstream's serial path.
cat > "$OUT/include/config.h" <<'EOF'
#define HAVE_PTHREADS 0
#define HAVE_W32THREADS 0
#define HAVE_OS2THREADS 0
#define HAVE_PRCTL 0
#define HAVE_PTHREAD_SETNAME_NP 0
#define HAVE_PTHREAD_SET_NAME_NP 0
#define HAVE_PTHREAD_NP_H 0
EOF
cat > "$OUT/include/libavutil/avconfig.h" <<'EOF'
#define AV_HAVE_BIGENDIAN 0
#define AV_HAVE_FAST_UNALIGNED 0
EOF

FLAGS=(-std=c11 -O0 -g -ffunction-sections -fdata-sections
       -I"$OUT/include" -I"$SRC")

# The compiler currently fails selecting C11 atomic_fetch_add through an
# AS200 pointer. Use FFmpeg's OWN fallback for serial configurations in BOTH
# isolated arms. Its atomic_uint is intptr_t, so this is also a layout change.
# FFPOOL_ATOMICS=c11 reproduces the standard-atomics build attempt.
ATOMICS=${FFPOOL_ATOMICS:-serial}
case "$ATOMICS" in
    serial) FLAGS+=(-I"$SRC/compat/atomics/dummy");;
    c11) ;;
    *) echo "FFPOOL_ATOMICS must be serial or c11" >&2; exit 2;;
esac
echo "ffpool mode=$MODE atomics=$ATOMICS (isolated execution is serial)"

# Select the host compiler or prepare the freestanding domain toolchain.
if [[ "$MODE" == native ]]; then
    CC=${CC:-cc}
else
    CC=$CAPSTONE_CLANG

    # MUSL supplies headers here; the replay does not link a MUSL libc.
    MUSL=${FFPOOL_MUSL:-$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5}
    [[ -f "$MUSL/obj/include/bits/alltypes.h" ]] || {
        echo "set FFPOOL_MUSL to the prepared musl headers" >&2; exit 2; }
    RESOURCE=$("$CC" -print-resource-dir)
    FLAGS+=(-target capstone64-unknown-elf
            -Xclang -target-feature -Xclang +m
            -Xclang -target-feature -Xclang +a
            -ffreestanding -fno-builtin -nostdinc -DFFPOOL_DOMAIN
            -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
            -isystem "$MUSL/obj/include" -isystem "$MUSL/include"
            -isystem "$RESOURCE/include")

    # This repository's copy preserves capability tags in pointer-bearing data.
    "$CC" "${FLAGS[@]}" -c \
        "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c" \
        -o "$OUT/obj/string.o"

    # Assemble domain startup and the global-capability-table terminator.
    for file in "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S" \
                "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/gct-section-end.S"; do
        obj="$OUT/obj/$(basename "${file%.S}").o"
        "$CC" -target capstone64-unknown-elf -ffreestanding -c "$file" -o "$obj"
    done
fi
