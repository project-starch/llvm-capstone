#!/usr/bin/env bash
# build.sh SDK SQLITE_SOURCES FFMPEG_ARCHIVE OUTPUT BASELINE_REVISION
set -euo pipefail
SDK=$(realpath "${1:?SDK}")
SQLITE=$(realpath "${2:?prepared SQLite sources}")
ARCHIVE=$(realpath "${3:?FFmpeg 9.0.1 archive}")
OUT=$(realpath -m "${4:?fresh output directory}")
BASE=${5:?revision before the lifetime fixes}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
cd "$REPO"
source capstone/tests/capstone-test-env.sh
[[ ! -e "$OUT" ]] || { echo 'Use a fresh output directory' >&2; exit 2; }
mkdir -p "$OUT/bin"
CC=$SDK/capstone-cc
INC=$REPO/capstone/runtime/include
SQLPORT=$REPO/capstone/ports/sqlite/sublet
FFPORT=$REPO/capstone/ports/ffmpeg
for arm in request block page old; do
    dir=$OUT/sqlite-$arm
    mkdir -p "$dir"
    cp "$SQLITE/sqlite3.c" "$dir/sqlite3-capstone.c"
    patchname=sublet-3220000-memsys5.patch
    [[ $arm != block ]] || patchname=sublet-3220000-memsys5-block.patch
    if [[ $arm == old ]]; then
        git -c submodule.recurse=false show "$BASE:capstone/ports/sqlite/sublet/$patchname" > "$OUT/old-sqlite.patch"
        patchfile=$OUT/old-sqlite.patch
    else
        patchfile=$SQLPORT/$patchname
    fi
    patch -s -F0 -p1 -d "$dir" < "$patchfile"
    flags=()
    if [[ $arm == page ]]; then
        patch -s -F0 -p1 -d "$dir" < "$SQLPORT/sublet-3220000-pagecache.patch"
        flags+=(-DTEST_PAGE)
    fi
    "$CC" -O1 -DSQLITE_THREADSAFE=0 -DSQLITE_ENABLE_MEMSYS5 -DSQLITE_DEBUG \
        -DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 "${flags[@]}" \
        -I"$HERE" -I"$INC" -I"$dir" "$SQLPORT/tests/failure.c" "$HERE/inject.c" \
        -o "$OUT/bin/sqlite-$arm.dom"
done
# The generated block source must exactly match the applied block patch.
python3 "$SQLPORT/make-sublet-block-port.py" "$OUT/sqlite-request/sqlite3-capstone.c" "$OUT/generated.c"
cmp "$OUT/generated.c" "$OUT/sqlite-block/sqlite3-capstone.c"
sha=cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635
printf '%s  %s\n' "$sha" "$ARCHIVE" | sha256sum -c -
mkdir -p "$OUT/ff-source" "$OUT/ff-build"
tar xf "$ARCHIVE" -C "$OUT/ff-source" --strip-components=1
for p in "$FFPORT/app/patches/"*.patch "$FFPORT/buffer-pool/patches/"*0001*.patch; do
    patch -s -F0 -p1 -d "$OUT/ff-source" < "$p"
done
cp "$OUT/ff-source/libavutil/buffer.c" "$OUT/base-buffer.c"
cp "$OUT/ff-source/libavutil/refstruct.c" "$OUT/base-refstruct.c"
cp "$OUT/ff-source/libavutil/buffer_internal.h" "$OUT/base-buffer_internal.h"
cp "$FFPORT/buffer-pool/src/shared/trace.h" "$FFPORT/app/src/pool/"*.c "$OUT/ff-source/libavutil/"
patchname=ffmpeg-9.0.1-0003-libavutil-pools-as-sublet-lifetimes.patch
patch -s -F0 -p1 -d "$OUT/ff-source" < "$FFPORT/buffer-pool/patches/$patchname"
cd "$OUT/ff-build"
"$OUT/ff-source/configure" --enable-cross-compile --cc="$CC" --ld="$CC" \
    --ar="$CAPSTONE_LLVM_BIN/llvm-ar" --ranlib="$CAPSTONE_LLVM_BIN/llvm-ranlib" \
    --arch=riscv64 --target-os=none --extra-cflags="-I$HERE -I$INC" \
    --disable-everything --disable-autodetect --disable-doc --disable-network \
    --disable-asm --disable-inline-asm --disable-pthreads --disable-programs \
    --disable-debug --disable-iconv --disable-swresample --disable-swscale \
    --disable-avfilter --disable-avdevice --disable-avformat --disable-avcodec \
    --enable-static --disable-shared --optflags=-O1 > configure.log 2>&1
sed -i 's/#define HAVE_POSIX_MEMALIGN 1/#define HAVE_POSIX_MEMALIGN 0/; s/#define HAVE_MEMALIGN 1/#define HAVE_MEMALIGN 0/' config.h
for arm in fixed old; do
    if [[ $arm == old ]]; then
        for f in buffer.c refstruct.c buffer_internal.h; do
            cp "$OUT/base-$f" "$OUT/ff-source/libavutil/$f"
        done
        git -c submodule.recurse=false -C "$REPO" show "$BASE:capstone/ports/ffmpeg/buffer-pool/patches/$patchname" > "$OUT/old-ffmpeg.patch"
        patch -s -F0 -p1 -d "$OUT/ff-source" < "$OUT/old-ffmpeg.patch"
    fi
    make -j"${JOBS:-8}" libavutil/libavutil.a > "build-$arm.log" 2>&1
    name=ffmpeg
    [[ $arm != old ]] || name=ffmpeg-old
    "$CC" -O1 -I"$OUT/ff-source" -I. "$FFPORT/buffer-pool/tests/lifetime/failure.c" \
        "$HERE/inject.c" libavutil/libavutil.a -o "$OUT/bin/$name.dom"
done
printf 'Built lifetime tests in %s/bin\n' "$OUT"
