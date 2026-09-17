#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
MODE=${1:?usage: build-replay.sh native|capstone}
WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
SRC="$WORK/combined-port-src/ffmpeg-9.0.1"
OUT="$WORK/combined-port-$MODE"
if [[ ! -d "$SRC" ]]; then
    mkdir -p "$(dirname "$SRC")"
    cp -a "$WORK/combined-src/ffmpeg-9.0.1" "$SRC"
fi
for name in buffer.c buffer_internal.h refstruct.c; do
    cp "$WORK/combined-src/ffmpeg-9.0.1/libavutil/$name" "$SRC/libavutil/$name"
done
python3 "$HERE/port.py" "$SRC"
if [[ ! -f "$WORK/$MODE/obj/buffer.o" ]]; then
    bash "$HERE/../build.sh" "$MODE"
fi
mkdir -p "$OUT/obj"
FLAGS=(-std=c11 -O0 -g -ffunction-sections -fdata-sections
       -I"$HERE" -I"$WORK/$MODE/include" -I"$SRC" -I"$SRC/compat/atomics/dummy")
if [[ "$MODE" == native ]]; then
    CC=${CC:-cc}
else
    CC=$CAPSTONE_CLANG
    MUSL=${FFPOOL_MUSL:-$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5}
    FLAGS+=(-target capstone64-unknown-elf
        -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
        -ffreestanding -fno-builtin -nostdinc -DFFPOOL_DOMAIN
        -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
        -isystem "$MUSL/obj/include" -isystem "$MUSL/include"
        -isystem "$("$CC" -print-resource-dir)/include")
fi
for name in buffer refstruct; do
    "$CC" "${FLAGS[@]}" -c "$SRC/libavutil/$name.c" -o "$OUT/obj/$name.o"
done
for name in replay observe memory; do
    "$CC" "${FLAGS[@]}" -c "$HERE/$name.c" -o "$OUT/obj/$name.o"
done
OBJS=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/replay.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${OBJS[@]}" -o "$OUT/replay"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/replay.dom" "${OBJS[@]}" "$WORK/$MODE/obj/string.o" \
        "$WORK/$MODE/obj/start.o" "$WORK/$MODE/obj/gct-section-end.o"
    LIB="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
    "$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        -O2 -Wall -I"$LIB" -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
        "$HERE/host.c" "$LIB/libcapstone.c" -o "$OUT/host.user"
fi
sha256sum "$OUT/obj/"*.o > "$OUT/objects.sha256"
"$CC" "${FLAGS[@]}" -DFF2_SECURITY -c "$HERE/replay.c" -o "$OUT/obj/security-driver.o"
"$CC" "${FLAGS[@]}" -c "$HERE/security.c" -o "$OUT/obj/security.o"
SECURITY=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/security-driver.o"
          "$OUT/obj/security.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${SECURITY[@]}" -o "$OUT/security"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/security.dom" "${SECURITY[@]}" "$WORK/$MODE/obj/string.o" \
        "$WORK/$MODE/obj/start.o" "$WORK/$MODE/obj/gct-section-end.o"
fi
