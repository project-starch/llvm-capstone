#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../tests/capstone-test-env.sh"
MODE=${1:?usage: build-replay.sh native|capstone}
WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
SRC="$WORK/ffmpeg-9.0.1"
OUT="$WORK/$MODE"
# Recheck pristine upstream source and the small semantic controls first.
bash "$HERE/build.sh" "$MODE"
FLAGS=(-std=c11 -O0 -g -ffunction-sections -fdata-sections
       -I"$OUT/include" -I"$SRC" -I"$SRC/compat/atomics/dummy")
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
for name in replay replay-memory; do
    "$CC" "${FLAGS[@]}" -c "$HERE/$name.c" -o "$OUT/obj/$name.o"
done
OBJS=("$OUT/obj/buffer.o" "$OUT/obj/replay.o" "$OUT/obj/replay-memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${OBJS[@]}" -o "$OUT/replay"
else
    "$CAPSTONE_LD_LLD" --gc-sections \
        -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/replay.dom" "${OBJS[@]}" "$OUT/obj/string.o" \
        "$OUT/obj/start.o" "$OUT/obj/gct-section-end.o"
    LIB="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
    "$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        -O2 -Wall -I"$LIB" -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
        "$HERE/replay-host.c" "$LIB/libcapstone.c" -o "$OUT/replay-host.user"
fi
sha256sum "$HERE/replay.c" "$HERE/replay-memory.c" "$HERE/replay-format.h" \
    > "$OUT/replay-source.sha256"
