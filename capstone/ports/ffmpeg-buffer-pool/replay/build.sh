#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../runtime/prepare.sh" "${1:?usage: build.sh native|capstone}"
SUPPORT="$OUT"
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
mkdir -p "$OUT/obj"
FLAGS=(-I"$FFPOOL_ROOT/trace" -I"$SRC" "${FLAGS[@]}")
for name in buffer refstruct; do
    "$CC" "${FLAGS[@]}" -c "$SRC/libavutil/$name.c" -o "$OUT/obj/$name.o"
done
"$CC" "${FLAGS[@]}" -c "$HERE/replay.c" -o "$OUT/obj/replay.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/trace/observe.c" -o "$OUT/obj/observe.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/runtime/pool-memory.c" -o "$OUT/obj/memory.o"
OBJS=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/replay.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${OBJS[@]}" -o "$OUT/replay"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/replay.dom" "${OBJS[@]}" "$SUPPORT/obj/string.o" \
        "$SUPPORT/obj/start.o" "$SUPPORT/obj/gct-section-end.o"
    LIB="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
    "$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        -O2 -Wall -I"$FFPOOL_ROOT/trace" -I"$LIB" -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
        "$HERE/host.c" "$LIB/libcapstone.c" -o "$OUT/host.user"
fi
sha256sum "$OUT/obj/"*.o > "$OUT/objects.sha256"
"$CC" "${FLAGS[@]}" -DFF2_SECURITY -c "$HERE/replay.c" -o "$OUT/obj/security-driver.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/security-tests/security.c" -o "$OUT/obj/security.o"
SECURITY=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/security-driver.o"
          "$OUT/obj/security.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${SECURITY[@]}" -o "$OUT/security"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/security.dom" "${SECURITY[@]}" "$SUPPORT/obj/string.o" \
        "$SUPPORT/obj/start.o" "$SUPPORT/obj/gct-section-end.o"
fi
