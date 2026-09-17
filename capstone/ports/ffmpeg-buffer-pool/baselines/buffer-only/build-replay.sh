#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../runtime/prepare.sh" "${1:?usage: build-replay.sh native|capstone}"
"$CC" "${FLAGS[@]}" -c "$SRC/libavutil/buffer.c" -o "$OUT/obj/buffer.o"
"$CC" "${FLAGS[@]}" -c "$HERE/replay.c" -o "$OUT/obj/replay.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/runtime/metadata-memory.c" -o "$OUT/obj/replay-memory.o"
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
sha256sum "$HERE/replay.c" "$FFPOOL_ROOT/runtime/metadata-memory.c" "$HERE/replay-format.h" \
    > "$OUT/replay-source.sha256"
