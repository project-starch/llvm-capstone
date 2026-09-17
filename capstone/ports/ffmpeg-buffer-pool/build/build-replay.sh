#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/prepare-toolchain.sh" "${1:?usage: build.sh native|capstone}"

# Keep the prepared support objects before selecting the replay output directory.
SUPPORT="$OUT"
if [[ "$MODE" == native ]]; then
    ENTRY="$FFPOOL_ROOT/native/replay/main.c"
else
    ENTRY="$FFPOOL_ROOT/capstone/domain/entry.c"
fi

# Build the allocator port from the instrumented native sources.
SRC="$WORK/combined-port-src/ffmpeg-9.0.1"
OUT="$WORK/combined-port-$MODE"
if [[ ! -d "$SRC" ]]; then
    mkdir -p "$(dirname "$SRC")"
    cp -a "$WORK/combined-src/ffmpeg-9.0.1" "$SRC"
fi

# Restore these files before patching so repeated builds do not stack edits.
for name in buffer.c buffer_internal.h refstruct.c; do
    cp "$WORK/combined-src/ffmpeg-9.0.1/libavutil/$name" "$SRC/libavutil/$name"
done
python3 "$FFPOOL_ROOT/patches/apply-pool-port.py" "$SRC"

# Compile the upstream pool code, target entry point and shared allocator.
mkdir -p "$OUT/obj"
FLAGS=(-I"$FFPOOL_ROOT/shared" -I"$SRC" "${FLAGS[@]}")
for name in buffer refstruct; do
    "$CC" "${FLAGS[@]}" -c "$SRC/libavutil/$name.c" -o "$OUT/obj/$name.o"
done
"$CC" "${FLAGS[@]}" -c "$ENTRY" -o "$OUT/obj/replay.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/shared/observe-pool-events.c" -o "$OUT/obj/observe.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/shared/pool-allocator.c" -o "$OUT/obj/memory.o"

# Link either a native executable or a freestanding Capstone domain.
OBJS=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/replay.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${OBJS[@]}" -o "$OUT/replay"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/replay.dom" "${OBJS[@]}" "$SUPPORT/obj/string.o" \
        "$SUPPORT/obj/start.o" "$SUPPORT/obj/gct-section-end.o"

    # The domain loader runs as a normal RISC-V Linux process in the guest.
    LIB="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
    "$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        -O2 -Wall -I"$FFPOOL_ROOT/shared" -I"$LIB" -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
        "$FFPOOL_ROOT/capstone/linux-host/domain-loader.c" "$LIB/libcapstone.c" -o "$OUT/host.user"
fi

sha256sum "$OUT/obj/"*.o > "$OUT/objects.sha256"

# Reuse the pool and allocator objects with the security-probe entry point.
"$CC" "${FLAGS[@]}" -DFF2_SECURITY -c "$ENTRY" -o "$OUT/obj/security-driver.o"
"$CC" "${FLAGS[@]}" -c "$FFPOOL_ROOT/security-tests/shared/pool-lifetime-probes.c" -o "$OUT/obj/security.o"
SECURITY=("$OUT/obj/buffer.o" "$OUT/obj/refstruct.o" "$OUT/obj/security-driver.o"
          "$OUT/obj/security.o" "$OUT/obj/observe.o" "$OUT/obj/memory.o")
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${SECURITY[@]}" -o "$OUT/security"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/security.dom" "${SECURITY[@]}" "$SUPPORT/obj/string.o" \
        "$SUPPORT/obj/start.o" "$SUPPORT/obj/gct-section-end.o"
fi
