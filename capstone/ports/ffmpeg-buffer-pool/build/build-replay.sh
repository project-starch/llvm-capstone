#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/prepare-replay.sh" "${1:?usage: build-replay.sh native|capstone}"

if [[ "$BUILD_TARGET" == native ]]; then
    ENTRY_SOURCE="$FFPOOL_ROOT/native/replay/main.c"
else
    ENTRY_SOURCE="$FFPOOL_ROOT/capstone/domain/entry.c"

    # Domain string operations preserve capability tags in pointer-bearing data.
    "$CC" "${COMPILE_FLAGS[@]}" -c \
        "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c" \
        -o "$SUPPORT_BUILD_DIR/obj/string.o"

    # Domain startup and the terminator for its global capability table.
    for source_file in "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S" \
                       "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/gct-section-end.S"; do
        object_file="$SUPPORT_BUILD_DIR/obj/$(basename "${source_file%.S}").o"
        "$CC" -target capstone64-unknown-elf -ffreestanding \
            -c "$source_file" -o "$object_file"
    done
fi

# Apply the allocator port to a separate copy of the instrumented FFmpeg tree.
PORT_SOURCE_DIR="$WORK_DIR/combined-port-src/ffmpeg-$FFMPEG_VERSION"
REPLAY_BUILD_DIR="$WORK_DIR/combined-port-$BUILD_TARGET"
if [[ ! -d "$PORT_SOURCE_DIR" ]]; then
    mkdir -p "$(dirname "$PORT_SOURCE_DIR")"
    cp -a "$TRACED_SOURCE_DIR" "$PORT_SOURCE_DIR"
fi

# Restore the input files before patching so repeated builds do not stack edits.
for name in buffer.c buffer_internal.h refstruct.c; do
    cp "$TRACED_SOURCE_DIR/libavutil/$name" "$PORT_SOURCE_DIR/libavutil/$name"
done
python3 "$FFPOOL_ROOT/patches/apply-pool-port.py" "$PORT_SOURCE_DIR"

# Compile the pool code, target entry point and shared allocator.
mkdir -p "$REPLAY_BUILD_DIR/obj"
COMPILE_FLAGS=(-I"$FFPOOL_ROOT/shared" -I"$PORT_SOURCE_DIR" "${COMPILE_FLAGS[@]}")
for name in buffer refstruct; do
    "$CC" "${COMPILE_FLAGS[@]}" -c "$PORT_SOURCE_DIR/libavutil/$name.c" \
        -o "$REPLAY_BUILD_DIR/obj/$name.o"
done
"$CC" "${COMPILE_FLAGS[@]}" -c "$ENTRY_SOURCE" -o "$REPLAY_BUILD_DIR/obj/replay.o"
"$CC" "${COMPILE_FLAGS[@]}" -c "$FFPOOL_ROOT/shared/observe-pool-events.c" \
    -o "$REPLAY_BUILD_DIR/obj/observe.o"
"$CC" "${COMPILE_FLAGS[@]}" -c "$FFPOOL_ROOT/shared/pool-allocator.c" \
    -o "$REPLAY_BUILD_DIR/obj/memory.o"

REPLAY_OBJECTS=(
    "$REPLAY_BUILD_DIR/obj/buffer.o"
    "$REPLAY_BUILD_DIR/obj/refstruct.o"
    "$REPLAY_BUILD_DIR/obj/replay.o"
    "$REPLAY_BUILD_DIR/obj/observe.o"
    "$REPLAY_BUILD_DIR/obj/memory.o"
)

# Link a native executable or a freestanding domain with its support objects.
if [[ "$BUILD_TARGET" == native ]]; then
    "$CC" -Wl,--gc-sections "${REPLAY_OBJECTS[@]}" -o "$REPLAY_BUILD_DIR/replay"
else
    DOMAIN_SUPPORT_OBJECTS=(
        "$SUPPORT_BUILD_DIR/obj/string.o"
        "$SUPPORT_BUILD_DIR/obj/start.o"
        "$SUPPORT_BUILD_DIR/obj/gct-section-end.o"
    )
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$REPLAY_BUILD_DIR/replay.dom" "${REPLAY_OBJECTS[@]}" "${DOMAIN_SUPPORT_OBJECTS[@]}"

    # The loader is an ordinary Linux process that enters the Capstone domain.
    LIBCAPSTONE_DIR="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
    "$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        -O2 -Wall -I"$FFPOOL_ROOT/shared" -I"$LIBCAPSTONE_DIR" \
        -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
        "$FFPOOL_ROOT/capstone/linux-host/domain-loader.c" "$LIBCAPSTONE_DIR/libcapstone.c" \
        -o "$REPLAY_BUILD_DIR/host.user"
fi

sha256sum "$REPLAY_BUILD_DIR/obj/"*.o > "$REPLAY_BUILD_DIR/objects.sha256"

# Reuse the pool and allocator objects with the security-probe entry point.
"$CC" "${COMPILE_FLAGS[@]}" -DFF2_SECURITY -c "$ENTRY_SOURCE" \
    -o "$REPLAY_BUILD_DIR/obj/security-driver.o"
"$CC" "${COMPILE_FLAGS[@]}" -c "$FFPOOL_ROOT/security-tests/shared/pool-lifetime-probes.c" \
    -o "$REPLAY_BUILD_DIR/obj/security.o"

SECURITY_OBJECTS=(
    "$REPLAY_BUILD_DIR/obj/buffer.o"
    "$REPLAY_BUILD_DIR/obj/refstruct.o"
    "$REPLAY_BUILD_DIR/obj/security-driver.o"
    "$REPLAY_BUILD_DIR/obj/security.o"
    "$REPLAY_BUILD_DIR/obj/observe.o"
    "$REPLAY_BUILD_DIR/obj/memory.o"
)

if [[ "$BUILD_TARGET" == native ]]; then
    "$CC" -Wl,--gc-sections "${SECURITY_OBJECTS[@]}" -o "$REPLAY_BUILD_DIR/security"
else
    "$CAPSTONE_LD_LLD" --gc-sections -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$REPLAY_BUILD_DIR/security.dom" "${SECURITY_OBJECTS[@]}" "${DOMAIN_SUPPORT_OBJECTS[@]}"
fi
