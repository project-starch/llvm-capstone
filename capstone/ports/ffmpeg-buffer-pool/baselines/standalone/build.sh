#!/usr/bin/env bash
# Compare the upstream buffer API with the isolated serial component.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../runtime/prepare.sh" "${1:-native}"
if [[ "$MODE" == native ]]; then
    if [[ ! -f "$OUT/config.h" ]]; then
        (cd "$OUT"; "$SRC/configure" --disable-everything --disable-autodetect \
            --disable-programs --disable-doc --disable-network --disable-x86asm \
            --disable-pthreads --disable-w32threads --disable-os2threads \
            --enable-static --disable-shared > configure.out 2>&1)
    fi
    make -C "$OUT" -j"${JOBS:-4}" libavutil/libavutil.a > "$OUT/build.out" 2>&1
    "$CC" -std=c11 -O0 -g -I"$SRC" "$HERE/probe.c" \
        "$OUT/libavutil/libavutil.a" -lm -o "$OUT/probe-upstream"
fi
OBJS=()
for file in "$SRC/libavutil/buffer.c" "$HERE/probe.c" "$HERE/pilot_memory.c"; do
    obj="$OUT/obj/$(basename "${file%.c}").o"
    "$CC" "${FLAGS[@]}" -c "$file" -o "$obj"
    OBJS+=("$obj")
done
if [[ "$MODE" == native ]]; then
    "$CC" -Wl,--gc-sections "${OBJS[@]}" -o "$OUT/probe-isolated"
    "$OUT/probe-upstream"
    "$OUT/probe-isolated"
    # Validate the exit-status oracle with a deliberately failing build.
    "$CC" "${FLAGS[@]}" -DFFPOOL_FORCE_FAILURE -c "$HERE/probe.c" -o "$OUT/obj/negative.o"
    "$CC" -Wl,--gc-sections "$OUT/obj/buffer.o" "$OUT/obj/negative.o" \
        "$OUT/obj/pilot_memory.o" -o "$OUT/probe-negative"
    negative_rc=0
    "$OUT/probe-negative" > "$OUT/negative.out" || negative_rc=$?
    if [[ "$negative_rc" != 1 ]] || ! grep -qx 'ffpool status=106 result=FAIL' "$OUT/negative.out"; then
        echo "failure control did not report the expected failure" >&2; exit 1
    fi
    echo "corrupted-payload control: expected status 106 and exit 1"
else
    OBJS+=("$OUT/obj/start.o" "$OUT/obj/gct-section-end.o")
    "$CAPSTONE_LD_LLD" --gc-sections \
        -T "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld" \
        -o "$OUT/ffpool.dom" "${OBJS[@]}" "$OUT/obj/string.o"
    echo "built $OUT/ffpool.dom (expected return 42042, not yet an execution result)"
fi
{
    printf 'mode=%s atomics=%s\n' "$MODE" "$ATOMICS"
    "$CC" --version
    sha256sum "$(command -v "$CC")"
    sha256sum "$SRC/libavutil/buffer.c" "$HERE/probe.c" "$HERE/pilot_memory.c"
} > "$OUT/build-identity.txt"
