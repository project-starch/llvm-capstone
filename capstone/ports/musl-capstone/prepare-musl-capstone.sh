#!/usr/bin/env bash
# Stage a musl source tree with our arch/capstone64 port in place.
#
# arch/capstone64 is a copy of upstream arch/riscv64 with the files from
# arch-capstone64/ overlaid on top, and musl-patches/*.patch are applied to src/ and
# include/. The delta to upstream is therefore `diff -r arch/riscv64
# arch/capstone64` plus musl-patches/, each patch one decision:
# 0001 stdio buffers of 8 KiB (glibc's size) instead of 1 KiB, 0002 a 32 KiB
# getdents buffer instead of 2 KiB, 0003 no fcntl(F_SETFD) round after an open
# that already asked for O_CLOEXEC. Every delegated syscall is a round trip
# through the launcher, so the number of rounds is the port's cost model.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

MUSL_SRC_DIR=$(bash "$SCRIPT_DIR/fetch-musl.sh" | tail -1)
ARCH_DIR="$MUSL_SRC_DIR/arch/capstone64"

# Patches apply once: a patch that is already in (reverse applies cleanly) is
# skipped, one that fits neither way is an error, not a silent partial tree.
for patch in "$SCRIPT_DIR"/musl-patches/*.patch; do
  if patch -d "$MUSL_SRC_DIR" -p1 -R --dry-run -s -f < "$patch" > /dev/null 2>&1; then
    continue
  fi
  patch -d "$MUSL_SRC_DIR" -p1 -N -s -f < "$patch" || {
    echo "prepare failed: $patch does not apply to $MUSL_SRC_DIR" >&2
    exit 1
  }
done

rm -rf "$ARCH_DIR"
cp -r "$MUSL_SRC_DIR/arch/riscv64" "$ARCH_DIR"
cp -a "$SCRIPT_DIR/arch-capstone64/." "$ARCH_DIR/"

# Generated headers, exactly as musl's Makefile builds them (Makefile:98-106).
# Done here rather than by `make` because configure cannot probe a target it
# cannot link for, and the survey only needs the headers, not a build system.
mkdir -p "$MUSL_SRC_DIR/obj/include/bits" "$MUSL_SRC_DIR/obj/src/internal"
sed -f "$MUSL_SRC_DIR/tools/mkalltypes.sed" \
    "$ARCH_DIR/bits/alltypes.h.in" \
    "$MUSL_SRC_DIR/include/alltypes.h.in" \
    > "$MUSL_SRC_DIR/obj/include/bits/alltypes.h"
cp "$ARCH_DIR/bits/syscall.h.in" "$MUSL_SRC_DIR/obj/include/bits/syscall.h"
sed -n -e 's/__NR_/SYS_/p' < "$ARCH_DIR/bits/syscall.h.in" \
    >> "$MUSL_SRC_DIR/obj/include/bits/syscall.h"
printf '#define VERSION "%s"\n' "${MUSL_VERSION:-1.2.5}" \
    > "$MUSL_SRC_DIR/obj/src/internal/version.h"

for required in obj/include/bits/alltypes.h obj/include/bits/syscall.h \
                arch/capstone64/syscall_arch.h \
                arch/capstone64/pthread_arch.h \
                arch/capstone64/bits/setjmp.h; do
  [[ -s "$MUSL_SRC_DIR/$required" ]] || {
    echo "prepare failed, missing or empty: $MUSL_SRC_DIR/$required" >&2
    exit 1
  }
done

printf '%s\n' "$MUSL_SRC_DIR"
