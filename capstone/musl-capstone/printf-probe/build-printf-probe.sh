#!/usr/bin/env bash
# Build the printf-probe domain and its host servicer.
#
# Same shape as build-file-probe.sh; the difference is what gets pulled out of
# the archive. Nothing here references vfprintf or malloc directly: snprintf and
# printf pull vfprintf, and the probe's malloc/free/realloc/calloc now resolve
# inside libc-capstone.a instead of coming from a per-program allocator. That is
# the point of the change being tested.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-printf}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/printf_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/printf_probe.user}

RUNTIME_DIR="$SCRIPT_DIR/../runtime"
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

ARCHIVE=${ARCHIVE:-$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a}
if [[ ! -f "$ARCHIVE" ]]; then
  echo "missing $ARCHIVE -- run build-musl-capstone.sh first" >&2
  exit 2
fi
MUSL_SRC_DIR=${MUSL_SRC_DIR:-$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5}
if [[ ! -f "$MUSL_SRC_DIR/arch/capstone64/syscall_arch.h" ]]; then
  echo "missing prepared musl tree at $MUSL_SRC_DIR" >&2
  exit 2
fi

MUSL_CFLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature -Xclang +m
  -Xclang -target-feature -Xclang +a
  -std=c99 -nostdinc -ffreestanding -fno-builtin
  -fno-optimize-sibling-calls
  -D_XOPEN_SOURCE=700
  # APPLICATION include path: src/include and src/internal are musl's own build
  # headers and must not be here (see build-file-probe.sh).
  -I"$MUSL_SRC_DIR/arch/capstone64"
  -I"$MUSL_SRC_DIR/arch/generic"
  -I"$MUSL_SRC_DIR/obj/include"
  -I"$MUSL_SRC_DIR/include"
  -ffunction-sections -fdata-sections -fno-jump-tables
  -O1 -w
  -Wno-int-conversion -Wno-error=int-conversion
)
# EXTRA_CFLAGS exists for -DPRINTF_PROBE_TRACE, which turns the probe into a
# bisector: a capability fault kills QEMU outright, so the last trace line the
# host printed is the only localization available.
# shellcheck disable=SC2206
[[ -n "${EXTRA_CFLAGS:-}" ]] && MUSL_CFLAGS+=( ${EXTRA_CFLAGS} )

mkdir -p "$OBJ_DIR"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$RUNTIME_DIR/start-musl.S" -o "$OBJ_DIR/start-musl.o"
"$CLANG" "${MUSL_CFLAGS[@]}" -c "$RUNTIME_DIR/hostcall.c" -o "$OBJ_DIR/hostcall.o"
"$CLANG" "${MUSL_CFLAGS[@]}" -c "$SCRIPT_DIR/printf_probe.c" -o "$OBJ_DIR/printf_probe.o"

# NO beebs_freestanding_string.c, unlike every earlier domain here. memcpy,
# memmove, strlen and memchr now come from libc-ext inside the archive, and
# memcmp/memset/strcmp/strcpy from musl's own members. If this link ever needs a
# string routine passed in from outside again, the libc has a hole.
"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start-musl.o" "$OBJ_DIR/hostcall.o" "$OBJ_DIR/printf_probe.o" \
  "$ARCHIVE"
image=$("$CAPSTONE_LLVM_READOBJ" --program-headers "$OUT_DOM" \
         | awk '/MemSize/ {print $2; exit}')
tot=4096; while (( tot < image + 1536 )); do tot=$(( tot * 2 )); done
printf 'built %s (image %s, dom_data %s)\n' "$OUT_DOM" "$image" "$(( tot - image - 1536 ))"

"$GUEST_CC" -O2 -o "$OUT_HOST" "$RUNTIME_DIR/hostcall_host.c" "$LIBCAPSTONE_C"

printf 'built %s\n' "$OUT_HOST"
