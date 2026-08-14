#!/usr/bin/env bash
# Build the musl-hello domain and its host servicer.
#
# Link order matters: our objects come BEFORE libc-capstone.a so the archive is
# only consulted for symbols still undefined. That is also what keeps
# pthread_cancel.o out and lets musl's weak __syscall_cp_c alias -- which calls
# __syscall directly -- win over the cancellation machinery.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-file}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/file_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/file_probe.user}

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
  # See survey-musl-capstone.py: without this the backend miscompiles
  # `return callee(...)` -- call with link, then no epilogue and no return.
  -fno-optimize-sibling-calls
  -D_XOPEN_SOURCE=700
  # APPLICATION include path. src/include and src/internal are musl's OWN build
  # headers and must NOT be here: they define `weak` and `hidden` as MACROS,
  # which collide with ordinary identifiers. Harmless for this program, fatal for
  # Lua (a field named `weak` broke 18 of 22 files), so it is wrong here too.
  -I"$MUSL_SRC_DIR/arch/capstone64"
  -I"$MUSL_SRC_DIR/arch/generic"
  -I"$MUSL_SRC_DIR/obj/include"
  -I"$MUSL_SRC_DIR/include"
  -ffunction-sections -fdata-sections -fno-jump-tables
  -O1 -w
  # See survey-musl-capstone.py: int -> pointer is an error, not a warning, and
  # this ABI needs it.
  -Wno-int-conversion -Wno-error=int-conversion
)

mkdir -p "$OBJ_DIR"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$RUNTIME_DIR/start-musl.S" -o "$OBJ_DIR/start-musl.o"

"$CLANG" "${MUSL_CFLAGS[@]}" -c "$RUNTIME_DIR/hostcall.c" -o "$OBJ_DIR/hostcall.o"
# There is no gap file any more: open() and fsync() come from musl itself, once
# the C-21 frontend fix let src/fcntl/open.c and src/unistd/fsync.c compile.
# ONE arm. A matched pair differing only in image size (5536 vs 6176, i.e.
# dom_data 1120 vs 480) was run and BOTH arms faulted identically, so dom_data
# size is refuted as the variable and the second arm buys nothing.
"$CLANG" "${MUSL_CFLAGS[@]}" -c "$SCRIPT_DIR/file_probe.c" -o "$OBJ_DIR/file_probe.o"

# The word-at-a-time string routines are among the 9 src/string files that do not
# compile pure-cap ((uintptr_t)s % ALIGN), so memcpy/memmove/strlen are absent
# from the archive. Reuse the byte-at-a-time replacements SQLite already links
# rather than writing new ones; they provide memcmp, memcpy, memmove, memset,
# strcmp, strcpy, strlen. Passed BEFORE the archive so these definitions win and
# the archive's memset/memcmp/strcmp/strcpy members are simply never pulled.
BEEBS_STRING_SRC="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
"$CLANG" "${MUSL_CFLAGS[@]}" -c "$BEEBS_STRING_SRC" -o "$OBJ_DIR/beebs_string.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start-musl.o" "$OBJ_DIR/hostcall.o" "$OBJ_DIR/file_probe.o" \
  "$OBJ_DIR/beebs_string.o" "$ARCHIVE"
image=$("$CAPSTONE_LLVM_READOBJ" --program-headers "$OUT_DOM" \
         | awk '/MemSize/ {print $2; exit}')
tot=4096; while (( tot < image + 1536 )); do tot=$(( tot * 2 )); done
printf 'built %s (image %s, dom_data %s)\n' "$OUT_DOM" "$image" "$(( tot - image - 1536 ))"

"$GUEST_CC" -O2 -o "$OUT_HOST" "$RUNTIME_DIR/hostcall_host.c" "$LIBCAPSTONE_C"

printf 'built %s\n' "$OUT_HOST"
