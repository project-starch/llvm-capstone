#!/usr/bin/env bash
# Build reference Lua 5.4 against musl as a pure-capability domain.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-lua}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/lua_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/lua_probe.user}

RUNTIME_DIR="$SCRIPT_DIR/../runtime"
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
BUILTINS_DIR="$REPO_ROOT/compiler-rt/lib/builtins"
BEEBS_STRING_SRC="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
SETJMP_SRC="$REPO_ROOT/xlang/lua-cdp/capstone-lua/capstone_setjmp.S"
LUA_SRC=${LUA_SRC:-$REPO_ROOT/xlang/lua-cdp/_toolchain/.work/lua54}

ARCHIVE=${ARCHIVE:-$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a}
MUSL_SRC_DIR=${MUSL_SRC_DIR:-$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5}
for needed in "$ARCHIVE" "$MUSL_SRC_DIR/arch/capstone64/syscall_arch.h" "$LUA_SRC/lapi.c" "$SETJMP_SRC"; do
  [[ -e "$needed" ]] || { echo "missing prerequisite: $needed" >&2; exit 2; }
done

# APPLICATION include path. Deliberately NOT src/include or src/internal: those
# are musl's own build headers and define `weak` and `hidden` as MACROS, which
# collide with ordinary identifiers -- Lua has a field `GCObject *weak`, and the
# collision presents as "expected member name or ';'" in 18 of 22 files.
MUSL_INC=(-I"$MUSL_SRC_DIR/arch/capstone64" -I"$MUSL_SRC_DIR/arch/generic"
          -I"$MUSL_SRC_DIR/obj/include" -I"$MUSL_SRC_DIR/include")

# -O0 because that is the level the working Lua recipe uses. At -O1 five core
# TUs (lapi lfunc lgc ltable ltm) hit the i128-on-a-capability family.
# -fno-jump-tables and -DLUA_USE_JUMPTABLE=0: two SEPARATE absolute-addressed
# jump-table mechanisms, one from the compiler's switch lowering and one from
# Lua's own computed-goto VM dispatch. Both fault on Capstone; see
# history/06-08-2026_19-45-00_lua-runs-on-capstone-cjalr-jumptables.md.
# -fno-optimize-sibling-calls: ISSUES.md C-19.
# DBGP/DBGC: this Lua tree carries the project's debug hooks, normally supplied
# by capstone_lua_libc.h, which musl replaces.
LUA_FLAGS=(-target capstone64-unknown-elf
           -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
           -std=c99 -nostdinc -ffreestanding -fno-builtin
           -fno-jump-tables -fno-optimize-sibling-calls -DLUA_USE_JUMPTABLE=0
           -D_XOPEN_SOURCE=700 -Wno-int-conversion -Wno-error=int-conversion
           '-DDBGP(x)=((void)0)' '-DDBGC(p)=((void)0)'
           "${MUSL_INC[@]}" -I"$LUA_SRC"
           -ffunction-sections -fdata-sections -O0 -w)

LUA_TUS=(lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes
         lparser lstate lstring ltable ltm lundump lvm lzio lauxlib lbaselib)

rm -rf "$OBJ_DIR"; mkdir -p "$OBJ_DIR"
OBJS=()

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$RUNTIME_DIR/start-musl.S" -o "$OBJ_DIR/start.o"
OBJS+=("$OBJ_DIR/start.o")

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$SETJMP_SRC" -o "$OBJ_DIR/setjmp.o"
OBJS+=("$OBJ_DIR/setjmp.o")

for src in "$RUNTIME_DIR/hostcall.c" "$SCRIPT_DIR/lua_probe.c" \
           "$SCRIPT_DIR/lua_probe_stubs.c" "$SCRIPT_DIR/lua_probe_alloc.c" \
           "$BEEBS_STRING_SRC"; do
  obj="$OBJ_DIR/$(basename "${src%.c}").o"
  "$CLANG" "${LUA_FLAGS[@]}" -c "$src" -o "$obj"
  OBJS+=("$obj")
done

for tu in "${LUA_TUS[@]}"; do
  "$CLANG" "${LUA_FLAGS[@]}" -c "$LUA_SRC/$tu.c" -o "$OBJ_DIR/$tu.o"
  OBJS+=("$OBJ_DIR/$tu.o")
done

# Soft-float builtins. Lua is float-heavy (lua_Number is double) and musl's
# printf pulls the 128-bit ones. Names differ from the symbols: comparedf2.c
# provides __eqdf2/__gedf2/__gtdf2/__ltdf2/__nedf2, and so on.
for builtin in adddf3 comparedf2 divdf3 extenddftf2 extendsftf2 fixdfdi fixdfsi \
               fixunsdfdi fixunsdfsi floatdidf floatsidf floatsisf floatsitf \
               floatunsidf floatunsitf fp_mode muldf3 mulsf3 multf3 subdf3 \
               trunctfdf2 trunctfsf2; do
  src="$BUILTINS_DIR/$builtin.c"
  [[ -f "$src" ]] || continue        # not every symbol has a file of that name
  obj="$OBJ_DIR/rt_$builtin.o"
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -fno-builtin -fno-optimize-sibling-calls -O0 -w \
    -I"$BUILTINS_DIR" -c "$src" -o "$obj" 2>/dev/null || continue
  OBJS+=("$obj")
done

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${OBJS[@]}" "$ARCHIVE"

"$GUEST_CC" -O2 -o "$OUT_HOST" "$RUNTIME_DIR/hostcall_host.c" "$LIBCAPSTONE_C"

image=$("$CAPSTONE_LLVM_READOBJ" --program-headers "$OUT_DOM" | awk '/MemSize/ {print $2; exit}')
tot=4096; while (( tot < image + 1536 )); do tot=$(( tot * 2 )); done
printf 'built %s (image %s, dom_data %s)\nbuilt %s\n' \
  "$OUT_DOM" "$image" "$(( tot - image - 1536 ))" "$OUT_HOST"
