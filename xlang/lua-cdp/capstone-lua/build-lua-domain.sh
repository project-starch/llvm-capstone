#!/usr/bin/env bash
# Build the minimal real-Lua-5.4.7 Capstone domain + its host controller.
#
#   ./build-lua-domain.sh
#
# Links, in one domain image:
#   start.S                              the cjalr-ABI entry (reused verbatim)
#   lua_domain.o                         domain_main: newstate, open base, run a chunk
#   <20 Lua core>.o + lauxlib + lbaselib the ACTUAL reference interpreter + base lib
#   lua_libc.o                           the freestanding libc gaps (ctype/snprintf/...)
#   capstone_setjmp.o                    capability-aware setjmp/longjmp (LUAI_TRY)
#   revoke_arena_domain.o                the REVOKING allocator + tag-preserving realloc
#   beebs string + softfloat libm        memcpy/strlen/... and floor/pow/exp/...
#   compiler-rt double soft-float        adddf3/muldf3/.../divdf3
#
# -O0 like every domain here. The allocator is the revoking one (SPLIT/MREV/REVOKE);
# revocation is NOT disabled. See lua_domain.c / revoke_arena_domain.c.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LUA_SRC="$CAPSTONE_REPO_ROOT/xlang/lua-cdp/_toolchain/.work/lua54"
LIBC_DIR="$CAPSTONE_REPO_ROOT/xlang/lua-cdp/capstone-lua"
COMMON_DIR="$CAPSTONE_REPO_ROOT/xlang/common"
HOSTCALL_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"   # sqlite_hostcall.h + rof header
BEEBS_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/adapted"
BUILTINS_DIR="$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins"
START_SRC="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S"
LINKER_SCRIPT="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/lua-domain}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/lua_domain.dom}
mkdir -p "$OBJ_DIR"

# Freestanding recipe for every Lua / libc / domain TU (from the port's spec).
# -fno-jump-tables: a `switch` otherwise lowers to an ABSOLUTE-addressed jump table
# loaded with a plain `lw` through a raw (non-capability) address, which faults
# cause-24 ("requires capability") on Capstone — the exact fault hit in lua_gc's
# switch(what) and singlestep's switch(g->gcstate). This is the SAME flag the WORKING
# SQLite cjalr build carries (build-sqlite-capstone.sh COMMON_FLAGS); Lua's build was
# simply missing it. -fno-optimize-sibling-calls matches that recipe too.
# -DLUA_USE_JUMPTABLE=0: separate from -fno-jump-tables. Lua's VM loop (lvm.c) has
# its OWN application-level computed-goto dispatch (ljumptab.h: `goto *disptab[op]`
# over a table of `&&label` addresses), enabled under __GNUC__ (clang defines it).
# Those are ABSOLUTE code addresses with no PCC bounds -> a cause-1 instruction-access
# fault at a raw static pc on Capstone. -fno-jump-tables (a compiler switch-lowering
# flag) does NOT touch it; this macro forces the plain `switch` dispatch instead.
LUA_FLAGS=(-target capstone64-unknown-elf
           -Xclang -target-feature -Xclang +m
           -ffreestanding -nostdlibinc -fno-builtin
           -fno-jump-tables -fno-optimize-sibling-calls -DLUA_USE_JUMPTABLE=0
           -ffunction-sections -fdata-sections -O0
           -include "$LIBC_DIR/capstone_lua_libc.h"
           -I"$LIBC_DIR/include" -I"$LUA_SRC")

# The 20 core TUs + the two libs we actually need (base). luaopen_base is opened
# directly, so linit/the other libs are not linked; --gc-sections drops the rest.
LUA_TUS=(lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes
         lparser lstate lstring ltable ltm lundump lvm lzio lauxlib lbaselib)

OBJS=()

# --- entry ---
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"
OBJS+=("$OBJ_DIR/start.o")

# --- the reference interpreter ---
for tu in "${LUA_TUS[@]}"; do
  "$CLANG" "${LUA_FLAGS[@]}" -c "$LUA_SRC/$tu.c" -o "$OBJ_DIR/$tu.o"
  OBJS+=("$OBJ_DIR/$tu.o")
done

# --- domain driver (needs the hostcall struct) ---
# DOMAIN_EXTRA_DEFS: extra -D flags for the driver only (bisection knobs like
# -DLUA_STAGE=5 -DLUA_DBG_STAGE), mirroring SQLite's DOMAIN_EXTRA_DEFS convention.
# shellcheck disable=SC2206
DOMAIN_EXTRA_DEFS=(${DOMAIN_EXTRA_DEFS:-})
"$CLANG" "${LUA_FLAGS[@]}" "${DOMAIN_EXTRA_DEFS[@]}" -I"$HOSTCALL_DIR" \
  -c "$LIBC_DIR/lua_domain.c" -o "$OBJ_DIR/lua_domain.o"
OBJS+=("$OBJ_DIR/lua_domain.o")

# --- freestanding libc gap-fill ---
"$CLANG" "${LUA_FLAGS[@]}" -c "$LIBC_DIR/lua_libc.c" -o "$OBJ_DIR/lua_libc.o"
OBJS+=("$OBJ_DIR/lua_libc.o")

# --- capability-aware setjmp/longjmp ---
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$LIBC_DIR/capstone_setjmp.S" -o "$OBJ_DIR/setjmp.o"
OBJS+=("$OBJ_DIR/setjmp.o")

# --- the REVOKING allocator (sole owner of the static rof allocator) ---
# Sized for real Lua: many small objects, and rof never coalesces.
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -DROF_MAX_SLOTS=8192 -I"$HOSTCALL_DIR" \
  -c "$COMMON_DIR/revoke_arena_domain.c" -o "$OBJ_DIR/alloc.o"
OBJS+=("$OBJ_DIR/alloc.o")

# --- reused string + libm (capability-preserving memcpy; softfloat floor/pow/...) ---
"$CLANG" "${LUA_FLAGS[@]}" -c "$BEEBS_DIR/beebs_freestanding_string.c" \
  -o "$OBJ_DIR/beebs_string.o"
"$CLANG" "${LUA_FLAGS[@]}" -c "$BEEBS_DIR/beebs_softfloat_libm.c" \
  -o "$OBJ_DIR/beebs_libm.o"
OBJS+=("$OBJ_DIR/beebs_string.o" "$OBJ_DIR/beebs_libm.o")

# --- compiler-rt double soft-float builtins (Lua does FP; +divdf3 for division) ---
for builtin in adddf3 comparedf2 fixdfdi fixdfsi fixunsdfdi fixunsdfsi \
               floatdidf floatsidf floatunsidf fp_mode muldf3 subdf3 divdf3; do
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -fno-builtin -O0 -I"$BUILTINS_DIR" \
    -c "$BUILTINS_DIR/$builtin.c" -o "$OBJ_DIR/$builtin.o"
  OBJS+=("$OBJ_DIR/$builtin.o")
done

# --- link ---
"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${OBJS[@]}"

# Verify the revoke instruction is actually in the image — the whole point of this
# allocator is that free() REVOKES. A build without it would measure nothing.
ops=$("$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
        -ffreestanding -O0 -DROF_MAX_SLOTS=8192 -I"$HOSTCALL_DIR" \
        -S -o - "$COMMON_DIR/revoke_arena_domain.c" 2>/dev/null |
        grep -cE '^[[:space:]]+revoke\b' || true)
if [ "$ops" -eq 0 ]; then
  echo "BUILD-INVALID: no 'revoke' instruction in the allocator" >&2
  exit 3
fi

# --- host controller (guest userspace binary) ---
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
"$GUEST_CC" -O2 -I"$HOSTCALL_DIR" -o "$OUT_DIR/lua_host.user" \
  "$LIBC_DIR/lua_host.c" "$LIBCAPSTONE"

echo "built  $OUT_DOM"
echo "       host   $OUT_DIR/lua_host.user"
echo "       revoke sites in allocator: $ops"
