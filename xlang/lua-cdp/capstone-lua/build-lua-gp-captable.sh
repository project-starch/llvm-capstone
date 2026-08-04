#!/usr/bin/env bash
# Build the real-Lua-5.4.7 Capstone domain on the gp-captable ("interpreter-scale")
# ABI + its host controller.
#
#   ./build-lua-gp-captable.sh                 # normal (staged) build
#   LUA_STAGE=5 ./build-lua-gp-captable.sh      # staged bisection harness (default here)
#
# WHY THIS EXISTS (vs build-lua-domain.sh, the cjalr build). Under the cjalr ABI the
# static capability tables (base_funcs[], luaX_tokens[], ...) are tagged by the
# .capstone_cap_init walk, which under-tags at Lua's scale -> luaopen_base reads an
# untagged capability and faults (cause 24). The gp-captable ABI moves every global
# behind `ldc gp[i]` and tags them from a single .capstone_gp_initdesc descriptor the
# entry glue (start-gp-captable-interp.S) walks -- the same mechanism that unblocked
# SQLite. See build-sqlite-silicon.sh, which this mirrors.
#
# THE ONE HARD CONSTRAINT (asserted below). The descriptor carries a per-MODULE
# {built_flag,count,...} header and the glue reads exactly ONE header's count, so
# only ONE module may own gp-captable globals. Everything with globals is folded into
# lua_gp_amalg.c; every separately-linked object must be provably globals-free.
#
# -O0 like every domain here. The allocator is the revoking one (SPLIT/MREV/REVOKE).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
READELF="$CAPSTONE_LLVM_BIN/llvm-readelf"
LUA_SRC="$CAPSTONE_REPO_ROOT/xlang/lua-cdp/_toolchain/.work/lua54"
LIBC_DIR="$CAPSTONE_REPO_ROOT/xlang/lua-cdp/capstone-lua"
COMMON_DIR="$CAPSTONE_REPO_ROOT/xlang/common"
SQLITE_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"      # hostcall + floatdidf-noglobals
BEEBS_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/adapted"
BUILTINS_DIR="$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins"
LADDER="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder"
GPFREE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/lua-gp-captable}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/lua_gp.dom}
mkdir -p "$OBJ_DIR"

# The gp-captable / gp-free silicon flags, matching build-sqlite-silicon.sh exactly.
SILICON=(-mllvm -capstone-merge-string-constants=true
         -mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         -fno-jump-tables
         -DCAPSTONE_GP_CAPTABLE_ABI=1)

# Freestanding recipe for the Lua/libc/domain TUs (the header-shadow technique:
# -nostdlibinc + empty hosted stubs + the force-included decl header).
COMMON=(-target capstone64-unknown-elf
        -Xclang -target-feature -Xclang +m
        -ffreestanding -nostdlibinc -fno-builtin
        -ffunction-sections -fdata-sections -O0
        -include "$LIBC_DIR/capstone_lua_libc.h"
        -I"$LIBC_DIR" -I"$LIBC_DIR/include" -I"$LUA_SRC"
        -I"$COMMON_DIR" -I"$BEEBS_DIR" -I"$SQLITE_DIR")

STAGE_DEF=()
[[ -n "${LUA_STAGE:-5}" ]] && STAGE_DEF=(-DLUA_STAGE="${LUA_STAGE:-5}")
# LUA_DBG_STAGE=1 turns on csdebugprint stage markers that survive a domain wedge (QEMU only).
[[ -n "${LUA_DBG_STAGE:-}" ]] && STAGE_DEF+=(-DLUA_DBG_STAGE)
# LUA_SKIP_BASE=1 runs the pure-core chunk without opening the base library.
[[ -n "${LUA_SKIP_BASE:-}" ]] && STAGE_DEF+=(-DLUA_SKIP_BASE)

# Assert an object owns NO gp-captable globals (no .capstone_gp_initdesc). The whole
# ABI depends on exactly one module owning globals; a silently-globals-owning support
# object would corrupt the descriptor and fault like the cjalr build did.
assert_globals_free() { # $1 = object
  if "$READELF" -SW "$1" 2>/dev/null | grep -q '\.capstone_gp_initdesc'; then
    echo "BUILD-INVALID: $(basename "$1") owns gp-captable globals (must be folded into lua_gp_amalg.c)" >&2
    exit 3
  fi
}

OBJS=()

echo "== the single globals-owning module (all of Lua + libc + libm + allocator + harness)"
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" "${STAGE_DEF[@]}" -DROF_MAX_SLOTS=${ROF_MAX_SLOTS:-1024} \
  -I"$SQLITE_DIR" -c "$LIBC_DIR/lua_gp_amalg.c" -o "$OBJ_DIR/amalgam.o"
# It MUST own exactly one descriptor header. Match only the data section, not its
# relocation section (.rela.capstone_gp_initdesc), by requiring whitespace before the
# name -- ".rela.capstone..." has 'a' there, so it does not match.
n=$("$READELF" -SW "$OBJ_DIR/amalgam.o" 2>/dev/null | grep -cE '[[:space:]]\.capstone_gp_initdesc[[:space:]]')
[[ "$n" == "1" ]] || { echo "BUILD-INVALID: amalgam.o has $n descriptor sections (want 1)" >&2; exit 3; }
OBJS+=("$OBJ_DIR/amalgam.o")

echo "== globals-free support objects (linked separately; asserted globals-free)"
# beebs string primitives (LINEAR-safe, no globals).
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DBEEBS_STRING_LINEAR_SAFE=1 \
  -c "$BEEBS_DIR/beebs_freestanding_string.c" -o "$OBJ_DIR/beebs_string.o"
assert_globals_free "$OBJ_DIR/beebs_string.o"
OBJS+=("$OBJ_DIR/beebs_string.o")

# NOTE: __floatdidf is NOT linked here -- at -O0 it owns .rodata constant locals, so it
# is folded into lua_gp_amalg.c (the single globals-owning module) instead.

# The compiler-rt double soft-float builtins (all globals-free; asserted).
for b in adddf3 subdf3 muldf3 divdf3 comparedf2 eqdf2 gedf2 gtdf2 ltdf2 nedf2 \
         fixdfdi fixdfsi fixunsdfdi fixunsdfsi floatsidf floatunsidf fp_mode; do
  [[ -f "$BUILTINS_DIR/$b.c" ]] || continue
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -O0 -I"$BUILTINS_DIR" \
    -c "$BUILTINS_DIR/$b.c" -o "$OBJ_DIR/$b.o"
  assert_globals_free "$OBJ_DIR/$b.o"
  OBJS+=("$OBJ_DIR/$b.o")
done

# capability-aware setjmp/longjmp (gp-captable ABI: scalar ra, capability s-regs).
"$CLANG" -target capstone64-unknown-elf -ffreestanding -O0 \
  -c "$LIBC_DIR/capstone_setjmp.S" -o "$OBJ_DIR/setjmp.o"
OBJS+=("$OBJ_DIR/setjmp.o")

# Assert the revoke instruction is really in the allocator (free() must REVOKE).
ops=$("$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DROF_MAX_SLOTS=${ROF_MAX_SLOTS:-1024} -I"$SQLITE_DIR" \
        -S -o - "$COMMON_DIR/revoke_arena_domain.c" 2>/dev/null |
        grep -cE '^[[:space:]]+revoke\b' || true)
[[ "$ops" -gt 0 ]] || { echo "BUILD-INVALID: no 'revoke' instruction in the allocator" >&2; exit 3; }

# Two-pass link (build-sqlite-silicon.sh pattern): pass 1 sizes .text, pass 2 places
# the globals region at that offset. start-gp-captable-interp.S is the entry glue.
link() { # $1 = globals-offset literal, $2 = output
  local lds="$OBJ_DIR/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  # INTERP_EXTRA_CFLAGS lets a diagnostic build enable the interp start's csdebugprint
  # peeks (-DINTERP_PEEK_SP / _GP / _PRECALL) without editing this script. QEMU-only.
  "$CLANG" -target capstone64-unknown-elf -ffreestanding ${INTERP_EXTRA_CFLAGS:-} \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$LD_LLD" --gc-sections -T "$lds" -o "$2" "$OBJ_DIR/start.o" "${OBJS[@]}"
}

echo "== pass 1: link at a provisional 8 MiB offset, only to measure .text"
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$READELF" -SW "$OUT_DIR/pass1.dom" 2>/dev/null | python3 -c '
import sys,re
for l in sys.stdin:
    m=re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2),16)); break
else: print(0)')
: "${TEXT:=0}"
[[ "$TEXT" -gt 0 ]] || { echo "could not measure .text from pass 1" >&2; exit 1; }
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 ))
[[ $GOFF -lt 65536 ]] && GOFF=65536
printf "   .text = %d bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"

echo "== pass 2: link with the real globals offset"
link "$(printf '0x%x' $GOFF)" "$OUT_DOM"

echo "== gates"
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DOM")
echo "   cjalr=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)  ldc-gp=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)  revoke-sites=$ops"

# host controller (guest userspace binary; libcapstone packs the globals offset).
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
"$GUEST_CC" -O2 -I"$SQLITE_DIR" -o "$OUT_DIR/lua_host.user" \
  "$LIBC_DIR/lua_host.c" "$LIBCAPSTONE"

echo "built  $OUT_DOM"
echo "       host   $OUT_DIR/lua_host.user"
