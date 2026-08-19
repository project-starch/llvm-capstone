#!/usr/bin/env bash
# Build JerryScript as a silicon-config Capstone domain.
#
# Uses UPSTREAM'S OWN amalgamator, tools/amalgam.py, rather than a hand-rolled one.
# That matters for three reasons and each was learned by doing it the other way first:
#
#   1. -capstone-gp-captable emits its carve descriptor PER TRANSLATION UNIT and only
#      one survives the link. Compiling 200 objects gave a table of 3 globals and 144
#      bytes of storage, with jerry_global_heap -- 512 KB of it -- not among them, so
#      the heap would have faulted on first use. One TU gives 242.
#   2. A hand-rolled amalgamation that includes every .c unconditionally breaks as soon
#      as a feature is switched off: upstream's CMake SELECTS files, and e.g.
#      ecma-builtin-regexp-string-iterator-prototype.c has no guard of its own and
#      stops compiling with JERRY_BUILTIN_REGEXP=0. 53 of the 200 files are like that.
#   3. amalgam.py also emits jerry-math, JerryScript's OWN libm. The first version of
#      this script borrowed MicroPython's lib/libm_dbl, which worked but coupled this
#      port to that checkout and measured a runtime with someone else's
#      transcendentals in it.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

JS_SRC_DIR=${JS_SRC_DIR:-$CAPSTONE_TMP_ROOT/jerryscript}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/jerryscript-silicon}
OBJ_DIR=$OUT_DIR/obj
AMALGAM_DIR=$OBJ_DIR/amalgam
DOM_NAME=${DOM_NAME:-jerryscript}
mkdir -p "$OBJ_DIR"

CLANG=${CAPSTONE_CLANG}
LD_LLD=${CAPSTONE_LD_LLD}
LADDER=$SCRIPT_DIR/../../tests/runtime-qemu/silicon-ladder
GPFREE=$SCRIPT_DIR/../../tests/runtime-qemu/gp-free-domain

echo "== amalgamating with upstream tools/amalgam.py"
( cd "$JS_SRC_DIR" && python3 tools/amalgam.py --jerry-core --jerry-math \
    --output-dir "$AMALGAM_DIR" ) 2>&1 | grep -v "^WARNING.*memcheck.h" || true
[[ -f $AMALGAM_DIR/jerryscript.c ]] || { echo "amalgamation produced nothing"; exit 1; }

# Feature switches go into the GENERATED CONFIG, not onto the compiler command line.
# amalgam.py bakes jerryscript-config.h with the defaults and the amalgamated source is
# already selected against it; a -D afterwards contradicts what was emitted and the
# build dies in the middle of jerryscript.c. Upstream's CMake overwrites this same file
# via configure_file, so rewriting it here is what that flow does.
#   JS_CONFIG="JERRY_BUILTIN_REGEXP=0 JERRY_BUILTIN_DATE=0"  bash build-...sh
for kv in ${JS_CONFIG:-}; do
  k=${kv%%=*}; v=${kv#*=}
  if grep -qE "^#define $k " "$AMALGAM_DIR/jerryscript-config.h"; then
    sed -i "s|^#define $k .*|#define $k $v|" "$AMALGAM_DIR/jerryscript-config.h"
    echo "   config: $k = $v"
  else
    echo "   !! $k is not in the generated config -- refusing to guess" >&2
    exit 1
  fi
done
echo "   jerryscript.c $(stat -c%s "$AMALGAM_DIR/jerryscript.c") bytes, jerryscript-math.c $(stat -c%s "$AMALGAM_DIR/jerryscript-math.c")"

COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding
        # -nostdlibinc, or clang searches /usr/include even for a bare-metal triple and
        # <string.h> silently resolves to the HOST glibc header.
        -nostdlibinc
        -fno-builtin -fno-optimize-sibling-calls
        # A dense switch otherwise lowers to a table of code addresses in .rodata plus an
        # indirect jump, and under gp-captable .rodata is not reachable as plain data.
        # JerryScript's opcode dispatch is exactly that shape.
        -fno-jump-tables
        # JS_OPT: the domain image has a ceiling of 5,570,560 bytes of code_len
        # (module/capstone.c needs code_len + 64K + 2*code_len to fit in one
        # __get_free_pages allocation, capped at MAX_ORDER - 1). That was 1,376,256
        # until 2026-08-19, when the kernel gained CONFIG_ARCH_FORCE_MAX_ORDER=13;
        # this image is 2,965,680 and did NOT fit before it. -O0 fits now, and it is
        # the only level that compiles -- see C-23 in the README.
        -std=c99 ${JS_OPT:--O0} -w
        # C-20: __builtin_ctz crashes the backend. patches/0001 guards the one use.
        -DJERRY_NO_BUILTIN_CTZ=1
        # jerry-math's own headers come FIRST so its math.h wins over the shim's.
        -I"$AMALGAM_DIR" -I"$SCRIPT_DIR/adapted/include" -I"$SCRIPT_DIR/port")

SILICON=(-mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         -mllvm -capstone-merge-string-constants=true
         -DCAPSTONE_GP_CAPTABLE_ABI=1
         ${DOMAIN_EXTRA_DEFS:-})

OBJS=()
echo "== compiling"
for u in jerryscript jerryscript-math; do
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -c "$AMALGAM_DIR/$u.c" -o "$OBJ_DIR/$u.o"
  OBJS+=("$OBJ_DIR/$u.o")
done
for f in jerry_domain.c capstone_setjmp.c capstone_libc_extra.c; do
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DCJ_DEFINE_SETJMP=1 \
    -c "$SCRIPT_DIR/port/$f" -o "$OBJ_DIR/port_${f%.c}.o"
  OBJS+=("$OBJ_DIR/port_${f%.c}.o")
done
BEEBS_STRING=$SCRIPT_DIR/../beebs/adapted/beebs_freestanding_string.c
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DBEEBS_STRING_LINEAR_SAFE=1 \
  -c "$BEEBS_STRING" -o "$OBJ_DIR/beebs_string.o"
OBJS+=("$OBJ_DIR/beebs_string.o")

echo "== compiling the soft-float builtins"
# JS numbers are doubles and this domain has no FP hardware ABI, so every arithmetic op
# lowers to a compiler-rt libcall. The shared BEEBS set covers the eleven the link named.
COMPILER_RT=${COMPILER_RT:-$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins}
CLANG=$CLANG OBJ_DIR=$OBJ_DIR COMPILER_RT=$COMPILER_RT
COMMON_FLAGS=("${COMMON[@]}" "${SILICON[@]}" -D__SOFTFP__)
source "$SCRIPT_DIR/../beebs/build-beebs-softfloat-common.sh"
OBJS+=("${softfloat_objs[@]}")
echo "   ${#softfloat_objs[@]} builtins"

"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$LADDER/../gct-section-end.S" -o "$OBJ_DIR/gct.o"

echo "== linking, two passes"
# The globals offset must clear .text, and .text is only known by linking. Pass 1 at a
# provisional 8 MiB measures it; pass 2 links for real.
link() {
  local lds="$OBJ_DIR/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding ${INTERP_EXTRA_CFLAGS:-} \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$LD_LLD" -T "$lds" -o "$2" "$OBJ_DIR/start.o" "${OBJS[@]}" "$OBJ_DIR/gct.o"
}
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" | python3 -c '
import sys, re
for l in sys.stdin:
    m = re.search(r"\.text\s+PROGBITS\s+\S+\s+\S+\s+(\S+)", l)
    if m: print(int(m.group(1), 16)); break')
GOFF=$(python3 -c "print(hex(max(0x10000, (($TEXT + 0xFFFF)//0x10000)*0x10000)))")
echo "   .text = $TEXT bytes -> globals offset $GOFF"
link "$GOFF" "$OUT_DIR/$DOM_NAME.dom"
echo "== built $OUT_DIR/$DOM_NAME.dom"
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" || true
