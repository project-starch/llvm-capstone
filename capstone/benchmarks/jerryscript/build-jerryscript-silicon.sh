#!/usr/bin/env bash
# Build JerryScript as a silicon-config Capstone domain.
#
# Same structure as ../micropython/build-micropython-silicon.sh: compile every core
# source with the domain flags, add the port glue, link at a globals offset sized to
# .text. Kept separate rather than generalised -- the two ports differ in what they
# need, and a shared script that handles both would be harder to read than either.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

JS_SRC_DIR=${JS_SRC_DIR:-$CAPSTONE_TMP_ROOT/jerryscript}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/jerryscript-silicon}
OBJ_DIR=$OUT_DIR/obj
DOM_NAME=${DOM_NAME:-jerryscript}
mkdir -p "$OBJ_DIR"

CLANG=${CAPSTONE_CLANG}
LD_LLD=${CAPSTONE_LD_LLD}
LADDER=$SCRIPT_DIR/../../tests/runtime-qemu/silicon-ladder
GPFREE=$SCRIPT_DIR/../../tests/runtime-qemu/gp-free-domain

COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding
        # -nostdlibinc, or clang searches /usr/include even for a bare-metal triple
        # and <string.h> silently resolves to the HOST glibc header.
        -nostdlibinc
        -fno-builtin -fno-optimize-sibling-calls
        # A dense switch otherwise lowers to a table of code addresses in .rodata plus
        # an indirect jump, and under gp-captable .rodata is not reachable as plain
        # data. JerryScript's opcode dispatch is exactly that shape.
        -fno-jump-tables
        -std=c99 -O0 -w
        # C-20: __builtin_ctz crashes the backend. patches/0001 guards the one use.
        -DJERRY_NO_BUILTIN_CTZ=1
        -I"$SCRIPT_DIR/adapted/include" -I"$SCRIPT_DIR/port")
while IFS= read -r d; do COMMON+=(-I"$d"); done < <(find "$JS_SRC_DIR/jerry-core" -type d | sort)

SILICON=(-mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         -mllvm -capstone-merge-string-constants=true
         -DCAPSTONE_GP_CAPTABLE_ABI=1
         ${DOMAIN_EXTRA_DEFS:-})

echo "== compiling jerry-core"
OBJS=()
while IFS= read -r f; do
  o="$OBJ_DIR/$(echo "${f#$JS_SRC_DIR/}" | tr / _).o"
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -c "$f" -o "$o"
  OBJS+=("$o")
done < <(find "$JS_SRC_DIR/jerry-core" -name '*.c' | sort)
echo "   ${#OBJS[@]} objects"

echo "== compiling the port glue"
for f in jerry_domain.c capstone_setjmp.c capstone_libc_extra.c; do
  o="$OBJ_DIR/port_${f%.c}.o"
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" ${f:+-DCJ_DEFINE_SETJMP=1} -c "$SCRIPT_DIR/port/$f" -o "$o"
  OBJS+=("$o")
done

echo "== compiling the shared freestanding string functions"
BEEBS_STRING=$SCRIPT_DIR/../beebs/adapted/beebs_freestanding_string.c
if [[ -f $BEEBS_STRING ]]; then
  "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DBEEBS_STRING_LINEAR_SAFE=1 \
    -c "$BEEBS_STRING" -o "$OBJ_DIR/beebs_string.o"
  OBJS+=("$OBJ_DIR/beebs_string.o")
fi

echo "== compiling the soft-float builtins"
# JS numbers are doubles and this domain has no FP hardware ABI, so every arithmetic
# op lowers to a compiler-rt libcall. The first link named exactly eleven of them and
# the shared BEEBS set covers all eleven -- reused rather than reimplemented.
COMPILER_RT=${COMPILER_RT:-$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins}
if [[ -d $COMPILER_RT ]]; then
  CLANG=$CLANG OBJ_DIR=$OBJ_DIR COMPILER_RT=$COMPILER_RT
  COMMON_FLAGS=("${COMMON[@]}" "${SILICON[@]}" -D__SOFTFP__)
  source "$SCRIPT_DIR/../beebs/build-beebs-softfloat-common.sh"
  OBJS+=("${softfloat_objs[@]}")
  echo "   ${#softfloat_objs[@]} builtins"
else
  echo "   !! no compiler-rt at $COMPILER_RT -- the link will name the missing builtins"
fi

echo "== compiling libm"
# MicroPython's lib/libm_dbl, the SAME math the MicroPython port measured with. That
# is a real coupling to that checkout and it is deliberate: two runtimes compared
# against each other should not differ in their transcendentals. 18 of the 20 symbols
# the link named come from here; fabs and cbrt are in port/capstone_libc_extra.c.
LIBM_DIR=${LIBM_DIR:-${MPY_SRC_DIR:-$CAPSTONE_TMP_ROOT/micropython}/lib/libm_dbl}
if [[ -d $LIBM_DIR ]]; then
  n=0
  for f in "$LIBM_DIR"/*.c; do
    b=$(basename "$f" .c)
    # thumb_vfp_sqrt is ARM-only. __fpclassify needs FP_NAN and friends, which the
    # freestanding <math.h> shim does not define -- excluded for the same reason the
    # MicroPython build excludes it, and if something ever calls it the link says so.
    case $b in thumb_vfp_sqrt|__fpclassify) continue;; esac
    o="$OBJ_DIR/libm_$b.o"
    "$CLANG" "${COMMON[@]}" "${SILICON[@]}" -D__SOFTFP__ -c "$f" -o "$o"
    OBJS+=("$o"); n=$((n+1))
  done
  echo "   $n units from $LIBM_DIR"
else
  echo "   !! no libm at $LIBM_DIR -- the link will name the missing functions"
fi

echo "== linking, two passes"
# The globals offset must clear .text, and .text is only known by linking. Pass 1 at a
# provisional 8 MiB measures it; pass 2 links for real. Same shape as
# build-micropython-silicon.sh, and the reason is the same: link-gpfree.ld hardcodes
# the boundary and a too-small offset silently overlaps globals with code.
link() {  # $1 = globals offset literal, $2 = output
  local lds="$OBJ_DIR/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding ${INTERP_EXTRA_CFLAGS:-} \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$LD_LLD" -T "$lds" -o "$2" "$OBJ_DIR/start.o" "${OBJS[@]}" "$OBJ_DIR/gct.o"
}

link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" 2>/dev/null | python3 -c '
import sys, re
for l in sys.stdin:
    m = re.search(r"\.text\s+PROGBITS\s+\S+\s+\S+\s+(\S+)", l)
    if m: print(int(m.group(1), 16)); break')
GOFF=$(python3 -c "t=$TEXT; import sys; print(hex(max(0x10000, ((t + 0xFFFF)//0x10000)*0x10000)))")
echo "   .text = $TEXT bytes -> globals offset $GOFF"
link "$GOFF" "$OUT_DIR/$DOM_NAME.dom"
echo "== built $OUT_DIR/$DOM_NAME.dom"
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" 2>/dev/null || true
