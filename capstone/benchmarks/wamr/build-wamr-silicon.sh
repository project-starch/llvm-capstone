#!/usr/bin/env bash
# Build the WAMR interpreter as a Capstone domain image.
#
# Mirrors build-micropython-silicon.sh: one translation unit set, a two-pass link
# to measure .text before placing the gp-captable globals, and a third pass that
# DECLARES the domain's budget. The declaration is not optional even though this
# image is small -- an image that fits anyway is exactly when a missing
# declaration goes unnoticed, which is how SQLite and JerryScript both shipped
# unloadable images this week.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

SRC=${WAMR_SRC_DIR:-$CAPSTONE_TMP_ROOT/wamr-src/wasm-micro-runtime}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/wamr-silicon}
OBJ_DIR=$OUT_DIR/obj
DOM_NAME=${DOM_NAME:-wamr}
WD_STAGE=${WD_STAGE:-4}
WAMR_STACK=${WAMR_STACK:-$((1024 * 1024))}

LADDER="$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder"
GPFREE="$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain"
BEEBS_STRING="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
CLANG=${CAPSTONE_CLANG:-$CAPSTONE_LLVM_BIN/clang}
LD_LLD=${CAPSTONE_LD_LLD}

[[ -d "$SRC/core" ]] || { echo "no WAMR source at $SRC -- run fetch-wamr.sh" >&2; exit 2; }
mkdir -p "$OBJ_DIR"

RESOURCE_DIR=$("$CLANG" -print-resource-dir)
COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding -fno-builtin -nostdinc -isystem "$RESOURCE_DIR/include"
        # Both LOAD-BEARING and documented in ISSUES.md: a sibling call loses its
        # epilogue, and a jump table is .rodata reached through gp that lands
        # outside its bounds.
        -fno-optimize-sibling-calls -fno-jump-tables
        -ffunction-sections -fdata-sections
        -O1 -w -DBUILD_TARGET_RISCV64_LP64
        # THE gp-captable ABI, which start-gp-captable-interp.S requires. Without
        # these the link succeeds and domdata-budget reports "cap table 0 (0
        # globals)" -- an image whose entry glue loops over an empty descriptor
        # table and hands the runtime globals that were never carved. It builds
        # and it cannot work, which is the worst combination.
        -mllvm -capstone-gp-captable
        -mllvm -capstone-shrink-stack=false
        -mllvm -capstone-shrink-globals=false
        -mllvm -capstone-merge-string-constants=true
        -DCAPSTONE_GP_CAPTABLE_ABI=1
        -DWASM_ENABLE_INTERP=1 -DWASM_ENABLE_FAST_INTERP=0
        -DWASM_ENABLE_AOT=0 -DWASM_ENABLE_JIT=0
        # Defaults to 1 in core/config.h even with AOT off, and its init builds an
        # entry table for a path this configuration does not have.
        -DWASM_ENABLE_QUICK_AOT_ENTRY=0
        -DWASM_ENABLE_LIBC_WASI=0 -DWASM_ENABLE_LIBC_BUILTIN=0
        -DWASM_ENABLE_MULTI_MODULE=0 -DWASM_ENABLE_SHARED_MEMORY=0
        -DWASM_ENABLE_THREAD_MGR=0 -DWASM_ENABLE_MEMORY_PROFILING=0
        -DWASM_ENABLE_GLOBAL_HEAP_POOL=1
        -DBH_MALLOC=wasm_runtime_malloc -DBH_FREE=wasm_runtime_free
        -I"$SCRIPT_DIR/port" -I"$SCRIPT_DIR/adapted/include"
        -I"$SRC/core/shared/utils" -I"$SRC/core/shared/platform/include"
        -I"$SRC/core/shared/mem-alloc" -I"$SRC/core/iwasm/include"
        -I"$SRC/core/iwasm/common" -I"$SRC/core/iwasm/interpreter" -I"$SRC/core")

# ONE TRANSLATION UNIT, which the gp-captable ABI requires rather than prefers:
# getGpCaptableIndex numbers globals PER MODULE, so 28 separate objects each start
# at zero and collide. Built separately the image links, reports "cap table 1 (1
# global)" and cannot work -- it builds and is wrong, which is the worst outcome
# and the reason for the gate below.
echo "== amalgamating the runtime and the port into one translation unit"
AMALGAM="$OBJ_DIR/wamr_all.c"
python3 "$SCRIPT_DIR/tools/gen-amalgam.py" "$SRC" "$SCRIPT_DIR/port" "$AMALGAM" \
  "$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_softfloat_libm.c"

OBJS=()
"$CLANG" "${COMMON[@]}" -DWD_STAGE="$WD_STAGE" -c "$AMALGAM" -o "$OBJ_DIR/wamr_all.o"
OBJS+=("$OBJ_DIR/wamr_all.o")

echo "== compiling the shared freestanding pieces"
# Reused, not rewritten. snprintf/vsnprintf come from the reference-Lua domain's
# gap-fill, which has its own native self-test; libm and the soft-float builtins
# from beebs, which every other domain in this tree links.
BEEBS_LIBM="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_softfloat_libm.c"
BEEBS_SOFTFLOAT="$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
COMPILER_RT=${COMPILER_RT:-$REPO_ROOT/compiler-rt/lib/builtins}

# beebs_libm goes INTO the amalgamation, not beside it: it owns two file-scope
# constants, and the gp-captable ABI allows exactly one TU to own globals. The
# gate below is what caught that.
# Compiled the way ITS OWN domain compiles it, with its force-included header,
# rather than against this port's includes: that header supplies FILE, struct
# lconv and the prototypes the file's definitions must match, and substituting our
# headers produced sixteen errors about types it never asked us for. Reuse means
# reusing the recipe too.

COMMON_FLAGS=("${COMMON[@]}" -D__SOFTFP__)
source "$BEEBS_SOFTFLOAT"
OBJS+=("${softfloat_objs[@]}")

# Four builtins the shared beebs list does not carry, because no benchmark there
# converts a float to a 64-bit integer. wasm does: i64.trunc_f32_s and friends.
for b in fixsfdi fixunssfdi fixunssfsi floatunsisf; do
  "$CLANG" "${COMMON_FLAGS[@]}" -I"$COMPILER_RT" -c "$COMPILER_RT/$b.c" \
    -o "$OBJ_DIR/softfloat-$b.o"
  OBJS+=("$OBJ_DIR/softfloat-$b.o")
done

"$CLANG" "${COMMON[@]}" -c "$BEEBS_STRING" -o "$OBJ_DIR/beebs_string.o"
OBJS+=("$OBJ_DIR/beebs_string.o")
"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$LADDER/../gct-section-end.S" \
  -o "$OBJ_DIR/gct.o"

link() {  # $1 = globals offset literal, $2 = output, $3.. = extra objects
  local lds="$OBJ_DIR/link.ld" off="$1" out="$2"; shift 2
  sed "s/0x10000 + 0x1000/0x10000 + $off/" "$GPFREE/link-gpfree.ld" > "$lds"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$LD_LLD" --gc-sections -T "$lds" -o "$out" \
    "$OBJ_DIR/start.o" "${OBJS[@]}" "$OBJ_DIR/gct.o" "$@"
}

echo "== gate: only one TU may own globals"
_owners=()
for _o in "${OBJS[@]}"; do
  _n=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -sW "$_o" 2>/dev/null \
        | awk '$4=="OBJECT" && $3+0>0 && $7!="UND" && $7!="ABS"' | wc -l)
  [[ "$_n" -gt 0 ]] && _owners+=("$(basename "$_o"):$_n")
done
printf '   %s\n' "${_owners[@]}"
if [[ ${#_owners[@]} -gt 1 ]]; then
  echo "more than one TU owns globals; gp-captable indices collide" >&2; exit 2
fi

echo "== pass 1: link at a provisional 8 MiB offset, only to measure .text"
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" | python3 -c '
import sys, re
for l in sys.stdin:
    m = re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2), 16)); break
else: print(0)')
[[ "${TEXT:-0}" -gt 0 ]] || { echo "could not measure .text from pass 1" >&2; exit 1; }
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 )); (( GOFF < 65536 )) && GOFF=65536
printf "   .text = %d bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"

echo "== pass 2: link with the real globals offset"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom"

echo "== pass 3: declare the domain budget"
CARVE=$(python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" --carve)
[[ "$CARVE" =~ ^[0-9]+$ ]] || { echo "--carve gave '$CARVE'" >&2; exit 1; }
_segs() { "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$1" | grep -E '^\s+LOAD'; }
BEFORE=$(_segs "$OUT_DIR/$DOM_NAME.dom")
"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -DCAPSTONE_DOMREQ_DATA=$(( CARVE + WAMR_STACK )) -DCAPSTONE_DOMREQ_STACK=$WAMR_STACK \
  -c "$LADDER/../domreq.S" -o "$OBJ_DIR/domreq.o"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom" "$OBJ_DIR/domreq.o"
# Non-alloc, so nothing loaded may move. Verified rather than asserted.
[[ "$BEFORE" == "$(_segs "$OUT_DIR/$DOM_NAME.dom")" ]] || {
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2; exit 2; }
echo "   declared dom_data >= $(( CARVE + WAMR_STACK )) (carve $CARVE + stack $WAMR_STACK)"
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" || {
  echo "the declared budget does not fit" >&2; exit 1; }
echo "== built $OUT_DIR/$DOM_NAME.dom (stage $WD_STAGE)"
