#!/usr/bin/env bash
# Build mruby as a Capstone domain image.
#
# Mirrors build-wamr-silicon.sh, including its three-pass link: measure .text,
# place the gp-captable globals behind it, then DECLARE the domain budget. The
# declaration is not optional even for a small image -- an image that fits anyway
# is exactly when a missing declaration goes unnoticed, which is how SQLite and
# JerryScript both shipped unloadable images.
#
# WHY MRUBY IS HERE. Every object is carved from `RVALUE objects[]` inside a GC
# page, and the free list is threaded through the objects themselves, so a
# use-after-free returns a pointer that is still tagged and still in bounds and no
# free() ever reached the allocator. That is the case standard CHERI cannot see,
# and the reason for the whole exercise. See
# agent-handoff/ref/blindspot-cases/mruby.md.
#
# TWO ALLOCATORS, DELIBERATELY, and the contrast between them IS the measurement:
#   * the OUTER one is cap_heap.c from the rv8 corpus, which narrows every returned
#     capability to exactly the request. That is CHERI-equivalent behaviour.
#   * the INNER one is mruby's own GC, handed one wide region via
#     mrb_gc_add_region(). Objects inside it are never narrowed.
# So an overflow past a malloc'd buffer faults, and the same overflow inside a GC
# page does not. Do not "fix" the second one.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

SRC=${MRUBY_SRC_DIR:-$CAPSTONE_TMP_ROOT/mruby-src}
AMALGAM_DIR="$SRC/build/host/amalgam"
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/mruby-silicon}
OBJ_DIR=$OUT_DIR/obj
DOM_NAME=${DOM_NAME:-mruby}
# The GC region, and the domain stack. The region is what makes mruby's heap ONE
# capability; size it so mruby never falls back to malloc for a page, because that
# fallback is silent and would change what is being measured.
MRUBY_REGION=${MRUBY_REGION:-$((512 * 1024))}
# The OUTER heap. umm indexes blocks with 15 bits, so this divided by
# UMM_BLOCK_BODY_SIZE must stay under 32767 -- and umm does NOT fail loudly when it
# does not: it leaves pheap NULL and the first malloc dereferences it. A
# _Static_assert in the port checks the pair, so the mistake cannot be silent twice.
MRUBY_HEAP=${MRUBY_HEAP:-$((2 * 1024 * 1024))}
MRUBY_UMM_BLOCK=${MRUBY_UMM_BLOCK:-256}
# The kernel's buddy allocator caps a region at order 10, i.e. 4 MiB, and the
# domain's data region must hold the carve AND the stack. These three numbers
# therefore have a hard ceiling between them; pass 3 refuses the build if they
# exceed it, which is how the first attempt at 4 MiB was caught.
MRUBY_STACK=${MRUBY_STACK:-$((512 * 1024))}

LADDER="$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder"
GPFREE="$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain"
RV8="$REPO_ROOT/capstone/benchmarks/rv8/adapted"
BEEBS_STRING="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
BEEBS_LIBM="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_softfloat_libm.c"
BEEBS_SOFTFLOAT="$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
COMPILER_RT=${COMPILER_RT:-$REPO_ROOT/compiler-rt/lib/builtins}
CLANG=${CAPSTONE_CLANG:-$CAPSTONE_LLVM_BIN/clang}
LD_LLD=${CAPSTONE_LD_LLD}

[[ -f "$AMALGAM_DIR/mruby.c" ]] || {
  echo "no amalgamation at $AMALGAM_DIR -- run tools/gen-mruby-sources.sh" >&2; exit 2; }
mkdir -p "$OBJ_DIR"

RESOURCE_DIR=$("$CLANG" -print-resource-dir)
COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding -fno-builtin -nostdinc -isystem "$RESOURCE_DIR/include"
        # Both LOAD-BEARING and documented in ISSUES.md: a sibling call loses its
        # epilogue, and a jump table is .rodata reached through gp that lands
        # outside its bounds.
        -fno-optimize-sibling-calls -fno-jump-tables
        -ffunction-sections -fdata-sections
        -O1 -w
        -mllvm -capstone-gp-captable
        -mllvm -capstone-shrink-stack=false
        -mllvm -capstone-shrink-globals=false
        -mllvm -capstone-merge-string-constants=true
        -DCAPSTONE_GP_CAPTABLE_ABI=1

        # THE FOUR THAT MAKE MRUBY SURVIVE A CAPABILITY TARGET. Established by
        # xlang/cheri/mruby-port for CheriBSD purecap; the same set applies here.
        #
        # MRB_NO_BOXING: mrbconf.h:62-65 defaults to MRB_WORD_BOXING when nothing
        # is chosen, which packs a pointer into an integer word and truncates it.
        # A static size assertion catches it, which is the good case.
        -DMRB_NO_BOXING
        # Otherwise proc.h packs a C function pointer as (uintptr_t)fn << 2 | flag
        # and clears the tag; calling the method then traps.
        -DMRB_USE_METHOD_T_STRUCT
        # src/pool.c picks 8; the parser's AST cons cells hold capabilities.
        -DPOOL_ALIGNMENT=16
        -DMRB_NO_STDIO
        # Small pages make whole-page frees frequent, which is how a latent
        # free-list case becomes an observable one. Taken from mruby issue 6326.
        -DMRB_HEAP_PAGE_SIZE=169

        -DCAPSTONE_HEAP_SIZE="$MRUBY_HEAP"
        -DUMM_BLOCK_BODY_SIZE="$MRUBY_UMM_BLOCK"
        # The measurement knob, not a debug hack: 1 makes the outer allocator hand
        # back WIDE arena capabilities instead of narrowing to the request, which is
        # what a purecap malloc effectively does when it rounds bounds up for
        # representability. Running both arms is the only way to tell a program that
        # overruns by a little from bounds of ours that are wrong.
        -DCAPSTONE_HEAP_NO_NARROW=${MRUBY_NO_NARROW:-0}
        -DMD_REGION_BYTES="$MRUBY_REGION"
        -include "$SCRIPT_DIR/port/capstone_mruby_libc.h"
        -I"$AMALGAM_DIR" -I"$SCRIPT_DIR/port" -I"$RV8"
        -I"$REPO_ROOT/capstone/benchmarks/micropython/adapted/include"
        -I"$REPO_ROOT/capstone/benchmarks/micropython/port"
        -I"$REPO_ROOT/capstone/benchmarks/wamr/adapted/include")

# The stack-bounds probe in mrb_vm_run (patch 0003 + port/md_probe.c). Off by
# default and added as a flag rather than as -D...=0, because the patch tests it
# with #ifdef -- defining it to zero would switch it ON, which is exactly the kind
# of knob that reads as "off" in a build log and is not.
# S-07's shape, tested here because the stage-3 fault sits on it: the clear reloads
# c->ci->stack with two ADJACENT ldc where the second's rs1 is the first's rd, which
# is the instruction pair CapstoneLdcRetry.cpp was written for. -capstone-double-ldc
# re-issues every ldc and uses the second result, and unlike the type-query retry it
# puts nothing between the pair, so it does not serialise the very overlap under
# test. If the fault goes away, the first read is delivering something wrong.
if [[ ${MRUBY_DOUBLE_LDC:-0} == 1 ]]; then
  COMMON+=(-mllvm -capstone-double-ldc)
  echo "== -capstone-double-ldc is ON: every ldc is issued twice"
fi

if [[ ${MRUBY_PROBE:-0} == 1 ]]; then
  COMMON+=(-DMD_PROBE_STACK -DMD_ESCAPE_AFTER=${MRUBY_ESCAPE_AFTER:-1000000} -DMD_PROBE_SKIP_CLEAR=${MRUBY_SKIP_CLEAR:-0} -DMD_PROBE_FORCE_STACK=${MRUBY_FORCE_STACK:-0} -DMD_PROBE_DO_CLEAR=${MRUBY_DO_CLEAR:-0})
  echo "== MD_PROBE_STACK is ON: mrb_vm_run clamps its stack clear and reports"
fi

# ONE TRANSLATION UNIT, which the gp-captable ABI requires rather than prefers.
# cap_heap.c and umm_malloc.c go INSIDE it, not beside it: both own file-scope
# globals (the arena, the umm config), and only one TU may.
echo "== amalgamating mruby, the allocator and the port into one translation unit"
ALL="$OBJ_DIR/mruby_all.c"
python3 "$SCRIPT_DIR/tools/gen-amalgam.py" "$AMALGAM_DIR" "$SCRIPT_DIR/port" "$ALL" \
  "$RV8/umm/umm_malloc.c" "$RV8/cap_heap.c" "$BEEBS_LIBM"

OBJS=()
"$CLANG" "${COMMON[@]}" -c "$ALL" -o "$OBJ_DIR/mruby_all.o"
OBJS+=("$OBJ_DIR/mruby_all.o")

echo "== compiling the shared freestanding pieces"
COMMON_FLAGS=("${COMMON[@]}" -D__SOFTFP__)
source "$BEEBS_SOFTFLOAT"
OBJS+=("${softfloat_objs[@]}")

for b in fixsfdi fixunssfdi fixunssfsi floatunsisf; do
  "$CLANG" "${COMMON_FLAGS[@]}" -I"$COMPILER_RT" -c "$COMPILER_RT/$b.c" \
    -o "$OBJ_DIR/softfloat-$b.o"
  OBJS+=("$OBJ_DIR/softfloat-$b.o")
done

"$CLANG" "${COMMON[@]}" -c "$BEEBS_STRING" -o "$OBJ_DIR/beebs_string.o"
OBJS+=("$OBJ_DIR/beebs_string.o")
"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$LADDER/../gct-section-end.S" \
  -o "$OBJ_DIR/gct.o"

link() {  # $1 = globals offset literal, $2 = output
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
# NOT optional and NOT allowed to fail quietly. The first version of this step used
# the wrong path for domreq.S and swallowed the error with `2>/dev/null || true`,
# so the build reported success on an image whose own budget check said
# "DOES NOT FIT". An image that cannot load, built by a green build, is the worst
# outcome available here -- so every part of this fails loudly.
_segs() { "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$1" | grep -E '^\s+LOAD'; }
BEFORE=$(_segs "$OUT_DIR/$DOM_NAME.dom")
CARVE=$(python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" --carve)
[[ "$CARVE" =~ ^[0-9]+$ ]] || { echo "--carve gave '$CARVE'" >&2; exit 1; }

"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -DCAPSTONE_DOMREQ_DATA=$(( CARVE + MRUBY_STACK )) -DCAPSTONE_DOMREQ_STACK=$MRUBY_STACK \
  -c "$LADDER/../domreq.S" -o "$OBJ_DIR/domreq.o"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom" "$OBJ_DIR/domreq.o"

# Non-alloc, so nothing loaded may move. Verified rather than asserted.
[[ "$BEFORE" == "$(_segs "$OUT_DIR/$DOM_NAME.dom")" ]] || {
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2; exit 2; }
echo "   declared dom_data >= $(( CARVE + MRUBY_STACK )) (carve $CARVE + stack $MRUBY_STACK)"
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" || {
  echo "the declared budget does not fit" >&2; exit 1; }

echo "== built $OUT_DIR/$DOM_NAME.dom -- one image, all stages, selected at run time"
