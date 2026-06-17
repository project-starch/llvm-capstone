#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-mergesort}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_mergesort_capstone.dom}

MERGESORT_SRC=$BEEBS_SRC_DIR/src/mergesort/libmergesort.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_MERGESORT_SRC=$OUT_DIR/libmergesort_capstone.c
MERGESORT_TAIL_SRC=$SCRIPT_DIR/adapted/beebs_mergesort_capstone_tail.c

if [[ ! -f "$MERGESORT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS mergesort source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$MERGESORT_TAIL_SRC" ]]; then
  echo "missing adapted mergesort tail source: $MERGESORT_TAIL_SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Capstone-specific patches applied to the upstream source:
#
# 1. Strip <stdio.h>, <stdlib.h>, <string.h>, <math.h> — not available
#    freestanding.  Replace <stdio.h> with a forward declaration of memcpy
#    (defined in the tail file).
#
# 2. Strip from `typedef bool (*Comparison)` onwards (everything before
#    TestingPathological is replaced by the tail).  The tail provides:
#    - Sort functions (BinaryLast, InsertionSort, MergeSortR, MergeSort) with
#      inline comparison — no function pointer parameter.  Function pointers
#      in Capstone domains require capability-tagged values; the domain ELF
#      loader does not process relocations so function pointer arrays in .data
#      are untagged and cjalr through them faults.
#    - Test-data generators with integer FP replacements.
#    - benchmark() using switch dispatch (no function pointer table).
#    - verify_benchmark() with global const arrays (avoids stc Bug #9).
#
# 3. Change `static long int seed` to `static int seed` in rand_beebs while
#    generating the retained upstream prefix.
#    On Capstone, global/static variable access uses delin (delinearize) to
#    strip the capability tag, then ld/sd to load/store.  Only 32-bit lw/sw
#    work with the delinearized integer pointer; 64-bit ld/sd require a
#    capability register and fault.  int (32 bits) causes the compiler to
#    emit lw/sw instead of ld/sd.  The 31-bit mask keeps the value in range.
awk '
  BEGIN { patched = 0 }
  /^#include <stdio\.h>$/ {
    if (!patched) {
      print "extern void *memcpy(void *, const void *, unsigned long);"
      patched = 1
    }
    next
  }
  /^#include <stdlib\.h>$/ { next }
  /^#include <string\.h>$/ { next }
  /^#include <math\.h>$/   { next }
  /^  static long int seed = 0;$/ { print "  static int seed = 0;"; next }
  /^typedef bool \(\*Comparison\)/ { exit }
  { print }
' "$MERGESORT_SRC" > "$PATCHED_MERGESORT_SRC"

cat "$MERGESORT_TAIL_SRC" >> "$PATCHED_MERGESORT_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_MERGESORT_SRC" \
  -o "$OBJ_DIR/beebs_mergesort.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_mergesort_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_mergesort.o" \
  "$OBJ_DIR/beebs_mergesort_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
