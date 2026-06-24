#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RV8_BENCH=primes

source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
"$SCRIPT_DIR/fetch-rv8.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
RV8_SRC_DIR=${RV8_SRC_DIR:-$CAPSTONE_TMP_ROOT/rv8-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/rv8-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-$RV8_BENCH}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/rv8_${RV8_BENCH}_capstone.dom}

BEEBS_DIR=$REPO_ROOT/capstone/benchmarks/beebs
ADAPTED_DIR=$SCRIPT_DIR/adapted
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
LIBM_SRC=$BEEBS_DIR/adapted/beebs_softfloat_libm.c
DOMAIN_SRC=$BEEBS_DIR/beebs_simple_domain.c
SRC=$RV8_SRC_DIR/src/primes.c
PATCHED_SRC=$OUT_DIR/primes_capstone.c

for f in "$SRC" "$LIBM_SRC" "$DOMAIN_SRC" "$ADAPTED_DIR/rv8_capstone_preamble.h"; do
  [[ -f "$f" ]] || { echo "missing required source: $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Transform the tiny all-in-main sieve into a value-returning rv8_primes_run():
#  - keep freestanding <stdint.h>; strip hosted stdio/stdlib/math;
#  - reduce the sieve limit (33,333,333 -> 100000) to fit the domain bump arena;
#  - main() -> rv8_primes_run(); print-the-prime -> return it; drop `return 0`;
#  - fix the latent UB `1 << (p&0x3f)` (32-bit int shifted up to 63) -> `1ull <<`,
#    matching intent (correct at any limit; largest prime <= 100000 == 99991).
sed -E -e '/^[[:space:]]*#[[:space:]]*include[[:space:]]+<(stdio|stdlib|math)\.h>/d' \
       -e 's/int limit = 33333333;/int limit = 100000;/' \
       -e 's/int main\(\)/int rv8_primes_run(void)/' \
       -e 's/printf\(.*\);/return i;/' \
       -e 's/return 0;//' \
       -e 's/1 << \(p & 0x3f\)/1ull << (p \& 0x3f)/g' \
       "$SRC" > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  -ffp-contract=off
  "$DOMAIN_OPT_LEVEL"
  -include "$ADAPTED_DIR/rv8_capstone_preamble.h"
  -I"$ADAPTED_DIR"
  -Wno-incompatible-library-redeclaration
  -Wno-implicit-function-declaration
  -Wno-builtin-requires-header
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/primes.o"; objs+=("$OBJ_DIR/primes.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$ADAPTED_DIR/rv8_primes_tail.c" -o "$OBJ_DIR/tail.o"; objs+=("$OBJ_DIR/tail.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$ADAPTED_DIR/rv8_malloc.c" -o "$OBJ_DIR/malloc.o"; objs+=("$OBJ_DIR/malloc.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$LIBM_SRC" -o "$OBJ_DIR/libm.o"; objs+=("$OBJ_DIR/libm.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$DOMAIN_SRC" -o "$OBJ_DIR/domain.o"; objs+=("$OBJ_DIR/domain.o")

# Soft-float compiler-rt builtins (sqrt + double arithmetic in the sieve/libm).
source "$BEEBS_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"
"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null
[[ -f "$OUT_DOM" ]] || { echo "failed to build $OUT_DOM" >&2; exit 1; }
echo "Built $OUT_DOM"
