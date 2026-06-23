#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-fasta}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_fasta_capstone.dom}

FASTA_SRC=$BEEBS_SRC_DIR/src/fasta/libfasta.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
STRING_SRC=$SCRIPT_DIR/adapted/beebs_freestanding_string.c
PATCHED_SRC=$OUT_DIR/libfasta_capstone.c
TAIL_SRC=$SCRIPT_DIR/adapted/beebs_fasta_capstone_tail.c

for f in "$FASTA_SRC" "$SUPPORT_DIR/support.h" "$STRING_SRC" "$TAIL_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "missing required source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip the hosted <string.h>/<stdlib.h> includes (size_t now comes from
# <stddef.h>), and drop the upstream consumers from `repeat_fasta` onward
# (repeat_fasta/random_fasta/verify_benchmark/initialise_benchmark/benchmark),
# keeping the deterministic generator core (myrandom/accumulate_probabilities,
# the aminoacid_t typedef, and the WIDTH/MIN/NELEMENTS macros). The tail
# reimplements the consumers to FNV-checksum the generated characters.
{
  printf '#include <stddef.h>\n'
  sed -e '/#include <string.h>/d' \
      -e '/#include <stdlib.h>/d' \
      "$FASTA_SRC" \
    | awk '/^static void repeat_fasta/ { exit } { print }'
  cat "$TAIL_SRC"
} > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  -ffp-contract=off
  -ffunction-sections
  -fdata-sections
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/libfasta.o"
objs+=("$OBJ_DIR/libfasta.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$STRING_SRC" -o "$OBJ_DIR/fasta_string.o"
objs+=("$OBJ_DIR/fasta_string.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_fasta_domain.o"
objs+=("$OBJ_DIR/beebs_fasta_domain.o")

# --gc-sections drops the kept-but-unused generator helpers and any string
# routines fasta does not reference.
"$LD_LLD" -T "$LINKER_SCRIPT" --gc-sections -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
