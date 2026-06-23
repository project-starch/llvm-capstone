#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-whetstone}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_whetstone_capstone.dom}

WHET_SRC=$BEEBS_SRC_DIR/src/whetstone/libwhetstone.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
LIBM_SRC=$SCRIPT_DIR/adapted/beebs_softfloat_libm.c
PATCHED_SRC=$OUT_DIR/libwhetstone_capstone.c
TAIL_SRC=$SCRIPT_DIR/adapted/beebs_whetstone_capstone_tail.c

for f in "$WHET_SRC" "$SUPPORT_DIR/support.h" "$LIBM_SRC" "$TAIL_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "missing required source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip hosted includes; provide freestanding math prototypes; rename the
# upstream -1 verify stub; drop the upstream printf POUT *definition* (the last
# function in the file) so the tail's capturing POUT is the only one.  The tail
# (built -DPRINTOUT) folds every module's POUT outputs into an FNV checksum.
{
  printf 'double sin(double); double cos(double); double atan(double);\n'
  printf 'double log(double); double exp(double); double sqrt(double);\n'
  printf '#define verify_benchmark whet_orig_verify\n'
  sed -e '/#include <stdio.h>/d' \
      -e '/#include <stdlib.h>/d' \
      -e '/#include <string.h>/d' \
      -e '/#include <math.h>/d' \
      "$WHET_SRC" \
    | awk '
        { line[NR] = $0 }
        END {
          def = 0
          for (i = 1; i <= NR; i++) if (line[i] ~ /^POUT\(long N/) { def = i; break }
          cut = def
          for (i = def; i >= 1; i--) if (line[i] ~ /^#ifdef PRINTOUT/) { cut = i; break }
          for (i = 1; i < cut; i++) print line[i]
        }
      '
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
  -DPRINTOUT
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/libwhetstone.o"
objs+=("$OBJ_DIR/libwhetstone.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$LIBM_SRC" -o "$OBJ_DIR/whetstone_libm.o"
objs+=("$OBJ_DIR/whetstone_libm.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_whetstone_domain.o"
objs+=("$OBJ_DIR/beebs_whetstone_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
