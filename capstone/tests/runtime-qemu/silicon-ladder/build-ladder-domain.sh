#!/usr/bin/env bash
# Build a silicon-ladder domain in the SILICON config: -capstone-gp-captable
# (ldc gp[i] global indirection) + gp-free call/ret + shrink OFF (the RTL
# shrink->store hazard workaround). The per-app cap-table builder glue is
# generated from the compiled domain's .capstone_gp_table descriptor.
#
#   usage: build-ladder-domain.sh <domain_main.c> <out.dom>
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
GPFREE_DIR="$SCRIPT_DIR/../gp-free-domain"
GENERIC_GLUE="$SCRIPT_DIR/start-gp-captable-generic.S"
GCT_TAIL="$SCRIPT_DIR/../gct-section-end.S"
LINKER_SCRIPT="$GPFREE_DIR/link-gpfree.ld"
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

[[ $# -eq 2 ]] || { echo "usage: $0 <domain_main.c> <out.dom>" >&2; exit 1; }
SRC=$1; OUT=$2
OBJ_DIR=${OBJ_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/capstone-ladder.XXXXXX")}
mkdir -p "$OBJ_DIR" "$(dirname -- "$OUT")"

SILICON_FLAGS=(-mllvm -capstone-gp-captable
               -mllvm -capstone-shrink-stack=false
               -mllvm -capstone-shrink-globals=false)
# +m: native integer multiply (CVA6 has the M extension) instead of a __muldi3
# libcall the freestanding domain can't link.
MARCH_FLAGS=(-Xclang -target-feature -Xclang +m)

# 1. domain_main -> descriptor-bearing object
"$CLANG" -target capstone64-unknown-elf -ffreestanding "$DOMAIN_OPT_LEVEL" \
  "${MARCH_FLAGS[@]}" "${SILICON_FLAGS[@]}" -I"$(dirname -- "$SRC")" \
  -c "$SRC" -o "$OBJ_DIR/main.o"

# 2. generate the per-app cap-table builder from the descriptor
python3 "$SCRIPT_DIR/gen-gp-captable-glue.py" "$OBJ_DIR/main.o" "$OBJ_DIR/gp_captable_build.inc"

# 3. glue (includes the generated builder) + gct tail
"$CLANG" -target capstone64-unknown-elf -ffreestanding -I"$OBJ_DIR" \
  -c "$GENERIC_GLUE" -o "$OBJ_DIR/start.o"
"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$GCT_TAIL" -o "$OBJ_DIR/gct.o"

# 4. link
"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT" "$OBJ_DIR/start.o" "$OBJ_DIR/main.o" "$OBJ_DIR/gct.o"

# static gate: silicon ABI = no cjalr, no gp cincoffset/scc-for-globals; globals
# reached via ldc gp[i] (the compiler emits `ldc rd, imm(gp)`).
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT")
NCJALR=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)
NLDCGP=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)
echo "static: cjalr=$NCJALR ldc-gp=$NLDCGP"
[[ "$NCJALR" == "0" ]] || { echo "FAIL: cjalr present (not gp-free)" >&2; exit 1; }
[[ "$NLDCGP" -ge 1 ]] || { echo "FAIL: no ldc gp[i] global access found" >&2; exit 1; }
echo "Built $OUT"
