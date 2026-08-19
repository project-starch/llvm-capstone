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
INTERP_GLUE="$SCRIPT_DIR/start-gp-captable-interp.S"
# DOMAIN_GLUE=interp selects the descriptor-driven entry glue: a FIXED loop over
# .capstone_gp_initdesc instead of the per-app unrolled prologue that
# gen-gp-captable-glue.py emits. Opt-in while both paths coexist, so every measured
# rung keeps its current binary until the interpreter is gated on the whole ladder.
DOMAIN_GLUE=${DOMAIN_GLUE:-generated}
GCT_TAIL="$SCRIPT_DIR/../gct-section-end.S"
# DOMAIN_WINDOW=32k selects the 32 KiB code window (globals at image offset 0x8000)
# instead of the default 4 KiB (0x1000) -- issue C-5.
#
# Opt-in PER RUNG, deliberately not a global default: changing the window changes
# image layout, and this project has a documented layout sensitivity (a 2026-07-26
# A/B where four added instructions flipped a passing rung). Every measured rung
# stays at 4 KiB so its published number stands; only rungs that cannot build at
# 4 KiB opt in, and they are re-gated on their oracle.
#
# Why it matters beyond code size: with 32 KiB there is room to materialise a large
# initializer with unrolled li/sd instead of the large-RO COPY path, and the copy
# path is broken (C-4b, cssplit tag assertion). So the bigger window does not just
# raise a limit, it lets a whole class of rungs avoid a broken code path.
# The code window is now produced by substituting the offset into the ONE linker
# script, instead of maintaining a second copy of it (link-gpfree-32k.ld). That
# second copy was the bug: the monitor had a single hardcoded 0x1000, so a 32 KiB
# window plus the copy path silently disagreed about where the blob started.
LINKER_SCRIPT=${LINKER_SCRIPT:-"$GPFREE_DIR/link-gpfree.ld"}
case "${DOMAIN_WINDOW:-}" in
  "")   GLOBALS_OFF_VAL=0x1000 ;;
  32k)  GLOBALS_OFF_VAL=0x8000 ;;
  *)    GLOBALS_OFF_VAL=${DOMAIN_WINDOW} ;;
esac
if [[ "$GLOBALS_OFF_VAL" != "0x1000" ]]; then
  _lds=$(mktemp "${TMPDIR:-/tmp}/link-gpfree-off-XXXXXX.ld")
  sed "s/0x10000 + 0x1000/0x10000 + $GLOBALS_OFF_VAL/" "$LINKER_SCRIPT" > "$_lds"
  LINKER_SCRIPT=$_lds
  echo "  globals offset -> $GLOBALS_OFF_VAL"
fi
# DOMAIN_BASE_VA relocates the domain's entry VA (default 0x10000).
#
# Purpose: test whether issue R-3 is really about the entry ADDRESS. R-3 says a
# second domain reused at entry VA 0x10000 within one boot silently hangs its
# cscall -- read as a missing icache invalidate on domain switch. If that is
# address-keyed, two domains at DIFFERENT VAs should coexist in one boot, which
# would remove the full power-cycle + JTAG reload we currently pay per rung
# (~2.5 min, the dominant cost of every sweep). Nobody has tested it.
#
# Substituted into a temp copy of the script rather than passed via --defsym:
# ld.lld resolves --defsym after the script is evaluated, so DEFINED() inside
# SECTIONS would not see it and the base would silently stay 0x10000 -- exactly
# the kind of silent wrong-config that cost us a sweep on 2026-07-27.
DOMAIN_BASE_VA=${DOMAIN_BASE_VA:-}
if [[ -n "$DOMAIN_BASE_VA" && "$DOMAIN_BASE_VA" != "0x10000" ]]; then
  _relocated=$(mktemp "${TMPDIR:-/tmp}/link-gpfree-XXXXXX.ld")
  sed "s/0x10000/$DOMAIN_BASE_VA/g" "$LINKER_SCRIPT" > "$_relocated"
  LINKER_SCRIPT=$_relocated
  echo "  relocated domain base VA -> $DOMAIN_BASE_VA"
fi
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
# Extra -D/-f flags for one build (e.g. -DLADDER_NO_MINSTRET for the
# un-instrumented control variant). Word-split on purpose.
DOMAIN_EXTRA_CFLAGS=${DOMAIN_EXTRA_CFLAGS:-}

[[ $# -eq 2 ]] || { echo "usage: $0 <domain_main.c> <out.dom>" >&2; exit 1; }
SRC=$1; OUT=$2
OBJ_DIR=${OBJ_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/capstone-ladder.XXXXXX")}
mkdir -p "$OBJ_DIR" "$(dirname -- "$OUT")"

SILICON_FLAGS=(-mllvm -capstone-gp-captable
               -mllvm -capstone-shrink-stack=false
               -mllvm -capstone-shrink-globals=false
               -fno-jump-tables)
# -fno-jump-tables: a dense `switch` otherwise lowers to a table of code addresses
# in .rodata plus an indirect `jr`. In a gp-captable domain .rodata is not reachable
# as plain data (globals come from the gp cap-table), so loading the table faults:
# "Cap mem access requires capability" -> "domain halted by capability fault:
# cause = 24" with a wild PC, which looks exactly like a domain crash. Costs a few
# compares; without it any switch-heavy workload (SQLite) breaks this way.
# +m: native integer multiply (CVA6 has the M extension) instead of a __muldi3
# libcall the freestanding domain can't link.
MARCH_FLAGS=(-Xclang -target-feature -Xclang +m)

# 1. domain_main -> descriptor-bearing object
"$CLANG" -target capstone64-unknown-elf -ffreestanding "$DOMAIN_OPT_LEVEL" \
  "${MARCH_FLAGS[@]}" "${SILICON_FLAGS[@]}" ${DOMAIN_EXTRA_CFLAGS} \
  -I"$(dirname -- "$SRC")" \
  -c "$SRC" -o "$OBJ_DIR/main.o"

if [[ "$DOMAIN_GLUE" == "interp" ]]; then
  # 2/3. Descriptor-driven glue: nothing to generate. The compiler emitted
  # .capstone_gp_initdesc, the linker script puts it first in the globals region
  # (blob offset 0), the monitor copies it into dom_data, and this glue interprets
  # it at entry. O(1) .text regardless of how many globals the domain has.
  echo "  glue: descriptor-driven interpreter (no generated prologue)"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
    ${INTERP_FAKE_COUNT:+-DINTERP_FAKE_COUNT=$INTERP_FAKE_COUNT} \
    ${INTERP_DIAG_STAGE:+-DINTERP_DIAG_STAGE=$INTERP_DIAG_STAGE} \
    ${INTERP_PEEK_OFF:+-DINTERP_PEEK_OFF=$INTERP_PEEK_OFF} \
    ${INTERP_PEEK_MODE:+-DINTERP_PEEK_MODE=$INTERP_PEEK_MODE} \
    ${INTERP_SKIP_CAPINIT:+-DINTERP_SKIP_CAPINIT=$INTERP_SKIP_CAPINIT} \
    ${INTERP_SKIP_GPPARK:+-DINTERP_SKIP_GPPARK=$INTERP_SKIP_GPPARK} \
    ${INTERP_BUILD_LIMIT:+-DINTERP_BUILD_LIMIT=$INTERP_BUILD_LIMIT} \
    ${INTERP_DOMAIN_MTVEC:+-DINTERP_DOMAIN_MTVEC=$INTERP_DOMAIN_MTVEC} \
    -c "$INTERP_GLUE" -o "$OBJ_DIR/start.o"
else
  # 2. generate the per-app cap-table builder from the descriptor
  python3 "$SCRIPT_DIR/gen-gp-captable-glue.py" "$OBJ_DIR/main.o" "$OBJ_DIR/gp_captable_build.inc"

  # 3. glue (includes the generated builder) + gct tail
  "$CLANG" -target capstone64-unknown-elf -ffreestanding -I"$OBJ_DIR" \
    -c "$GENERIC_GLUE" -o "$OBJ_DIR/start.o"
fi
"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$GCT_TAIL" -o "$OBJ_DIR/gct.o"

# 4. link
"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT" "$OBJ_DIR/start.o" "$OBJ_DIR/main.o" "$OBJ_DIR/gct.o"

# 4b. DOMAIN_STACK=<bytes> makes the image DECLARE what it needs instead of leaving the
# module to infer it from code size. Two passes, because the declaration depends on the
# carve and the carve depends on the link: pass 1 above is measured by domdata-budget.py
# (--carve = initializer blob + cap table + per-global storage), the declared total is
# that plus the requested stack, and pass 2 relinks with the section in.
#
# domreq.S is non-alloc, so adding it moves nothing in the loaded image -- deliberately,
# because this project has a documented layout sensitivity where four added instructions
# flipped a passing rung. Verified below rather than asserted.
if [[ -n "${DOMAIN_STACK:-}" ]]; then
  _carve=$(python3 "$SCRIPT_DIR/domdata-budget.py" "$OUT" --carve)
  [[ "$_carve" =~ ^[0-9]+$ ]] || { echo "FAIL: --carve gave '$_carve'" >&2; exit 1; }
  _need=$(( _carve + DOMAIN_STACK ))
  # Compare the SEGMENTS, not readelf's whole -lW text: that text also lists the
  # section-to-segment mapping, which of course gains a name, so comparing it makes
  # the gate fire on every declared build. The question is whether any LOADED byte
  # moved, and that is exactly the LOAD/NULL offsets and sizes.
  _segs() { "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$1" | grep -E '^  (LOAD|NULL)'; }
  _before=$(_segs "$OUT")
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
    -DCAPSTONE_DOMREQ_DATA=$_need -DCAPSTONE_DOMREQ_STACK=$DOMAIN_STACK \
    -c "$SCRIPT_DIR/../domreq.S" -o "$OBJ_DIR/domreq.o"
  "$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT" \
    "$OBJ_DIR/start.o" "$OBJ_DIR/main.o" "$OBJ_DIR/gct.o" "$OBJ_DIR/domreq.o"
  _after=$(_segs "$OUT")
  [[ "$_before" == "$_after" ]] || {
    echo "FAIL: declaring a requirement changed the program headers -- the section is" >&2
    echo "      not behaving as non-alloc and the image is no longer the one measured." >&2
    exit 1; }
  echo "declared: dom_data >= $_need (carve $_carve + stack $DOMAIN_STACK)"
fi

# static gate: silicon ABI = no cjalr, no gp cincoffset/scc-for-globals; globals
# reached via ldc gp[i] (the compiler emits `ldc rd, imm(gp)`).
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT")
NCJALR=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)
NLDCGP=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)
# The cap-table has TWO addressing forms and the gate must accept both. `ldc rd, imm(gp)` only
# reaches index 127: the immediate is 12-bit signed and entries are 16 bytes, so byte offset
# 2048 is the first one it cannot encode. Above that the compiler emits
# lui/addi/`cincoffset rd, gp, rs`/ldc instead.
#
# Counting only the immediate form made the gate FAIL a perfectly good image whose globals all
# sit above the boundary -- a padded rung with 170 cap-table entries has ZERO `imm(gp)` loads
# and 13 `cincoffset ..., gp, ...`, and was rejected as having "no global access" when in fact
# it accesses globals exclusively by the path we most want to test.
NCINCGP=$(grep -cE 'cincoffset[[:space:]]+[a-z0-9]+,[[:space:]]*gp,' <<<"$DIS" || true)
echo "static: cjalr=$NCJALR ldc-gp=$NLDCGP cincoffset-gp=$NCINCGP"
[[ "$NCJALR" == "0" ]] || { echo "FAIL: cjalr present (not gp-free)" >&2; exit 1; }
[[ "$NLDCGP" -ge 1 || "$NCINCGP" -ge 1 ]] || { echo "FAIL: no gp[i] global access found (neither ldc imm(gp) nor cincoffset off gp)" >&2; exit 1; }
echo "Built $OUT"
