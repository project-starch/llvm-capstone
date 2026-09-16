#!/bin/bash
# Build the R1 slots-and-pools harness as a Capstone domain for the board and the emulator, with
# the SQLite silicon build's link recipe (build-sqlite-silicon.sh): the cap-table gp ABI, the
# ladder's entry glue, link-gpfree.ld with the globals offset measured from .text, and a domreq
# declaration. Same host as the SQLite cells (sqlite_host.user / sqlite_host_rr.user, --speedtest1).
#
#   OUT_DIR=<dir> [R1_OPT=-O1] [DOMAIN_BASE_VA=0x10000] [R1_STACK=262144]
#   [R1_EXTRA_DEFS="-DM1_LIVE=4"] bash build-r1-silicon.sh
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO_ROOT/capstone/tests/capstone-test-env.sh" >/dev/null 2>&1 || true
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/r1-silicon}
OBJ_DIR=$OUT_DIR/obj
LADDER=$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder
GPFREE=$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain
OPT=${R1_OPT:--O1}
BASE=${DOMAIN_BASE_VA:-0x10000}
mkdir -p "$OBJ_DIR"
SILICON=(-mllvm -capstone-merge-string-constants=true -mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false -mllvm -capstone-shrink-globals=false
         -DCAPSTONE_GP_CAPTABLE_ABI=1)
COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -fno-builtin
        -I"$REPO_ROOT/capstone/sublet" -I"$REPO_ROOT/capstone/ports/sqlite")
echo "== compile $OPT"
# R1_EXTRA_DEFS: additional -D flags for the harness translation unit, e.g. R1_EXTRA_DEFS="-DM1_LIVE=4".
# Word-split deliberately so several may be given; a define that changes the fixture geometry (M1_LIVE
# sizes the carve) makes a DIFFERENT image with a different hash, which needs its own emulator pass.
read -r -a _r1_extra <<< "${R1_EXTRA_DEFS:-}"
"$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" "${_r1_extra[@]}" "$OPT" -c "$SCRIPT_DIR/r1_slots_pools.c" -o "$OBJ_DIR/r1.o"
link() {  # $1 = globals offset literal, $2 = output
  local lds="$OBJ_DIR/link.ld"
  # the same two substitutions build-nginx-domain.sh makes: the globals offset, and the base VA
  sed -e "s/0x10000 + 0x1000/@BASE@ + @WIN@/" -e "s/0x10000/@BASE@/g" -e "s/@WIN@/$1/" -e "s/@BASE@/$BASE/g" \
      "$GPFREE/link-gpfree.ld" > "$lds"
  grep -q -- "$BASE + $1" "$lds" || { echo "linker-script substitution FAILED (wanted '$BASE + $1')" >&2; exit 1; }
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$CAPSTONE_LD_LLD" -T "$lds" -o "$2" "$OBJ_DIR/start.o" "$OBJ_DIR/r1.o" "${EXTRA_LINK_OBJS[@]}"
}
EXTRA_LINK_OBJS=()
echo "== pass 1: provisional link to measure .text"
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" | python3 -c '
import sys,re
for l in sys.stdin:
    m=re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2),16)); break
else: print(0)')
[[ "$TEXT" -gt 0 ]] || { echo "could not measure .text" >&2; exit 1; }
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 )); [[ $GOFF -lt 65536 ]] && GOFF=65536
printf "   .text = %d bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"
echo "== pass 2: real globals offset"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/r1_slots_pools.dom"
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/r1_slots_pools.dom")
echo "   cjalr=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)  ldc-gp=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)  mcycle=$(grep -c 'mcycle' <<<"$DIS" || true)  minstret=$(grep -c 'minstret' <<<"$DIS" || true)"
NHDR=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/r1_slots_pools.dom" | grep -c "capstone_gp_table" || true)
[[ "$NHDR" -eq 1 ]] || { echo "capstone_gp_table sections: $NHDR (must be 1)" >&2; exit 1; }
# grep -c, not -q: under pipefail a -q that exits early hands readelf a SIGPIPE and the gate fires on a present section
NDESC=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/r1_slots_pools.dom" | grep -c capstone_gp_initdesc || true)
[[ "$NDESC" -ge 1 ]] || { echo "no .capstone_gp_initdesc: the monitor would deliver no gp (ISSUES M-8)" >&2; exit 1; }
R1_STACK=${R1_STACK:-$((256 * 1024))}
CARVE=$(python3 "$LADDER/domdata-budget.py" "$OUT_DIR/r1_slots_pools.dom" --carve)
[[ "$CARVE" -gt 0 ]] || { echo "could not measure the dom_data carve" >&2; exit 1; }
DATA=$(( CARVE + R1_STACK ))
echo "== pass 3: declare dom_data >= $DATA (carve $CARVE + stack $R1_STACK)"
_segs() { "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$OUT_DIR/r1_slots_pools.dom" | grep -E '^\s+LOAD'; }
_before=$(_segs)
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -DCAPSTONE_DOMREQ_DATA=$DATA -DCAPSTONE_DOMREQ_STACK=$R1_STACK \
  -c "$REPO_ROOT/capstone/tests/runtime-qemu/domreq.S" -o "$OBJ_DIR/domreq.o"
EXTRA_LINK_OBJS=("$OBJ_DIR/domreq.o")
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/r1_slots_pools.dom"
[[ "$(_segs)" == "$_before" ]] || { echo "domreq.S moved a loaded byte" >&2; exit 2; }
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/r1_slots_pools.dom" || { echo "the declared budget does not fit" >&2; exit 1; }
sha256sum "$OUT_DIR/r1_slots_pools.dom" | cut -c1-16 | sed 's/^/   sha256\/16 /'
echo "Built $OUT_DIR/r1_slots_pools.dom (base $BASE, $OPT)"
