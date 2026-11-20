#!/bin/bash
# Off-path identity for CSC_SUBLET: with CSC_SUBLET undefined, each emulator assembles to the same bytes as at the base
# commit, for every define combination in use. Exit 0 = identical everywhere; 1 = a difference (printed); 2 = no data.
set -u
# usage: offpath-ident.sh <capstone-sbi worktree> [base commit, default 22a907f]
W=${1:?usage: offpath-ident.sh <capstone-sbi worktree> [base]}; BASE=${2:-22a907f}
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
[ -x "$BIN/clang" ] || { echo "NO DATA: no clang at $BIN (set CAPSTONE_LLVM_BIN)"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
COMBOS=("-DCAPSTONE_CSCHECK_EMU -DCSC_IDS55 -DCSC_TABLE_TREE -DCAPSTONE_VP2_CSMINT -DCAPSTONE_TARGET_FPGA"
        "-DCAPSTONE_CSCHECK_EMU -DCSC_IDS55 -DCAPSTONE_VP2_CSMINT -DCAPSTONE_TARGET_FPGA"
        "-DCAPSTONE_CSCHECK_EMU -DCAPSTONE_VP2_CSMINT -DCAPSTONE_TARGET_FPGA")
rc=0; n=0
for f in capstone_cscheck.S capstone_csmint.S; do
  git -C "$W" show "$BASE:$f" > "$T/base-$f" || { echo "NO DATA: $BASE:$f"; exit 2; }
  for i in "${!COMBOS[@]}"; do
    for side in base work; do
      src=$T/base-$f; [ $side = work ] && src=$W/$f
      "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -c ${COMBOS[$i]} -x assembler-with-cpp "$src" -o "$T/$side-$i-$f.o" 2> "$T/$side-$i-$f.err" \
        || { echo "ASSEMBLY FAILED ($side, combo $i, $f):"; head -5 "$T/$side-$i-$f.err"; exit 2; }
      "$BIN/llvm-objdump" -s "$T/$side-$i-$f.o" | tail -n +3 > "$T/$side-$i-$f.sec"
      "$BIN/llvm-nm" "$T/$side-$i-$f.o" > "$T/$side-$i-$f.sym"
    done
    n=$((n+1))
    if cmp -s "$T/base-$i-$f.sec" "$T/work-$i-$f.sec" && cmp -s "$T/base-$i-$f.sym" "$T/work-$i-$f.sym"; then
      echo "IDENTICAL $f combo $i ($(wc -l < "$T/base-$i-$f.sec") section lines)"
    else echo "DIFFERS   $f combo $i"; diff "$T/base-$i-$f.sec" "$T/work-$i-$f.sec" | head -6; rc=1; fi
  done
done
[ $n -eq 6 ] || { echo "NO DATA: $n of 6 comparisons"; exit 2; }
exit $rc
