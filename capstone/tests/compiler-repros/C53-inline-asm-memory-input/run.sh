#!/usr/bin/env bash
# C-53: an inline-asm "m" INPUT operand crashes isel ("InOperandVal.getValueType()
# == TLI.getPointerTy(...)"), whatever the memory is; "=m" OUTPUT operands compile.
#
#   ./run.sh      the five shapes at -O0 and -O1; VERDICT
#                 (exit 1 = PRESENT, 2 = an output-operand control failed)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C53-inline-asm-memory-input
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -c)
O=${TMPDIR:-/tmp}/c53.$$.o
SIG='InOperandVal.getValueType'
present=0; control_bad=0
for src in output-pointer output-local input-pointer input-local input-global; do
  line=$(printf "  %-16s" "$src")
  for o in -O0 -O1; do
    rm -f "$O"
    if out=$("$CLANG" "${F[@]}" $o "$D/src/$src.c" -o "$O" 2>&1); then r=ok
    elif grep -q "$SIG" <<<"$out"; then r=ASSERT; else r=other; fi
    line+=$(printf " %s:%-6s" "$o" "$r")
    case $src in
      output-*) [[ $r == ok ]] || control_bad=1 ;;
      input-*)  [[ $r == ASSERT ]] && present=1 ;;
    esac
  done
  echo "$line"
done
(( control_bad )) && { echo "  VERDICT: CONTROL FAILED -- an \"=m\" output operand did not compile"; exit 2; }
(( present )) && { echo "  VERDICT: C-53 PRESENT"; exit 1; }
echo "  VERDICT: C-53 ABSENT"; exit 0
