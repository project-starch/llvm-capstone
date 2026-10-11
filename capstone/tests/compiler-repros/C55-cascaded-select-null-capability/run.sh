#!/usr/bin/env bash
# C-55: two cascaded selects on a capability with a null operand put the
# physical null register $c0 straight into a PHI; LiveVariables (-O1+) or
# PHIElimination (-O0) asserts. A residual of d5b5de228b38.
#
#   ./run.sh      three controls, then the failing shape and CPython's reduced
#                 function at -O0..-O3; VERDICT (exit 1 = PRESENT, 2 = control failed)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
# -Xclang -disable-llvm-passes keeps the IR as written: the optimizer would
# otherwise fold two selects on one condition into one, and the crash is gone.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C55-cascaded-select-null-capability
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
O=${TMPDIR:-/tmp}/c55.$$.o
arm() {  # $1 file, $2 -O level -> ok | PHI-ASSERT | other
  rm -f "$O"
  if out=$("$CLANG" -target capstone64-unknown-elf -Xclang -disable-llvm-passes "$2" -c "$D/src/$1" -o "$O" 2>&1); then
    echo ok
  elif grep -qE 'getVarInfo: not a virtual register|Machine PHI Operands must all be virtual' <<<"$out"; then
    echo PHI-ASSERT
  else echo other; fi
}
present=0; bad=0
for f in control-one-select-null.ll control-two-selects-nonnull.ll control-two-selects-i64.ll \
         two-selects-null.ll dict___contains__.reduced.ll; do
  line=$(printf "  %-32s" "$f")
  for o in -O0 -O1 -O2 -O3; do
    r=$(arm "$f" $o); line+=$(printf " %s:%-10s" "$o" "$r")
    case $f in control-*) [[ $r == ok ]] || bad=1 ;; *) [[ $r == PHI-ASSERT ]] && present=1 ;; esac
  done
  echo "$line"
done
(( bad )) && { echo "  VERDICT: CONTROL FAILED"; exit 2; }
(( present )) && { echo "  VERDICT: C-55 PRESENT"; exit 1; }
echo "  VERDICT: C-55 ABSENT"; exit 0
