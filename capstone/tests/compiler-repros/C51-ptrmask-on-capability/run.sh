#!/usr/bin/env bash
# C-51: llvm.ptrmask on a capability crashes isel ("Shift amount is not an
# integer type!"). Every 8- or 16-bit atomic reaches it, because AtomicExpand
# aligns the address with ptrmask; 32- and 64-bit atomics do not and compile.
#
#   ./run.sh      ptrmask alone, the operation x width matrix, CPython's PyMutex
#                 shape; VERDICT (exit 1 = PRESENT, 2 = a 32/64-bit control failed)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C51-ptrmask-on-capability
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
# +a: the A extension, as every musl-domain build here passes it.
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
   -Xclang -target-feature -Xclang +a -ffreestanding -O1 -c)
O=${TMPDIR:-/tmp}/c51.$$.o
SIG='Shift amount is not an integer type'

one() {  # prints ok|SHIFT|other, returns 0 only for ok
  rm -f "$O"
  if out=$("$CLANG" "${F[@]}" "$@" -o "$O" 2>&1); then echo ok; return 0; fi
  if grep -q "$SIG" <<<"$out"; then echo SHIFT; else echo other; fi
  return 1
}

present=0; control_bad=0
r=$(one "$D/src/ptrmask.ll"); [[ $r == SHIFT ]] && present=1
printf "  %-22s %s\n" "ptrmask.ll (alone)" "$r"
printf "  %-6s %6s %6s %6s %6s\n" op u8 u16 u32 u64
for op in cas add xchg; do
  line=$(printf "  %-6s" $op)
  for w in 8 16 32 64; do
    r=$(one -DW=$w -DOP_$op "$D/src/atomics.c")
    line+=$(printf " %6s" "$r")
    if (( w < 32 )); then [[ $r == SHIFT ]] && present=1
    else [[ $r == ok ]] || control_bad=1; fi
  done
  echo "$line"
done
printf "  %-22s %s\n" "pymutex.c (CPython)" "$(one "$D/src/pymutex.c")"
printf "  %-22s %s\n" "align-down.c" "$(one "$D/src/align-down.c")"

(( control_bad )) && { echo "  VERDICT: CONTROL FAILED -- a 32/64-bit atomic did not compile"; exit 2; }
(( present )) && { echo "  VERDICT: C-51 PRESENT"; exit 1; }
echo "  VERDICT: C-51 ABSENT"; exit 0
