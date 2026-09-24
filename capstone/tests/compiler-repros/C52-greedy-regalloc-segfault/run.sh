#!/usr/bin/env bash
# C-52: the Greedy register allocator segfaults in SplitEditor on CPython's
# compiler_visit_stmt (reduced: 208 instructions, 53 blocks).
#
#   ./run.sh      control (basic allocator) then the Greedy arm; VERDICT
#                 (exit 1 = PRESENT, 2 = control failed)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
#
# -Xclang -disable-llvm-passes is LOAD-BEARING: clang -O1 on a .ll re-runs the
# IR optimizer, which reshapes this function and the crash disappears -- a run
# without it reports ABSENT on a compiler that still has the defect.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C52-greedy-regalloc-segfault
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
   -Xclang -target-feature -Xclang +a -O1 -Xclang -disable-llvm-passes -c)
O=${TMPDIR:-/tmp}/c52.$$.o
IN=$D/src/compiler_visit_stmt.reduced.ll

rm -f "$O"
if ! "$CLANG" "${F[@]}" -mllvm -regalloc=basic "$IN" -o "$O" >/dev/null 2>&1; then
  echo "  control (regalloc=basic): FAILS -- the input or flags are not what this folder claims"
  echo "  VERDICT: CONTROL FAILED"; exit 2
fi
echo "  control (regalloc=basic): compiles"

rm -f "$O"
out=$("$CLANG" "${F[@]}" "$IN" -o "$O" 2>&1); rc=$?
if [[ $rc -eq 0 ]]; then
  echo "  greedy (default):         compiles"
  echo "  VERDICT: C-52 ABSENT"; exit 0
fi
if grep -q "Running pass 'Greedy Register Allocator'" <<<"$out" && grep -q "exit code 139" <<<"$out"; then
  echo "  greedy (default):         SIGSEGV in the Greedy Register Allocator"
  echo "  VERDICT: C-52 PRESENT"; exit 1
fi
echo "  greedy (default):         fails, but not with this signature:"
grep -m2 -E "error|Assertion|Running pass" <<<"$out" | sed 's/^/    /'
echo "  VERDICT: DIFFERENT FAILURE -- not C-52"; exit 2
