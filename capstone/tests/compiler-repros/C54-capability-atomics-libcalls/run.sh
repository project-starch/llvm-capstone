#!/usr/bin/env bash
# C-54: an atomic operation on a POINTER (a 16-byte capability) compiles to a
# call of __atomic_{load,store,compare_exchange,exchange}_16, which nothing in a
# domain provides; the same operations on a long compile inline.
#
#   ./run.sh      each operation's undefined symbols; VERDICT
#                 (exit 1 = PRESENT, 2 = the long control needed a library call)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang);
# llvm-nm is taken from beside it.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C54-capability-atomics-libcalls
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
NM=$(dirname "$CLANG")/llvm-nm
[[ -x $CLANG && -x $NM ]] || { echo "need clang and llvm-nm at $(dirname "$CLANG") (set CLANG=)"; exit 2; }
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
   -Xclang -target-feature -Xclang +a -ffreestanding -O1 -w -c)
O=${TMPDIR:-/tmp}/c54.$$.o
undef() { rm -f "$O"; "$CLANG" "${F[@]}" "$@" -o "$O" >/dev/null 2>&1 || { echo "COMPILE-FAILED"; return; }
          "$NM" -u "$O" | awk '{print $2}' | tr '\n' ' '; }
present=0
for op in load store cas xchg; do
  u=$(undef -DOP_$op "$D/src/pointer-atomics.c")
  printf "  pointer %-6s %s\n" "$op" "${u:-none}"
  [[ $u == *__atomic_*_16* ]] && present=1
done
c=$(undef "$D/src/long-atomics.c")
printf "  control: long   %s\n" "${c:-none}"
[[ -z $c ]] || { echo "  VERDICT: CONTROL FAILED -- long atomics did not compile to inline code ($c)"; exit 2; }
(( present )) && { echo "  VERDICT: C-54 PRESENT"; exit 1; }
echo "  VERDICT: C-54 ABSENT"; exit 0
