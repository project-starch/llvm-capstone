#!/usr/bin/env bash
# C-22: the condition is dropped from a select with an all-ones i128 arm.
# No board, no QEMU. Exit 1 if the miscompile is present.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C22-i128-or-sext-folds-to-constant
ASM=$(llvm/cmake-build-debug/bin/llc -mtriple=capstone64 -mattr=+m < "$D/src/sel.ll" 2>&1)
echo "$ASM" | sed -n '/mixed_sign_arms:/,/cjalr/p' | sed 's/^/  /'
if echo "$ASM" | sed -n '/mixed_sign_arms:/,/cjalr/p' | grep -qE "beqz|bnez|snez|sext"; then
  echo "  VERDICT: C-22 ABSENT -- the condition is still there"
  exit 0
fi
echo "  VERDICT: C-22 PRESENT -- no branch and no condition; the arm -1 is unreachable"
exit 1
