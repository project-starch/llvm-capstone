#!/usr/bin/env bash
# C-77: a pointer-typed common symbol (`char *x;` under -fcommon) crashes the
# capstone64 backend with llvm_unreachable("Unknown section kind").
#
#   ./run.sh      compiles src/common_ptr.c with -fcommon (the defect) and with
#                 -fno-common (the control, which must compile), and an integer
#                 common symbol (which must compile); VERDICT (exit 1 = PRESENT,
#                 2 = a control failed, 0 = ABSENT)
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C77-pointer-common-symbol
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
   -Xclang -target-feature -Xclang +a -ffreestanding -O2 -c)
T=${TMPDIR:-/tmp}/c77.$$
SIG='Unknown section kind'
trap 'rm -f "$T".*' EXIT

# the control: the same file without common symbols
if ! "$CLANG" "${F[@]}" -fno-common "$D/src/common_ptr.c" -o "$T.ctl.o" 2>"$T.ctl.err"; then
  echo "CONTROL FAILED: -fno-common did not compile"; tail -3 "$T.ctl.err"; exit 2
fi
# the second control: an integer common symbol
printf 'int shared_count;\nint main(void) { return shared_count; }\n' > "$T.int.c"
if ! "$CLANG" "${F[@]}" -fcommon "$T.int.c" -o "$T.int.o" 2>"$T.int.err"; then
  echo "CONTROL FAILED: an integer common symbol did not compile"; tail -3 "$T.int.err"; exit 2
fi
# the defect
if "$CLANG" "${F[@]}" -fcommon "$D/src/common_ptr.c" -o "$T.o" 2>"$T.err"; then
  echo "VERDICT: ABSENT (a pointer common symbol compiles)"; exit 0
fi
if grep -q "$SIG" "$T.err"; then
  echo "VERDICT: PRESENT ($SIG)"; grep -m1 "UNREACHABLE" "$T.err"; exit 1
fi
echo "VERDICT: compile failed for another reason"; tail -5 "$T.err"; exit 2
