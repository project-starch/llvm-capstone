#!/usr/bin/env bash
# C-50: with -g at -O1+, Assignment Tracking Analysis asserts on a local whose
# address escapes ("The offset bit width does not match the DL specification").
#
#   ./run.sh              reproduce; prints VERDICT (exit 1 = PRESENT)
#   ./run.sh workaround   show that -fexperimental-assignment-tracking=disabled avoids it
#
# CLANG=... overrides the compiler (default llvm/cmake-build-debug/bin/clang).
# No board, no QEMU, no build of anything but these two files.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C50-assignment-tracking-index-width
CLANG=${CLANG:-llvm/cmake-build-debug/bin/clang}
[[ -x $CLANG ]] || { echo "no clang at $CLANG (set CLANG=)"; exit 2; }
F=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -c)
O=${TMPDIR:-/tmp}/c50.$$.o
SIG='offset bit width does not match'

arm() {  # $1 label, $2 source, rest = flags; prints one line, returns 0 if it compiled
  local label=$1 src=$2; shift 2
  rm -f "$O"                       # a stale object from an earlier arm reads as a pass
  printf "  %-38s " "$label"
  if out=$("$CLANG" "${F[@]}" "$@" "$D/src/$src" -o "$O" 2>&1); then
    echo "compiles"; return 0
  fi
  if grep -q "$SIG" <<<"$out"; then echo "ASSERT ($SIG)"; else
    echo "FAILS, other: $(grep -m1 -E 'error|Assertion' <<<"$out" | cut -c1-70)"; fi
  return 1
}

if [[ ${1:-} == workaround ]]; then
  arm "escaping-local  -g -O1" escaping-local.c -g -O1
  arm "escaping-local  -g -O1, AT disabled" escaping-local.c -g -O1 \
      -Xclang -fexperimental-assignment-tracking=disabled
  exit 0
fi

# The three controls say the instrument works before the verdict is read:
# the same file must compile without -g and at -O0, and a local that SROA
# removes must compile with -g -O1.
arm "control: escaping-local -O1 (no -g)" escaping-local.c -O1 || { echo "  VERDICT: CONTROL FAILED"; exit 2; }
arm "control: escaping-local -g -O0" escaping-local.c -g -O0 || { echo "  VERDICT: CONTROL FAILED"; exit 2; }
arm "control: promoted-local -g -O1" promoted-local.c -g -O1 || { echo "  VERDICT: CONTROL FAILED"; exit 2; }
if arm "escaping-local -g -O1" escaping-local.c -g -O1; then
  echo "  VERDICT: C-50 ABSENT"; exit 0
fi
echo "  VERDICT: C-50 PRESENT"; exit 1
