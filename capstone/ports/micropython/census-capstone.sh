#!/usr/bin/env bash
# Compile every MicroPython core file for the Capstone silicon ABI and report
# how many make it through, with one line per failure.
#
# This is a MEASUREMENT, not a build: each file is compiled on its own, which a
# real domain never is (the gp-captable ABI numbers globals per module, so a
# domain is one TU). It exists to turn "the port is hard" into a number, and to
# tell a toolchain gap apart from a source problem.
#
# A file that compiles is NOT a file that is correct: the C front end accepts
# every tag manipulation, so anything that reads or builds an object word
# lowered to something whether or not it preserves a capability tag. See
# agent-handoff/plans/micropython-domain-compilation.md.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}

MPY_SRC_DIR=${MPY_SRC_DIR:-$CAPSTONE_TMP_ROOT/micropython}
CLANG=${CAPSTONE_CLANG:-$REPO_ROOT/llvm/cmake-build-debug/bin/clang}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/micropython-census}
JOBS=${JOBS:-8}

[[ -d $MPY_SRC_DIR/py ]] || { echo "no MicroPython at $MPY_SRC_DIR -- run fetch-micropython.sh" >&2; exit 1; }
[[ -x $CLANG ]] || { echo "no clang at $CLANG -- set CAPSTONE_CLANG" >&2; exit 1; }

# Assertions matter here. Without them APInt::getSExtValue returns the low 64
# bits of a capability-width constant silently, so a NoAsserts clang miscompiles
# where this one stops.
if ! "$CLANG" --version | grep -qi "assertions\|debug" && [[ -z ${CENSUS_ALLOW_NOASSERTS:-} ]]; then
  echo "== WARNING: cannot confirm this clang has assertions enabled" >&2
fi

# The generated qstr/module headers come from a stock RV64 build of ports/minimal.
GEN_DIR=${GEN_DIR:-$MPY_SRC_DIR/ports/minimal/build}
[[ -f $GEN_DIR/genhdr/qstrdefs.generated.h ]] || {
  echo "no generated headers at $GEN_DIR -- build ports/minimal once with a stock toolchain:" >&2
  echo "  make -C $MPY_SRC_DIR/ports/minimal CROSS_COMPILE=riscv64-linux-" >&2; exit 1; }

SILICON=(-mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         -mllvm -capstone-merge-string-constants=true
         -DCAPSTONE_GP_CAPTABLE_ABI=1)
# -nostdlibinc is load-bearing, not tidiness. Without it clang still searches
# /usr/include for a bare-metal triple, so `#include <string.h>` resolves to the
# HOST glibc header and the freestanding headers below are never read. That was
# true here for the first four rounds of this census: removing adapted/string.h
# entirely changed nothing, which is how it was caught. Clang's own stddef /
# stdint / stdarg / limits stay available, which is what freestanding means.
COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding -nostdlibinc -fno-builtin -fno-optimize-sibling-calls
        -fno-jump-tables -std=c99 -O0 -w
        -I"$SCRIPT_DIR/adapted/include" -I"$MPY_SRC_DIR"
        -I"$MPY_SRC_DIR/ports/minimal" -I"$GEN_DIR")

rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR/logs"
cd "$MPY_SRC_DIR"
ls py/*.c | xargs -P "$JOBS" -I{} sh -c '
  f="{}"; b=$(basename "$f" .c)
  "'"$CLANG"'" '"${COMMON[*]}"' '"${SILICON[*]}"' -c "$f" -o /dev/null \
    > "'"$OUT_DIR"'/logs/$b.log" 2>&1
  echo "$? $b" >> "'"$OUT_DIR"'/results.txt"
' 2>/dev/null || true

total=$(wc -l < "$OUT_DIR/results.txt")
ok=$(awk '$1==0' "$OUT_DIR/results.txt" | wc -l)
fail=$(( total - ok ))
echo "== $ok of $total py/ core files compile for the silicon ABI"
[[ $fail -eq 0 ]] && exit 0

echo "== $fail failing:"
while read -r _ b; do
  sig=$(grep -m1 -E "Assertion .* failed|fatal error|LLVM ERROR|error:" "$OUT_DIR/logs/$b.log" | cut -c1-100)
  fn=$(grep -m1 "Running pass .* on function" "$OUT_DIR/logs/$b.log" | sed "s/.*on function '@//; s/'.*//")
  printf "   %-14s %-70s %s\n" "$b" "$sig" "${fn:+fn=$fn}"
done < <(awk '$1!=0' "$OUT_DIR/results.txt" | sort -k2)
exit 1
