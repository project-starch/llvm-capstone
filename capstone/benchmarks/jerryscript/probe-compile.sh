#!/usr/bin/env bash
# Feasibility probe: does a candidate runtime compile for the Capstone domain?
#
#   ./probe-compile.sh [<source-root>] [<subdir>]
#
# Compiles every .c under <source-root>/<subdir> with the domain's own flags and
# classifies the failures BY KIND rather than by file. Ten minutes of this told us
# the whole shape of the JerryScript port before a line was written: 66 files
# compiled, 132 wanted two headers, one wanted a libm function, and exactly ONE was
# a compiler crash. Deciding without it is how MicroPython spent a day treating a
# probe artefact as a blocker.
#
# It reports, it does not fix. Exit 0 iff everything compiled.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
SRC=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/jerryscript}
SUB=${2:-jerry-core}
D=capstone/benchmarks/jerryscript
CLANG=llvm/cmake-build-debug/bin/clang
OUT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/probe-$(basename "$SRC")
[[ -x $CLANG ]] || { echo "no clang at $CLANG"; exit 1; }
[[ -d $SRC/$SUB ]] || { echo "no sources at $SRC/$SUB"; exit 1; }

# The domain's flags, not a convenient approximation of them. An omitted -mattr is
# how a probe reports a blocker that the real build does not have.
FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -nostdlibinc -fno-builtin -fno-optimize-sibling-calls
       -fno-jump-tables -std=c99 -O0 -w
       # C-20: __builtin_ctz crashes the backend. patches/0001 guards the one use
       # behind this, so it is a workaround with a switch rather than a deletion.
       -DJERRY_NO_BUILTIN_CTZ=1)
INC=(-I"$PWD/$D/adapted/include" -I"$PWD/$D/port")
while IFS= read -r d; do INC+=(-I"$d"); done < <(find "$SRC/$SUB" -type d | sort)

mkdir -p "$OUT/obj"; : > "$OUT/fails.txt"
ok=0; fail=0
while IFS= read -r f; do
  o="$OUT/obj/$(echo "${f#$SRC/}" | tr / _).o"
  rm -f "$o"                       # a stale object from an earlier arm reads as a pass
  if err=$("$CLANG" "${FLAGS[@]}" "${INC[@]}" -c "$f" -o "$o" 2>&1); then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    { echo "=== ${f#$SRC/}"; echo "$err" | grep -E "error:|Assertion" | head -3; } >> "$OUT/fails.txt"
  fi
done < <(find "$SRC/$SUB" -name '*.c' | sort)

echo "  compiled $ok    failed $fail    (of $((ok+fail)))"
if (( fail )); then
  echo
  echo "  failures by KIND:"
  grep -E "error:|Assertion" "$OUT/fails.txt" \
    | sed -E "s/.*(error: |llvm.*Assertion )//" | sed -E "s/'[^']*'/'X'/g" \
    | cut -c1-72 | sort | uniq -c | sort -rn | sed 's/^/   /'
  echo
  echo "  missing headers, and how many files each blocks:"
  grep -E "file not found" "$OUT/fails.txt" | grep -oE "'[a-zA-Z0-9_/.]+\.h'" \
    | sort | uniq -c | sort -rn | sed 's/^/   /'
  echo
  echo "  full log: $OUT/fails.txt"
fi
exit $(( fail ? 1 : 0 ))
