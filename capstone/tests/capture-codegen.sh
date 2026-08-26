#!/usr/bin/env bash
# Capture the code this toolchain generates for a corpus, so two builds can be
# diffed. Run it before and after a codegen change and `diff -r` the two dirs.
#
#   capture-codegen.sh <outdir>
#
# This is the right gate for a lowering change and the reason is worth stating:
# byte-identical output over the corpus is STRONGER evidence than a suite run.
# A suite says "the cases I exercise still pass"; identical assembly says "no
# case can behave differently, because nothing about it changed". It also costs
# minutes rather than the ~85 the nightly takes.
#
# A file that fails to compile records its error text instead of assembly, so a
# crash turning into code (or the reverse) shows up as a diff rather than as a
# missing file nobody notices.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/capstone-test-env.sh"

OUT=${1:-}
[ -n "$OUT" ] || { echo "usage: capture-codegen.sh <outdir>" >&2; exit 2; }
mkdir -p "$OUT"

B=${CAPSTONE_LLVM_BUILD_DIR:-$CAPSTONE_REPO_ROOT/llvm/cmake-build-debug}
LLC=$B/bin/llc
CLANG=${CAPSTONE_CLANG:-$B/bin/clang}
REPO=$CAPSTONE_REPO_ROOT

# .ident carries the git SHA of the compiler that produced the file, so EVERY
# output differs between any two builds. Left in, it buries the real diff: the
# first run of this reported 433 changed files, all of them this one line.
strip_build_metadata() { sed -i '/^\s*\.ident/d' "$1" 2>/dev/null || true; }

n=0
# IR corpus: every Capstone codegen test, at the triple they are written for.
while IFS= read -r f; do
  rel=${f#"$REPO"/}; dst="$OUT/${rel//\//__}.s"
  "$LLC" -mtriple=capstone64 -mattr=+m -o "$dst" "$f" > /dev/null 2>"$dst.err" || {
    # Keep the diagnosis, not the noise: the message, never the stack dump.
    grep -m2 -E "LLVM ERROR|error:|Assertion" "$dst.err" > "$dst" 2>/dev/null
    [ -s "$dst" ] || echo "FAILED (no message)" > "$dst"
  }
  strip_build_metadata "$dst"
  rm -f "$dst.err"; n=$((n + 1))
done < <(find "$REPO/llvm/test/CodeGen/Capstone" -name '*.ll' -type f 2>/dev/null | sort)

# C corpus: the workload sources, at the levels the suites build them.
while IFS= read -r f; do
  rel=${f#"$REPO"/}
  for O in -O0 -O1 -O2; do
    dst="$OUT/${rel//\//__}$O.s"
    "$CLANG" --target=capstone64-unknown-elf "$O" -S -o "$dst" "$f" \
      > /dev/null 2>"$dst.err" || {
      grep -m2 -E "LLVM ERROR|error:|Assertion" "$dst.err" > "$dst" 2>/dev/null
      [ -s "$dst" ] || echo "FAILED (no message)" > "$dst"
    }
    strip_build_metadata "$dst"
    rm -f "$dst.err"; n=$((n + 1))
  done
done < <(find "$REPO/capstone/benchmarks" -name '*.c' -type f 2>/dev/null | sort)

if [ "$n" -eq 0 ]; then
  echo "capture-codegen.sh: corpus is EMPTY -- captured nothing, so a later diff would" >&2
  echo "  compare two empty directories and read as 'no change'." >&2
  exit 2
fi
echo "capture-codegen.sh: $n output(s) in $OUT"
