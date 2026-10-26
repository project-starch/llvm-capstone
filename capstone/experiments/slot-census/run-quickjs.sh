#!/bin/bash
# Slot census of the QuickJS JavaScript engine running the Octane benchmark suite. QuickJS allocates every object,
# string, shape and function through malloc, so its census counts the program's objects.
#
# usage: run-quickjs.sh OUT_DIR [BENCHMARK...]   (default: the 15 Octane benchmarks below)
# Downloads quickjs-$VERSION and quickjs-extras-$VERSION (which carries tests/octane) from bellard.org into
# $SLOTCENSUS_CACHE (default ~/slot-census/cache) and records their sha256; builds qjs with its Makefile's defaults.
# One cell per benchmark, run as Octane's own tests/octane/run.js does with a name filter (one process each).
# A cell is valid when qjs exits 0 and Octane prints its score line, which it does only when no benchmark reported
# an error.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
VERSION=2026-06-04
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}
BENCH=("$@")
[ ${#BENCH[@]} -gt 0 ] || BENCH=(richards deltablue crypto raytrace earley-boyer regexp splay navier-stokes pdfjs
                                  mandreel gbemu code-load box2d zlib typescript)
[ -e "$OUT" ] && { echo "run-quickjs: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT/logs" "$CACHE"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

for a in quickjs-$VERSION.tar.xz quickjs-extras-$VERSION.tar.xz; do
  [ -s "$CACHE/$a" ] || curl -fsSL -o "$CACHE/$a" "https://bellard.org/quickjs/$a" || exit 1
  tar xJf "$CACHE/$a" -C "$TOOL" || exit 1
done
Q=$TOOL/quickjs-$VERSION
make -C "$Q" -j16 qjs > "$OUT/logs/build.log" 2>&1 || { echo "run-quickjs: build failed" >&2; exit 1; }
[ -f "$Q/tests/octane/run.js" ] || { echo "run-quickjs: no tests/octane in the extras" >&2; exit 1; }
{
  provenance_tracer
  (cd "$CACHE" && sha256sum quickjs-$VERSION.tar.xz quickjs-extras-$VERSION.tar.xz)
  echo "quickjs $VERSION, make qjs (Makefile defaults)"
  echo "octane benchmarks ${BENCH[*]}; qjs tests/octane/run.js <name>"
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
cd "$Q" || exit 1
for b in "${BENCH[@]}"; do
  cell=octane-$b
  t=$(date +%s)
  SLOTCENSUS_OUT=$OUT/$cell.jsonl LD_PRELOAD=$TOOL/slotcensus.so ./qjs tests/octane/run.js "$b" > "$OUT/$cell.out" 2>&1
  rc=$?
  v=INVALID; [ $rc = 0 ] && grep -q '^Score (version' "$OUT/$cell.out" && v=valid
  printf '%s\tqjs\t%s\trc=%s %ss\n' "$cell" "$v" "$rc" "$(( $(date +%s) - t ))" | tee -a "$OUT/cells.tsv"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
