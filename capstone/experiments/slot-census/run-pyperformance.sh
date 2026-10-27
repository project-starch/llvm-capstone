#!/bin/bash
# Slot census of CPython 3.13 under pyperformance (the default benchmark set, --fast), once with CPython's own
# small-object allocator (pymalloc, the default: objects up to 512 bytes come from 256 KiB arenas the tracer sees as
# one object each) and once with PYTHONMALLOC=malloc (every object is a malloc). The pair measures what pymalloc
# hides from the allocator below.
#
# usage: run-pyperformance.sh OUT_DIR [BENCHMARK...]   (default: every benchmark of pyperformance's default set)
# Downloads CPython $VERSION from python.org into $SLOTCENSUS_CACHE (default ~/slot-census/cache) and records its
# sha256; pyperformance $PYPERFORMANCE and its benchmarks' dependencies come from PyPI. One cell per arm and
# benchmark, py-<arm>-<benchmark>; its measured processes are pyperf's workers (command lines with --worker), the
# processes that run the benchmark. A cell is valid when pyperformance exits 0 and writes its result file.
# An extra control runs a script that holds 200,000 small objects: under PYTHONMALLOC=malloc its peak must reach
# 200,000, under pymalloc it must stay below 20,000, or the pass exits 75.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
VERSION=3.13.7
PYPERFORMANCE=1.14.0
JOBS=${SLOTCENSUS_PY_JOBS:-16}
. "$HERE/tracer.sh"
start_pass

T=Python-$VERSION.tgz
[ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" "https://www.python.org/ftp/python/$VERSION/$T" || exit 1
tar xzf "$CACHE/$T" -C "$TOOL" || exit 1
(cd "$TOOL/Python-$VERSION" && ./configure --prefix="$TOOL/python" > "$OUT/logs/build.log" 2>&1 &&
 make -j32 >> "$OUT/logs/build.log" 2>&1 && make install >> "$OUT/logs/build.log" 2>&1) \
  || { echo "run-pyperformance: CPython build failed" >&2; exit 1; }
PY=$TOOL/python/bin/python3
"$PY" -m venv "$TOOL/pp" && "$TOOL/pp/bin/pip" -q install "pyperformance==$PYPERFORMANCE" > "$OUT/logs/pip.log" 2>&1 \
  || { echo "run-pyperformance: pyperformance install failed" >&2; exit 1; }
PP=$TOOL/pp/bin/pyperformance

# the arm control (census passes): PYTHONMALLOC must reach the interpreter, and pymalloc must hide small objects
ARMS=(pymalloc malloc)
[ "$MEASURE" = instructions ] && ARMS=(malloc)   # the arm allocations per instruction is drawn for
cat > "$TOOL/hold.py" <<'EOF'
held = [object() for _ in range(200000)]
EOF
[ "$MEASURE" = census ] && for arm in malloc pymalloc; do
  rm -f "$OUT/control/py-$arm.jsonl"
  SLOTCENSUS_OUT=$OUT/control/py-$arm.jsonl LD_PRELOAD=$TOOL/slotcensus.so PYTHONMALLOC=$arm "$PY" "$TOOL/hold.py"
done
[ "$MEASURE" = census ] && { python3 - "$OUT/control" <<'EOF' || exit 75; }
import json, sys
peak = {a: max((json.loads(l)["peak_live"] for l in open(f"{sys.argv[1]}/py-{a}.jsonl")), default=0)
        for a in ("malloc", "pymalloc")}
ok = peak["malloc"] >= 200000 and peak["pymalloc"] < 20000
print(f"control python arms: malloc {peak['malloc']}, pymalloc {peak['pymalloc']}: "
      f"{'as expected' if ok else 'NOT as expected'}")
sys.exit(0 if ok else 1)
EOF

BENCH=("$@")
if [ ${#BENCH[@]} = 0 ]; then
  mapfile -t BENCH < <("$PP" list -b default 2>/dev/null | sed -n 's/^- //p')
fi
[ ${#BENCH[@]} -gt 0 ] || { echo "run-pyperformance: no benchmarks listed" >&2; exit 1; }
# one virtual environment with every benchmark's dependencies, created before anything is traced (pyperformance
# keeps it under the working directory, so every run starts from $OUT)
cd "$OUT" || exit 1
"$PP" venv create --python="$PY" -b default > "$OUT/logs/venv.log" 2>&1 \
  || { echo "run-pyperformance: venv create failed" >&2; exit 1; }
# Instruction passes start only pyperf's workers under DynamoRIO: the venv's python3 becomes a wrapper that runs a
# command line with --worker under it and everything else (pyperformance, pip, the benchmark's driver) natively, as
# the venv's python, so the workers the driver spawns come back through the wrapper. Under DynamoRIO the whole
# driver followed every child, and 16 cells ran 1 h 43 min without finishing one (2026-10-26).
if [ "$MEASURE" = instructions ]; then
  VB=$(echo "$OUT"/venv/*/bin)
  [ -L "$VB/python3" ] && [ ! -e "$VB/python3-real" ] || { echo "run-pyperformance: venv $VB?" >&2; exit 1; }
  mv "$VB/python3" "$VB/python3-real"
  cat > "$VB/python3" <<EOF
#!/bin/bash
for a; do
  [ "\$a" = --worker ] && exec "$DRRUN" -c "$SLOTINS" "\$SLOTCENSUS_OUT" -- "$VB/python3-real" "\$@"
done
exec -a "\$0" "$VB/python3-real" "\$@"
EOF
  chmod 755 "$VB/python3"
  # the wrapper's python must still be the venv's: sys.prefix the venv, sys.executable the name it was started as
  for x in "$VB/python" "$VB/python3"; do
    [ "$("$x" -c 'import sys; print(sys.prefix, sys.executable)')" = "${VB%/bin} $x" ] \
      || { echo "run-pyperformance: $x is not the venv's python" >&2; exit 1; }
  done
fi
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "$T")
  echo "python $("$PY" -VV)"
  echo "build ./configure (defaults) && make && make install"
  echo "pyperformance $("$PP" --version 2>&1)"
  echo "run pyperformance run --fast -b <benchmark>, arms PYTHONMALLOC=pymalloc and PYTHONMALLOC=malloc"
  echo "benchmarks ${#BENCH[@]}: ${BENCH[*]}"
  [ "$MEASURE" = instructions ] && echo "instructions: pyperf's workers (--worker) under DynamoRIO, the rest native"
} > "$OUT/provenance.txt"

run_cell() {   # arm benchmark
  local arm=$1 b=$2 cell=py-$1-$2 t rc v
  t=$(date +%s)
  local launch=(measured "$OUT/$cell.jsonl")
  [ "$MEASURE" = instructions ] && launch=(env SLOTCENSUS_OUT="$OUT/$cell.jsonl")   # the workers' wrapper reads it
  PYTHONMALLOC=$arm "${launch[@]}" \
    "$PP" run --python="$PY" --fast -b "$b" -o "$OUT/logs/$cell.json" \
    --inherit-environ PYTHONMALLOC,LD_PRELOAD,SLOTCENSUS_OUT > "$OUT/logs/$cell.log" 2>&1
  rc=$?
  v=INVALID; [ $rc = 0 ] && [ -s "$OUT/logs/$cell.json" ] && v=valid
  printf '%s\tre:--worker\t%s\trc=%s %ss\n' "$cell" "$v" "$rc" "$(( $(date +%s) - t ))" >> "$OUT/cells-$arm.tsv"
}
# $JOBS cells at a time (the counts do not depend on timing; one cell after another takes hours)
for b in "${BENCH[@]}"; do
  for arm in "${ARMS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run_cell "$arm" "$b" &
  done
done
wait
for arm in "${ARMS[@]}"; do cat "$OUT/cells-$arm.tsv"; done > "$OUT/cells.tsv"
echo CENSUS_DONE >> "$OUT/cells.tsv"
