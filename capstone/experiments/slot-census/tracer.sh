# Sourced by the runners: build the measuring tool and its controls into $TOOL and run the controls into $OUT/control.
# Needs HERE (this directory), OUT and TOOL. A control that does not behave as expected exits 75, before any
# measurement runs.
#
# SLOTCENSUS_MEASURE picks what a pass measures, with the same runners and the same workloads:
#   census        (default) the slotcensus.c tracer, preloaded: allocations, frees, peak live objects
#   instructions  the slotins.c DynamoRIO client: instructions and allocations per process from the same run
#                 (allocations per instruction), in a pass of its own, without the tracer
# A runner launches every measured program through measured() (or, where a harness runs the program, through the
# prefix measure_prefix prints), so both modes run exactly the same commands.

MEASURE=${SLOTCENSUS_MEASURE:-census}
DYNAMORIO=11.3.0
CACHE=${CACHE:-${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}}

build_tracer() {
  mkdir -p "$TOOL" "$OUT/control"
  [ "$MEASURE" = census ] || [ "$MEASURE" = instructions ] || { echo "SLOTCENSUS_MEASURE=$MEASURE?" >&2; exit 2; }
  if [ "$MEASURE" = instructions ]; then build_slotins; return; fi
  gcc -O2 -fPIC -shared -pthread -o "$TOOL/slotcensus.so" "$HERE/slotcensus.c" -ldl || exit 1
  gcc -O2 -fno-builtin -o "$TOOL/control" "$HERE/control.c" || exit 1   # see control.c
  g++ -O2 -o "$TOOL/control_cxx" "$HERE/control_cxx.cc" || exit 1
}

control() {   # name, binary, preload (1/0), want (pass/fail)
  local name=$1 bin=$2 pre=$3 want=$4 rc
  rm -f "$OUT/control/$name.jsonl"
  if [ "$pre" = 1 ]; then
    SLOTCENSUS_OUT=$OUT/control/$name.jsonl LD_PRELOAD=$TOOL/slotcensus.so "$bin" > "$OUT/control/$name.expect"
  else
    SLOTCENSUS_OUT=$OUT/control/$name.jsonl "$bin" > "$OUT/control/$name.expect"
  fi
  python3 "$HERE/check-control.py" "$OUT/control/$name.expect" "$OUT/control/$name.jsonl" > "$OUT/control/$name.check"
  rc=$?
  if { [ "$want" = pass ] && [ $rc = 0 ]; } || { [ "$want" = fail ] && [ $rc = 1 ]; }; then
    echo "control $name: as expected ($want)"
  else
    echo "control $name: NOT as expected (want $want, checker rc $rc)"; cat "$OUT/control/$name.check"; exit 75
  fi
}

# the C control (with its fork child) and the C++ control must pass under the tracer; the C control without the
# tracer must fail
run_controls() {
  if [ "$MEASURE" = instructions ]; then slotins_controls; return; fi
  control c "$TOOL/control" 1 pass
  control cxx "$TOOL/control_cxx" 1 pass
  control negative "$TOOL/control" 0 fail
}

provenance_tracer() {
  echo "date $(date -u +%FT%TZ)"
  echo "gcc $(gcc -dumpfullversion) g++ $(g++ -dumpfullversion)"
  echo "glibc $(ldd --version | head -1)"
  echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  if [ "$MEASURE" = instructions ]; then
    echo "measure instructions: DynamoRIO $DYNAMORIO, slotins client"
    (cd "$HERE" && sha256sum slotins.c insctl.c) && (cd "$TOOL" && sha256sum slotins/libslotins.so)
  else
    (cd "$HERE" && sha256sum slotcensus.c control.c control_cxx.cc) && (cd "$TOOL" && sha256sum slotcensus.so)
  fi
}

# --- instructions mode --------------------------------------------------------------------------------------------

build_slotins() {   # DynamoRIO $DYNAMORIO (its stable release) and the slotins client, built against its SDK
  local T=DynamoRIO-Linux-$DYNAMORIO.tar.gz
  [ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" \
    "https://github.com/DynamoRIO/dynamorio/releases/download/release_$DYNAMORIO-1/$T" || exit 1
  mkdir -p "$TOOL/dynamorio" && tar xzf "$CACHE/$T" -C "$TOOL/dynamorio" --strip-components=1 || exit 1
  mkdir -p "$TOOL/slotins-src" && cp "$HERE/slotins.c" "$TOOL/slotins-src/" &&
    cp "$HERE/slotins.cmake" "$TOOL/slotins-src/CMakeLists.txt" || exit 1
  cmake -S "$TOOL/slotins-src" -B "$TOOL/slotins" -DDynamoRIO_DIR="$TOOL/dynamorio/cmake" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > "$OUT/control/slotins-build.log" 2>&1 &&
    cmake --build "$TOOL/slotins" >> "$OUT/control/slotins-build.log" 2>&1 \
    || { echo "slotins: build failed" >&2; exit 1; }
  gcc -O2 -pthread -o "$TOOL/insctl" "$HERE/insctl.c" || exit 1
  gcc -O2 -fno-builtin -o "$TOOL/control" "$HERE/control.c" || exit 1   # see control.c
  g++ -O2 -o "$TOOL/control_cxx" "$HERE/control_cxx.cc" || exit 1
  DRRUN=$TOOL/dynamorio/bin64/drrun SLOTINS=$TOOL/slotins/libslotins.so
}

# Known instruction counts: insctl's loop is 2 instructions per iteration, so a run with 10^8 iterations must count
# 2x10^8 more than a run with none: one thread, four threads (8x10^8, two of them still alive at exit), and a fork
# (the parent 4x10^8, its thread's and its own, the child 4x10^8 in its own, forked line, none of the thread's). Tolerance: 1,000 instructions (parsing the longer number). Each check
# is a difference between two runs, so a client that counts nothing, or counts something fixed, fails it.
slotins_controls() {
  local C=$OUT/control m n
  for m in single threads fork; do
    for n in 0 100000000; do
      rm -f "$C/ins-$m-$n.jsonl"
      "$DRRUN" -c "$SLOTINS" "$C/ins-$m-$n.jsonl" -- "$TOOL/insctl" "$n" $([ $m = single ] || echo $m) \
        > /dev/null 2>&1
    done
  done
  python3 - "$C" > "$C/ins.check" <<'PY'
import json, os, sys
C = sys.argv[1]
def rows(m, n):
    path = f"{C}/ins-{m}-{n}.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []
def count(m, n, forked):
    r = [x["instructions"] for x in rows(m, n) if x["forked"] == forked]
    return r[0] if len(r) == 1 else None
ok = True
for m, forked, want in (("single", 0, 2 * 10**8), ("threads", 0, 8 * 10**8), ("fork", 1, 4 * 10**8),
                        ("fork", 0, 4 * 10**8)):
    a, b = count(m, 0, forked), count(m, 100000000, forked)
    good = a is not None and b is not None and abs((b - a) - want) <= 1000
    delta = None if a is None or b is None else b - a
    print(f"{'PASS' if good else 'FAIL'} {m} forked={forked}: {a} -> {b}, delta {delta}, want {want}")
    ok &= good
print("CONTROL PASS" if ok else "CONTROL FAIL")
sys.exit(0 if ok else 1)
PY
  if [ $? = 0 ]; then echo "control instructions: as expected (pass)"
  else echo "control instructions: NOT as expected"; cat "$C/ins.check"; exit 75; fi
  # the allocation counts of the same run: the census's own controls, under DynamoRIO, checked the same way
  local name bin
  for name in c cxx; do
    bin=$TOOL/control; [ $name = cxx ] && bin=$TOOL/control_cxx
    rm -f "$C/$name.jsonl"
    "$DRRUN" -c "$SLOTINS" "$C/$name.jsonl" -- "$bin" > "$C/$name.expect" 2> /dev/null
    if python3 "$HERE/check-control.py" "$C/$name.expect" "$C/$name.jsonl" \
         allocs,frees,realloc_moved,realloc_inplace,forked > "$C/$name.check"; then
      echo "control $name (allocations under DynamoRIO): as expected (pass)"
    else
      echo "control $name (allocations under DynamoRIO): NOT as expected"; cat "$C/$name.check"; exit 75
    fi
  done
}

# measured OUT_FILE command [args...]: run one measured program in the pass's mode. DynamoRIO starts binaries only,
# so a #! script (pyperformance's entry point, apachectl) starts its interpreter, with the script as its argument.
measured() {
  local out=$1; shift
  if [ "$MEASURE" = instructions ]; then
    local prog interp
    prog=$(command -v -- "$1") || prog=$1
    if [ "$(head -c 2 "$prog" 2>/dev/null)" = "#!" ]; then
      read -r interp < "$prog"; interp=${interp#\#!}
      shift
      # shellcheck disable=SC2086   # the #! line is the interpreter and its optional argument, split as the kernel does
      set -- $interp "$prog" "$@"
    fi
    "$DRRUN" -c "$SLOTINS" "$out" -- "$@"
  else
    SLOTCENSUS_OUT=$out LD_PRELOAD=$TOOL/slotcensus.so "$@"
  fi
}
