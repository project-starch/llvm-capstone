# shellcheck shell=bash
# Sourced by the runners, after they set HERE (this directory) and OUT (the pass directory, which must not exist).
# start_pass makes the pass directory, builds the measuring tool and its controls into $TOOL and runs the controls
# into $OUT/control: a control that does not behave as expected exits 75, before any measurement runs.
#
# SLOTCENSUS_MEASURE picks what a pass measures, with the same runners and the same workloads:
#   census        (default) the slotcensus.c tracer, preloaded: allocations, frees, peak live objects
#   instructions  the slotins.c DynamoRIO client: instructions and allocations per process from the same run
#                 (allocations per instruction), in a pass of its own, without the tracer
# A runner launches every measured program through measured() (or, where a harness runs the program, through the
# prefix measure_prefix prints), so both modes run exactly the same commands.

MEASURE=${SLOTCENSUS_MEASURE:-census}
DYNAMORIO=11.3.0
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}   # downloads, kept across passes

# start_pass: a fresh pass directory (OUT made absolute, OUT/logs), TOOL, the tools built and every control passed
start_pass() {
  local runner=${0##*/}
  [ -e "$OUT" ] && { echo "${runner%.sh}: $OUT exists" >&2; exit 2; }
  mkdir -p "$OUT/logs" "$CACHE" || exit 1
  OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
  build_tracer
  run_controls
}

# record_cell CELL SELECTOR VERDICT NOTE: one row of $OUT/cells.tsv (see summarize.py), echoed
record_cell() {
  printf '%s\t%s\t%s\t%s\n' "$@" | tee -a "$OUT/cells.tsv"
}

build_controls() {   # the census's controls, with known allocation counts
  gcc -O2 -fno-builtin -o "$TOOL/control" "$HERE/control.c" || exit 1   # see control.c
  g++ -O2 -o "$TOOL/control_cxx" "$HERE/control_cxx.cc" || exit 1
}

build_tracer() {
  mkdir -p "$TOOL" "$OUT/control"
  [ "$MEASURE" = census ] || [ "$MEASURE" = instructions ] || { echo "SLOTCENSUS_MEASURE=$MEASURE?" >&2; exit 2; }
  if [ "$MEASURE" = instructions ]; then build_slotins; return; fi
  gcc -O2 -fPIC -shared -pthread -o "$TOOL/slotcensus.so" "$HERE/slotcensus.c" -ldl || exit 1
  build_controls
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
    (cd "$HERE" && sha256sum slotins.c census.h insctl.c) && (cd "$TOOL" && sha256sum slotins/libslotins.so)
  else
    (cd "$HERE" && sha256sum slotcensus.c census.h control.c control_cxx.cc) && (cd "$TOOL" && sha256sum slotcensus.so)
  fi
}

# --- instructions mode --------------------------------------------------------------------------------------------

build_slotins() {   # DynamoRIO $DYNAMORIO (its stable release) and the slotins client, built against its SDK
  local T=DynamoRIO-Linux-$DYNAMORIO.tar.gz
  [ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" \
    "https://github.com/DynamoRIO/dynamorio/releases/download/release_$DYNAMORIO-1/$T" || exit 1
  mkdir -p "$TOOL/dynamorio" && tar xzf "$CACHE/$T" -C "$TOOL/dynamorio" --strip-components=1 || exit 1
  mkdir -p "$TOOL/slotins-src" && cp "$HERE/slotins.c" "$HERE/census.h" "$TOOL/slotins-src/" &&
    cp "$HERE/slotins.cmake" "$TOOL/slotins-src/CMakeLists.txt" || exit 1
  cmake -S "$TOOL/slotins-src" -B "$TOOL/slotins" -DDynamoRIO_DIR="$TOOL/dynamorio/cmake" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > "$OUT/control/slotins-build.log" 2>&1 &&
    cmake --build "$TOOL/slotins" >> "$OUT/control/slotins-build.log" 2>&1 \
    || { echo "slotins: build failed" >&2; exit 1; }
  gcc -O2 -pthread -o "$TOOL/insctl" "$HERE/insctl.c" || exit 1
  build_controls
  DRRUN=$TOOL/dynamorio/bin64/drrun SLOTINS=$TOOL/slotins/libslotins.so
}

# Known instruction counts: insctl runs its loop 0 and 10^8 times, alone, in four threads and across a fork, and
# check-instructions.py compares the differences with what the loop must add.
slotins_controls() {
  local C=$OUT/control m n mode
  for m in single threads fork; do
    mode=(); [ $m = single ] || mode=("$m")   # insctl's argument: none for one thread
    for n in 0 100000000; do
      rm -f "$C/ins-$m-$n.jsonl"
      "$DRRUN" -c "$SLOTINS" "$C/ins-$m-$n.jsonl" -- "$TOOL/insctl" "$n" "${mode[@]}" > /dev/null 2>&1
    done
  done
  if python3 "$HERE/check-instructions.py" "$C" > "$C/ins.check"; then echo "control instructions: as expected (pass)"
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
