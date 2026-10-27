# shellcheck shell=bash
# Sourced by the runners, after they set HERE (this directory) and OUT (the pass directory, which must not exist).
# start_pass makes the pass directory, builds the measuring tool and its controls into $TOOL and runs the controls
# into $OUT/control: a control that does not behave as expected exits 75, before any measurement runs.
#
# SLOTCENSUS_MEASURE picks what a pass measures, with the same runners and the same workloads:
#   census        (default) the slotcensus.c tracer, preloaded: allocations, frees, peak live objects
#   instructions  the slotins.c DynamoRIO client: instructions and allocations per process from the same run
#                 (allocations per instruction), in a pass of its own, without the tracer
# A runner launches every measured program through measured() (PARSEC's and pyperformance's harnesses, which start
# the programs themselves, get the same launch through a wrapper), so both modes run exactly the same commands.
#
# A pass: start_pass, then per cell measured ... and record_cell, then finish_pass. cached fetches a download once,
# took notes a cell's run time, control_verdict reports a control.

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

# record_cell CELL SELECTOR VERDICT NOTE: one row of cells.tsv (see summarize.py), echoed; CELLS names another file
# (pyperformance's parallel cells write one per arm)
record_cell() {
  printf '%s\t%s\t%s\t%s\n' "$@" | tee -a "${CELLS:-$OUT/cells.tsv}"
}

# finish_pass [FILE]: the marker a watcher waits for, after the last cell (run-spec.sh's is in runs.txt)
finish_pass() {
  echo CENSUS_DONE >> "$OUT/${1:-cells.tsv}"
}

# took T: the time since T (from date +%s), as a cells.tsv note writes it
took() {
  echo "$(( $(date +%s) - $1 ))s"
}

# cached FILE URL: FILE in $CACHE, downloaded once (under another name until it is complete, so a download cut
# short is never taken for the file)
cached() {
  [ -s "$CACHE/$1" ] && return
  curl -fsSL -o "$CACHE/$1.part" "$2" && mv "$CACHE/$1.part" "$CACHE/$1" \
    || { echo "${0##*/}: download of $1 failed" >&2; exit 1; }
}

# control_verdict NAME WANT CHECKER_RC CHECK_FILE: a control's checker passes with 0 and fails with 1; one that does
# not do what WANT (pass/fail) says stops the pass, before anything is measured
control_verdict() {
  local name=$1 want=$2 rc=$3 check=$4
  if { [ "$want" = pass ] && [ "$rc" = 0 ]; } || { [ "$want" = fail ] && [ "$rc" = 1 ]; }; then
    echo "control $name: as expected ($want)"
  else
    echo "control $name: NOT as expected (want $want, checker rc $rc)"; cat "$check"; exit 75
  fi
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
  control_verdict "$name" "$want" $rc "$OUT/control/$name.check"
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
  cached "$T" "https://github.com/DynamoRIO/dynamorio/releases/download/release_$DYNAMORIO-1/$T"
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
  local C=$OUT/control m n mode name bin rc
  for m in single threads fork; do
    mode=(); [ $m = single ] || mode=("$m")   # insctl's argument: none for one thread
    for n in 0 100000000; do
      rm -f "$C/ins-$m-$n.jsonl"
      "$DRRUN" -c "$SLOTINS" "$C/ins-$m-$n.jsonl" -- "$TOOL/insctl" "$n" "${mode[@]}" > /dev/null 2>&1
    done
  done
  python3 "$HERE/check-instructions.py" "$C" > "$C/ins.check"
  control_verdict instructions pass $? "$C/ins.check"
  # the allocation counts of the same run: the census's own controls, under DynamoRIO, checked the same way
  for name in c cxx; do
    bin=$TOOL/control; [ $name = cxx ] && bin=$TOOL/control_cxx
    rm -f "$C/$name.jsonl"
    "$DRRUN" -c "$SLOTINS" "$C/$name.jsonl" -- "$bin" > "$C/$name.expect" 2> /dev/null
    python3 "$HERE/check-control.py" "$C/$name.expect" "$C/$name.jsonl" \
      allocs,frees,realloc_moved,realloc_inplace,forked > "$C/$name.check"
    rc=$?
    control_verdict "$name (allocations under DynamoRIO)" pass $rc "$C/$name.check"
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
