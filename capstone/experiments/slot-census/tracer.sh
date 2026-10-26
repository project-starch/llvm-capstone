# Sourced by the runners: build the tracer and its controls into $TOOL and run the three controls into $OUT/control.
# Needs HERE (this directory), OUT and TOOL. A control that does not behave as expected exits 75, before any
# measurement runs.

build_tracer() {
  mkdir -p "$TOOL" "$OUT/control"
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
  control c "$TOOL/control" 1 pass
  control cxx "$TOOL/control_cxx" 1 pass
  control negative "$TOOL/control" 0 fail
}

provenance_tracer() {
  echo "date $(date -u +%FT%TZ)"
  echo "gcc $(gcc -dumpfullversion) g++ $(g++ -dumpfullversion)"
  echo "glibc $(ldd --version | head -1)"
  echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  (cd "$HERE" && sha256sum slotcensus.c control.c control_cxx.cc) && (cd "$TOOL" && sha256sum slotcensus.so)
}
