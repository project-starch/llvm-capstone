#!/usr/bin/env bash
# Shared driver behind every case's run.sh, in the shape fpga-repros already uses:
# the control runs FIRST, a run whose control fails carries NO verdict and exits 75,
# and the verdict line names the expected value next to the measured one.
#
# One driver rather than seventeen copies. A case's run.sh declares what it is and
# sources this; nothing case-specific lives here.
#
# Three modes, because there are three ways a measurement is taken:
#   repro_scripts  Python in the domain, via the baked test table and the resumable
#                  suite. The control is 00_sanity.py in the same image.
#   repro_glue     a C reconstruction behind a -D in port/mpy_domain.c, via
#                  run-domain-smoke.py. The control is that the domain returns at all
#                  with the expected marker in the high byte.
#   repro_ladder   a matched pair in tests/runtime-qemu/silicon-ladder. The control
#                  IS the other arm, and it runs first.
set -uo pipefail

REPO=$(git rev-parse --show-toplevel) || { echo "not in a git repo"; exit 1; }
cd "$REPO" || exit 1
export CAPSTONE_REPO_ROOT="$REPO"
# capstone-test-env.sh uses BASH_SOURCE and resolves the root wrongly under zsh, which
# is why CAPSTONE_REPO_ROOT is exported first rather than trusted from it.
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1

MPY=capstone/benchmarks/micropython
BUILD=$MPY/build-micropython-silicon.sh
SHARE=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/capstone-runtime-qemu-share
OBJ=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/micropython-silicon
ROM="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES"
ATTEMPTS=${REPRO_ATTEMPTS:-3}
INFRA=75

die()  { echo "   $*"; exit 1; }
void() { echo "   $* -> run VOID, no verdict"; exit $INFRA; }

# The suite runner needs mpy-resume-guest, NOT capstone-test.user. The latter takes
# <dom> <times> [<second-elf>] and reads the suite's <start> <count> as a second ELF:
# domain creation returns -1 and the monitor's own faults read like results. Built
# here rather than assumed present.
ensure_guest_runner() {
  local out=$SHARE/mpy-resume-guest
  [[ -x $out ]] && return 0
  mkdir -p "$SHARE"
  local gcc=capstone/caplifive-buildroot/build/host/bin/riscv64-buildroot-linux-gnu-gcc
  local u=capstone/caplifive-buildroot/package/modcapstone/userspace
  [[ -x $gcc ]] || die "no guest toolchain at $gcc -- build buildroot first"
  "$gcc" -O2 -I capstone/caplifive-buildroot/package/modcapstone/include -I "$u" \
     -o "$out" "$MPY/tools/mpy-resume-guest.c" "$u/lib/libcapstone.c" \
     || die "could not build the guest runner"
}

# $1 = directory IN THIS REPO holding the scripts, $2 = image name.
# The scripts are staged into the MicroPython tree here rather than by hand: a case
# folder that needs an undocumented copy first is not a reproduction.
repro_scripts() {
  local src=$1 name=$2; shift 2
  local dir=capstone-repro-$name
  local dst=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/micropython/tests/$dir
  [[ -d $src ]] || die "no scripts at $src"
  ls "$src"/*.py >/dev/null 2>&1 || die "no .py in $src"
  [[ -f $src/00_sanity.py ]] || die "$src has no 00_sanity.py -- refusing to run without a control"
  rm -rf "$dst"; mkdir -p "$dst"; cp "$src"/*.py "$dst"/
  echo "== staged $(ls "$dst"/*.py | wc -l) scripts into tests/$dir"
  ensure_guest_runner
  echo "== building $name from tests/$dir"
  # MPY_TEST_INCLUDE_UNSUPPORTED=1 is load-bearing: the table generator derives each
  # expectation by running the script on the HOST and drops any that exits non-zero.
  # Every script that is SUPPOSED to crash would otherwise be silently missing.
  MPY_TESTS=all MPY_TEST_BASE_DIR="$dir" MPY_TEST_INCLUDE_UNSUPPORTED=1 \
  MPY_FLOAT_CORE=1 DOMAIN_EXTRA_DEFS="$ROM ${REPRO_EXTRA_DEFS:-}" DOM_NAME="$name" \
    bash "$BUILD" >"/tmp/capstone/$name-build.log" 2>&1 \
    || { tail -15 "/tmp/capstone/$name-build.log"; die "build failed"; }
  grep -q "00_sanity.py" "$OBJ/obj/mpy_tests.expected" \
    || die "no 00_sanity.py in the baked table -- without a control this run means nothing"
  cp -f "$OBJ/$name.dom" "$SHARE/" || die "no image at $OBJ/$name.dom"

  local out=/tmp/capstone/$name-run
  for a in $(seq 1 "$ATTEMPTS"); do
    rm -rf "$out"
    python3 "$MPY/tools/run-resumable-suite.py" --domain "$SHARE/$name.dom" \
      --expected "$OBJ/obj/mpy_tests.expected" --guest-runner "$SHARE/mpy-resume-guest" \
      --out-dir "$out" --capture-output >"/tmp/capstone/$name-suite.log" 2>&1
    grep -qa "Created domain ID = 0" "$out"/round-*.log 2>/dev/null && break
    echo "   attempt $a: no domain was created"
  done
  grep -qa "Created domain ID = 0" "$out"/round-*.log 2>/dev/null \
    || void "domain never created in $ATTEMPTS attempts"
  awk -F'\t' '$2=="00_sanity.py" && $3=="PASS"' "$out/results.tsv" | grep -q . \
    || void "00_sanity.py did not pass"
  echo "   control ok (00_sanity.py PASS)"
  REPRO_OUT_DIR=$out
}

# $1 = -D flag, $2 = image name, $3 = expected retval in decimal, or FAULT
repro_glue() {
  local flag=$1 name=$2 want=$3
  echo "== building $name with $flag"
  MPY_FLOAT_CORE=1 DOMAIN_EXTRA_DEFS="-D$flag $ROM" DOM_NAME="$name" \
    bash "$BUILD" >"/tmp/capstone/$name-build.log" 2>&1 \
    || { tail -15 "/tmp/capstone/$name-build.log"; die "build failed"; }
  mkdir -p "$SHARE"; cp -f "$OBJ/$name.dom" "$SHARE/" || die "no image"
  echo "   image md5 $(md5sum "$SHARE/$name.dom" | cut -c1-12)"

  # --log-file explicitly: run-domain-smoke.py prints only "QEMU smoke passed." on
  # stdout and writes the loader lines, the retval and any fault to its log file,
  # which defaults to a SHARED path. Grepping stdout finds neither, and grepping the
  # shared default would read another run's result.
  local log=/tmp/capstone/$name-serial.log
  for a in $(seq 1 "$ATTEMPTS"); do
    python3 capstone/tests/runtime-qemu/run-domain-smoke.py --log-file "$log" \
      "$SHARE/$name.dom" >"/tmp/capstone/$name-run.log" 2>&1
    grep -qa "Created domain ID = 0" "$log" && break
    echo "   attempt $a: no domain was created"
  done
  grep -qa "Created domain ID = 0" "$log" || void "domain never created"
  local got; got=$(grep -ao "retval = [0-9]*" "$log" | tail -1 | awk '{print $3}')
  if [[ "$want" == "FAULT" ]]; then
    if grep -qa "capability fault" "$log"; then
      echo "   VERDICT: as recorded -- FAULT"
      grep -ao "bounds = ([^)]*)" "$log" | tail -1 | sed 's/^/            /'
    else
      echo "   VERDICT: DIFFERS -- expected a fault, got retval ${got:-<none>}"; exit 1
    fi
  elif [[ -z "$got" ]]; then
    void "no retval and no fault"
  elif [[ "$got" == "$want" ]]; then
    printf "   VERDICT: as recorded -- retval %s = 0x%08X\n" "$got" "$got"
  else
    printf "   VERDICT: DIFFERS -- got %s (0x%08X), recorded %s (0x%08X)\n" "$got" "$got" "$want" "$want"
    exit 1
  fi
}

# $1 = ladder base, $2 = rung name, $3 = -D, $4 = expected retval or FAULT
repro_ladder() {
  local base=$1 rung=$2 def=$3 want=$4
  local log=/tmp/capstone/$rung.log
  for a in $(seq 1 "$ATTEMPTS"); do
    DOMAIN_EXTRA_CFLAGS="$def" RUNG_NAME="$rung" \
      bash capstone/tests/runtime-qemu/silicon-ladder/run-ladder-qemu.sh "$base" >"$log" 2>&1
    local e=$?
    [[ $e -ne $INFRA ]] && break
    echo "   attempt $a: infra flake"
  done
  if [[ "$want" == "FAULT" ]]; then
    grep -qa "capability fault" "$log" \
      && { echo "   VERDICT: as recorded -- FAULT"; grep -ao "bounds = ([^)]*)" "$log" | tail -1 | sed 's/^/            /'; } \
      || { echo "   VERDICT: DIFFERS -- no fault"; grep -a "retval" "$log" | tail -1; exit 1; }
  else
    if grep -qa "PASSED_ (retval = $want)\|PASSED__ (retval = $want)" "$log"; then
      echo "   VERDICT: as recorded -- retval $want"
    else
      echo "   VERDICT: DIFFERS"; grep -aiE "oracle|retval|FAILED" "$log" | tail -3; exit 1
    fi
  fi
}

# $1 = script name in the baked table, $2 = expected status, $3 = expected retval or -
# Used after repro_scripts, which leaves the results in $REPRO_OUT_DIR.
check_row() {
  local script=$1 wstat=$2 wgot=$3
  local line; line=$(awk -F'\t' -v s="$script" '$2==s' "$REPRO_OUT_DIR/results.tsv")
  [[ -n "$line" ]] || void "no row for $script in results.tsv"
  local gstat ggot; gstat=$(cut -f3 <<<"$line"); ggot=$(cut -f4 <<<"$line")
  if [[ "$gstat" == "$wstat" && ( "$wgot" == "-" || "$ggot" == "$wgot" ) ]]; then
    echo "   VERDICT: as recorded -- $script $gstat ${ggot}"
  else
    echo "   VERDICT: DIFFERS -- $script got $gstat $ggot, recorded $wstat $wgot"
    echo "            captured output:"; sed 's/^/              /' "$REPRO_OUT_DIR/actual-output/"*"${script%.py}"*.actual 2>/dev/null | head -12
    exit 1
  fi
}

# Prints the case's captured stdout, which for several rows IS the measurement.
show_output() {
  local script=$1
  local f; f=$(ls "$REPRO_OUT_DIR/actual-output/"*"${script%.py}"*.actual 2>/dev/null | head -1)
  [[ -f "$f" ]] && { echo "   domain output:"; sed 's/^/     /' "$f"; }
}
