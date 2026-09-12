#!/usr/bin/env bash
# Does MicroPython still run under capabilities? The gate.
#
#   run-micropython-gate.sh
#
# Three steps, each one a thing a change would break:
#
#   1  the verdict logic still refuses wrong answers. No board, no build, and
#      first because a gate whose scoring is broken reports whatever it likes.
#   2  the image still links in two passes and still fits what the module can
#      create, which the build script's own verdict says
#   3  nothing: see the note in the body about census-capstone.sh, which
#      measures a configuration this port does not build
#   3  it still runs: a fixed set of the upstream tests, in a domain, scored
#      against what the host Python produced
#
# The first run fetches MicroPython at its pin and applies patches/, so it
# needs the network once. MPY_GATE_TESTS sets how many tests step 4 runs; the
# default is small on purpose, because this gate answers whether the path works
# and not how much of Python does.
#
# Exit 0 if all three hold, non-zero otherwise, as the nightly expects.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

GATE=${GATE:-$CAPSTONE_TMP_ROOT/micropython-gate}
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}
TESTS=${MPY_GATE_TESTS:-4}
PYTHON=${PYTHON:-python3}
DOM=mpy-gate

mkdir -p "$GATE" "$SHARE"

echo "== 1. the verdict logic, which scores everything below"
bash "$SCRIPT_DIR/repro-selftest.sh"

echo
echo "== the source at its pin"
bash "$SCRIPT_DIR/fetch-micropython.sh"

echo
echo "== 2. the image, in two passes"
# Before the census, not after: the build generates this port's qstr and module
# headers with a stock toolchain, and the census refuses without them.
MPY_TESTS=$TESTS MPY_FLOAT_CORE=1 DOM_NAME=$DOM \
  bash "$SCRIPT_DIR/build-micropython-silicon.sh" > "$GATE/build.txt" 2>&1 || {
  echo "the image did not build; $GATE/build.txt says why" >&2
  exit 1; }

# census-capstone.sh is NOT in this gate, and that is a finding rather than an
# omission. It compiles each py/*.c against ports/minimal's generated headers
# and mpconfigport.h, while the build uses this port's own, generated from this
# port's mpconfigport.h. The build script says why they cannot be borrowed:
# MICROPY_STACK_CHECK alone pulls in a qstr ports/minimal does not define, and
# a mismatched string pool is an error that compiles. So the census measures a
# configuration we do not build, and gating on it would gate on the wrong
# thing. Pointing it at this port's headers is the fix and it is not wiring.
grep -E "\.text =|VERDICT" "$GATE/build.txt"
grep -q "VERDICT: fits" "$GATE/build.txt" || {
  echo "the image does not fit what the module can create" >&2; exit 1; }

echo
echo "== the guest runner, which is not capstone-test.user"
# capstone-test.user takes <dom> <times> [<second-elf>] and would read the
# suite runner's arguments as a second ELF, fail to create the domain, and
# produce monitor-side faults that look like results.
GCC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
U=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace
"$GCC" -O2 -I "$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" -I "$U" \
    -o "$SHARE/mpy-resume-guest" \
    "$SCRIPT_DIR/tools/mpy-resume-guest.c" "$U/lib/libcapstone.c"

echo
echo "== 3. $TESTS tests in a domain"
cp "$CAPSTONE_TMP_ROOT/micropython-silicon/$DOM.dom" "$SHARE/"
"$PYTHON" "$SCRIPT_DIR/tools/run-resumable-suite.py" \
  --domain "$SHARE/$DOM.dom" \
  --expected "$CAPSTONE_TMP_ROOT/micropython-silicon/obj/mpy_tests.expected" \
  --guest-runner "$SHARE/mpy-resume-guest" \
  --out-dir "$GATE/out" | tail -3

# The suite runner reports and does not judge, because a measurement wants the
# numbers whatever they are. A gate has to judge.
rows=$(awk -F'\t' 'NR>1' "$GATE/out/results.tsv" | wc -l)
passed=$(awk -F'\t' 'NR>1 && $3=="PASS"' "$GATE/out/results.tsv" | wc -l)
if [ "$rows" -ne "$TESTS" ] || [ "$passed" -ne "$TESTS" ]; then
  echo "expected $TESTS rows all PASS, got $rows rows and $passed passing;" >&2
  echo "  $GATE/out/results.tsv has them" >&2
  exit 1
fi

echo
echo "the interpreter builds, links, fits, and runs $TESTS tests in a domain"
