#!/usr/bin/env bash
# Which programs give a different result when MOVC nulls an integer source, as the RTL does (Q-04)?
#
#   OUT=<dir> bash exposure.sh
#
# Runs the extended nightly and musl's libc-test twice, on the same QEMU binary with the same
# compiler: once with CAPSTONE_MOVC_NULL_SCALAR=0 (QEMU's default, an integer source survives) and
# once with =1 (the RTL, it is zeroed). Then exposure-compare.py compares every verdict. The
# difference is what today's silicon does differently from every emulator run so far.
#
# libc-test runs in chunks, and a fault in a chunk hides the rest of it, so each arm re-runs its
# NOTRUN tests one per boot before the comparison.
#
# Controls, so that a quiet result means something:
#   - first, run.sh in this directory must pass: this QEMU has the switch and it acts;
#   - every boot with the switch on prints one notice the first time it nulls a non-zero integer
#     (the monitor does so while booting), and no boot with it off may; the comparison checks both.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR, and CAPSTONE_QEMU_BINARY with the switch
# (capstone-qemu 39db72ecac or later) that the corpus runs on. The QEMU this repo pins (deb7d757 plus
# the switch) is not one: on dev it halts hostcall-all's fourth probe in the monitor with the switch
# off as well, and the suite stops there. capstone-qemu's c128-qemu-merge with the switch merged in
# runs it. The two arms use $OUT/off and $OUT/on as CAPSTONE_TMP_ROOT;
# anything a suite would otherwise fetch or build (a musl tarball in musl-src/, a host tree) can be
# placed there first. The nightly takes the QEMU lock itself; this script takes it for libc-test.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
# OUT comes in through the environment, so it is exported to everything started from here, and a
# child that assigns its own OUT (the nightly does) then hands that to ITS children: the PostgreSQL
# gates read ${OUT:-...} and looked for their host tree in the nightly's directory. Keep it local.
DEST=${OUT:?set OUT to an output directory}
unset OUT
mkdir -p "$DEST"
LIBC=$REPO/capstone/ports/musl-capstone/libc-test/run-libc-test.sh

echo "== control: the switch acts ($CAPSTONE_QEMU_BINARY)"
OUT=$DEST/control bash "$HERE/run.sh" > "$DEST/control.txt" 2>&1
rc=$?
tail -n 8 "$DEST/control.txt"
if [[ $rc != 0 ]]; then
  echo "control failed (rc=$rc): nothing below would be a measurement"
  exit 2
fi

for arm in off on; do
  v=0; [[ $arm == on ]] && v=1
  export CAPSTONE_MOVC_NULL_SCALAR=$v CAPSTONE_TMP_ROOT=$DEST/$arm
  mkdir -p "$CAPSTONE_TMP_ROOT"
  echo "== arm $arm (CAPSTONE_MOVC_NULL_SCALAR=$v): nightly"
  bash "$REPO/capstone/tests/run-nightly.sh" --skip-build --extended > "$DEST/nightly-$arm.console" 2>&1
  echo "rc=$?" >> "$DEST/nightly-$arm.console"
  # hostcall-all stops at its first failing probe, which would leave the rest unmeasured in both
  # arms, so every probe also runs on its own.
  echo "== arm $arm: each hostcall probe on its own"
  for r in $(grep -o 'run-hostcall-[a-z0-9-]*-probe\.sh' "$REPO/capstone/tests/runtime-qemu/run-hostcall-all.sh" | sort -u); do
    flock -w 21600 "$CAPSTONE_QEMU_LOCK" bash "$REPO/capstone/tests/runtime-qemu/$r" > "$DEST/hostcall-$arm-${r%.sh}.txt" 2>&1
    echo "rc=$?" >> "$DEST/hostcall-$arm-${r%.sh}.txt"
  done
  echo "== arm $arm: musl, and its file/stdio/write probes"
  if ! { bash "$REPO/capstone/ports/musl-capstone/prepare-musl-capstone.sh" > "$DEST/musl-prep-$arm.txt" 2>&1 &&
         bash "$REPO/capstone/ports/musl-capstone/build-musl-capstone.sh" > "$DEST/musl-build-$arm.txt" 2>&1; }; then
    echo "musl did not build for arm $arm: see $DEST/musl-build-$arm.txt"
    exit 2
  fi
  for p in file stdio write; do
    flock -w 21600 "$CAPSTONE_QEMU_LOCK" bash "$REPO/capstone/ports/musl-capstone/$p-probe/run-$p-probe.sh" \
      > "$DEST/probe-$arm-$p.txt" 2>&1
    echo "rc=$?" >> "$DEST/probe-$arm-$p.txt"
  done
  echo "== arm $arm: libc-test, chunked"
  CHUNK=10 RUN_ID=movc-$arm flock -w 21600 "$CAPSTONE_QEMU_LOCK" bash "$LIBC" > "$DEST/libc-$arm.txt" 2>&1
  echo "rc=$?" >> "$DEST/libc-$arm.txt"
  notrun=$(awk '$1 == "NOTRUN" { print $2 }' "$CAPSTONE_TMP_ROOT/musl-libc-test/logs/movc-$arm/results.txt" 2>/dev/null | tr '\n' ' ')
  if [[ -n ${notrun// /} ]]; then
    echo "== arm $arm: libc-test, the NOTRUN tests one per boot: $notrun"
    CHUNK=1 RUN_ID=movc-$arm-notrun TESTS="$notrun" flock -w 21600 "$CAPSTONE_QEMU_LOCK" bash "$LIBC" \
      > "$DEST/libc-$arm-notrun.txt" 2>&1
    echo "rc=$?" >> "$DEST/libc-$arm-notrun.txt"
  fi
done
unset CAPSTONE_MOVC_NULL_SCALAR

python3 "$HERE/exposure-compare.py" "$DEST"
