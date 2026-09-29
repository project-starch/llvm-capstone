#!/usr/bin/env bash
# Which programs give a different result when MOVC nulls an integer source, as the RTL does (Q-04)?
#
#   OUT=<dir> CAPSTONE_SDK=<sdk> VM_UP=<script> bash exposure.sh
#
# Runs everything twice, on the same QEMU binary with the same compiler: once with
# CAPSTONE_MOVC_NULL_SCALAR=0 (QEMU's default, an integer source survives) and once with =1 (the
# RTL, it is zeroed). Each arm runs
#   - the extended nightly, and each bare HostCall wire probe on its own (each boots its own QEMU);
#   - then, in a capstone_vm guest started with the arm's switch, the delegated runtime probes
#     (../run-delegated-probes.py) and musl's libc-test, one delegated application per test
#     (ports/musl-capstone/libc-test/run-libc-test-delegated.py).
# Then exposure-compare.py compares every verdict. The difference is what today's silicon does
# differently from every emulator run so far.
#
# VM_UP is a script run as `bash "$VM_UP" <state-dir> <share-dir>` with the switch exported: it
# brings up the lane's guest (python3 -m capstone_vm --state <state-dir> up --share <share-dir>
# with its qemu, kernel, firmware, rootfs and overrides) and exits 0. capstone_vm records the
# switch with the guest. This script stops the guest again with capstone_vm down: a guest holds
# the QEMU lock for as long as it runs, so it is up only between one arm's nightly and the next.
# CAPSTONE_SDK is the application SDK the delegated images are built with. libc-test must have
# been fetched (ports/musl-capstone/libc-test/fetch-libc-test.sh).
#
# Controls, so that a quiet result means something:
#   - the movc probe runs in each arm's guest and is judged by the switch that guest recorded:
#     b=5 c=5 with it off, b=5 c=0 with it on. A guest whose switch changed nothing fails it, and
#     the comparison refuses the arm;
#   - every boot with the switch on prints one notice the first time it nulls a non-zero integer
#     (the monitor does so while booting), and no boot with it off may; the comparison checks both.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR, and CAPSTONE_QEMU_BINARY with the switch
# (capstone-qemu 39db72ecac or later) that the nightly runs on. The two arms use $OUT/off and
# $OUT/on as CAPSTONE_TMP_ROOT; anything a suite would otherwise fetch or build (a musl tarball in
# musl-src/, a host tree) can be placed there first. The nightly takes the QEMU lock itself.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
# OUT comes in through the environment, so it is exported to everything started from here, and a
# child that assigns its own OUT (the nightly does) then hands that to ITS children: the PostgreSQL
# gates read ${OUT:-...} and looked for their host tree in the nightly's directory. Keep it local.
DEST=${OUT:?set OUT to an output directory}
unset OUT
: "${CAPSTONE_SDK:?set CAPSTONE_SDK to the application SDK build directory}"
: "${VM_UP:?set VM_UP to a script that brings up a capstone_vm guest}"
mkdir -p "$DEST"
DEST=$(cd -- "$DEST" && pwd)
PROBES=$REPO/capstone/tests/runtime-qemu/run-delegated-probes.py
LIBC=$REPO/capstone/ports/musl-capstone/libc-test/run-libc-test-delegated.py
VM=(env PYTHONPATH="$REPO/capstone/runtime/host" python3 -m capstone_vm)

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
  state=$DEST/$arm/vm share=$DEST/$arm/share
  mkdir -p "$share"
  echo "== arm $arm: a guest with the switch"
  if ! bash "$VM_UP" "$state" "$share" > "$DEST/vm-up-$arm.txt" 2>&1; then
    echo "the guest did not come up for arm $arm: see $DEST/vm-up-$arm.txt"
    "${VM[@]}" --state "$state" down > /dev/null 2>&1
    exit 2
  fi
  echo "== arm $arm: the delegated runtime probes"
  python3 "$PROBES" --sdk "$CAPSTONE_SDK" --work "$DEST/$arm/probes" --state "$state" \
    --report "$DEST/probes-$arm.json" > "$DEST/probes-$arm.txt" 2>&1
  echo "rc=$?" >> "$DEST/probes-$arm.txt"
  echo "== arm $arm: libc-test, one delegated application per test"
  python3 "$LIBC" --state "$state" --sdk "$CAPSTONE_SDK" --share "$share" \
    --work "$DEST/$arm/libc-work" --report "$DEST/libc-$arm.json" > "$DEST/libc-$arm.txt" 2>&1
  echo "rc=$?" >> "$DEST/libc-$arm.txt"
  "${VM[@]}" --state "$state" down > "$DEST/vm-down-$arm.txt" 2>&1
done
unset CAPSTONE_MOVC_NULL_SCALAR

python3 "$HERE/exposure-compare.py" "$DEST"
