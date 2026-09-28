#!/bin/bash
# Run the cases as mruby domains under QEMU, one heap arm at a time, with the
# port's own smoke.rb as each arm's control: an arm whose control fails reports
# nothing about a defect. A run that neither finishes nor faults is an infra flake
# and is retried, not recorded. level0 is the matched pair's control arm -- it
# revokes nothing, so a cause-24 fault there is never a catch
# (docs/ref/HOW-TO-RUN-ON-QEMU.md section 3).
SP=${1:?usage: measure-domain.sh <survey.sh workdir>}   # survey.sh's workdir, for cases/
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../../../ports/mruby/app" || exit 2
export PATH=${CAPSTONE_VENV_BIN:-$HOME/.venvs/capstone/bin}:$PATH  # run-domain-smoke.py needs pexpect
export CAPSTONE_LLVM_BUILD_DIR=${CAPSTONE_LLVM_BUILD_DIR:?a clang containing 7d01722aab88}
export RUNTIME_REPO=${RUNTIME_REPO:?a worktree of runtime/libc-test-second-grant}
export CAPSTONE_QEMU_BINARY=${CAPSTONE_QEMU_BINARY:?capstone-qemu movc-merge d621df553f}
export CAPSTONE_GP_NONLIN=1 CAPSTONE_REV_NODES=16777216
export MRUBY_MIRROR=${MRUBY_SRC:?set MRUBY_SRC to an mruby clone} MRBD_PIN=4.0.0-rc2
CASES=(string-strip-bang-uaf hash-matched-vacated hash-scans-vacated hash-read-back
       hash-delete-in-eql hashext-walk-carries hash-pair-into-set openter-block iv-walk-freed-block)

grants() { unset MRBD_HEAP_REGION_BYTES MRBD_GC_REGION_BYTES
  [[ $1 == sublet* ]]   && export MRBD_HEAP_REGION_BYTES=134217728
  [[ $1 == sublet-gc ]] && export MRBD_GC_REGION_BYTES=67108864; }

# A run that neither finishes nor faults is an infra flake, not a result: retry twice.
run_once() {  # <img> <work> <file> <out>
  local img=$1 work=$2 f=$3 out=$4 try
  for try in 1 2 3; do
    rm -rf "$work"
    timeout 1200 bash run-mruby-domain.sh "$img" "$work" 240 "$f" -- "/mnt/host/files/$(basename $f)" > "$out" 2>&1
    if grep -qE 'LT-RESULT|capability fault' "$out"; then echo "  (attempt $try)"; return 0; fi
    echo "  (attempt $try: no result, retrying)"
  done
  return 1
}

for arm in level0 sublet sublet-gc; do
  IMG=$SP/tc-dom-$arm/src/mruby/build/capstone/bin/mruby
  if [ ! -x "$IMG" ]; then
    echo "############ BUILD $arm $(date +%H:%M:%S)"
    MRBD_HEAP=$arm MRBD_ROOT=$SP/tc-dom-$arm bash build-mruby-domain.sh > $SP/tcb-$arm.log 2>&1
    echo "## build $arm rc=$?"
  fi
  [ -x "$IMG" ] || { echo "## $arm NO IMAGE"; continue; }
  grants $arm
  echo "############ CONTROL smoke.rb in $arm $(date +%H:%M:%S)"
  run_once "$IMG" "$SP/m2-smoke-$arm" scripts/smoke.rb "$SP/m2-smoke-$arm.out"
  if grep -q SMOKE_DONE "$SP/m2-smoke-$arm.out"; then echo "## control $arm: SMOKE_DONE"
  else echo "## control $arm FAILED: $(grep -oE 'cause = [0-9]+|no LT-RESULT' $SP/m2-smoke-$arm.out | head -1) -- cases skipped"; continue; fi
  for c in "${CASES[@]}"; do
    echo "############ $arm/$c $(date +%H:%M:%S)"
    run_once "$IMG" "$SP/m2-run-$arm-$c" "$SP/survey2/cases/$c.rb" "$SP/m2-out-$arm-$c.txt"
    grep -E '^\["|capability fault|LT-RESULT' $SP/m2-out-$arm-$c.txt | tail -4
  done
done
echo "############ ALL DONE $(date +%H:%M:%S)"
