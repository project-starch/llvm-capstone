#!/bin/bash
# The matched pair: every case in the no-revocation arm (level0) and the revoking arm
# (sublet), with a control in EACH arm first. The control is smoke.rb minus its two
# `deep` lines, because under sublet a script recursing ~40 deep faults in
# stack_extend_alloc while 20 is fine -- a measured limit of the buddy heap, not a
# catch. A row counts as caught only when the control arm COMPLETES and the revoking
# arm faults IN THE DEFECT'S OWN FUNCTION (docs/ref/HOW-TO-RUN-ON-QEMU.md section 3);
# a fault at the same place in both arms is an accident, and cause 24 alone proves
# nothing -- this build produces it in the arm that revokes nothing.
SP=${1:?usage: measure-arms.sh <workdir holding cases/ and the arm trees>}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../../../ports/mruby/app" || exit 2
export PATH=${CAPSTONE_VENV_BIN:-$HOME/.venvs/capstone/bin}:$PATH
export CAPSTONE_LLVM_BUILD_DIR=${CAPSTONE_LLVM_BUILD_DIR:?a clang containing 7d01722aab88}
export RUNTIME_REPO=${RUNTIME_REPO:?a worktree of runtime/libc-test-second-grant}
export CAPSTONE_QEMU_BINARY=${CAPSTONE_QEMU_BINARY:?capstone-qemu movc-merge d621df553f}
export CAPSTONE_GP_NONLIN=1 CAPSTONE_REV_NODES=16777216
CASES=(string-strip-long string-prepend-self hash-matched-vacated hash-scans-vacated
       hash-read-back hash-delete-in-eql hashext-walk-carries hash-pair-into-set
       openter-block iv-walk-freed-block)
run() { # arm img case
  local arm=$1 img=$2 c=$3 out=$SP/A-$arm-$c.txt try
  for try in 1 2 3; do
    rm -rf $SP/W-$arm-$c
    timeout 900 bash run-mruby-domain.sh "$img" "$SP/W-$arm-$c" 240 \
      "$SP/survey2/cases/$c.rb" -- "/mnt/host/files/$c.rb" > "$out" 2>&1
    grep -qE 'LT-RESULT|capability fault' "$out" && return 0
  done; return 1
}
for arm in level0 sublet; do
  IMG=$SP/tst-$arm/src/mruby/build/capstone/bin/mruby
  [ -x "$IMG" ] || IMG=$SP/tc-dom-$arm/src/mruby/build/capstone/bin/mruby
  unset MRBD_HEAP_REGION_BYTES; [[ $arm == sublet ]] && export MRBD_HEAP_REGION_BYTES=134217728
  echo "#### CONTROL $arm"
  run $arm "$IMG" ctl 2>/dev/null
  timeout 900 bash run-mruby-domain.sh "$IMG" "$SP/W-$arm-ctl" 240 $HERE/control-smoke-shallow.rb -- /mnt/host/files/control-smoke-shallow.rb > $SP/A-$arm-ctl.txt 2>&1
  grep -q SMOKE_DONE $SP/A-$arm-ctl.txt && echo "## control $arm SMOKE_DONE" || { echo "## control $arm FAILED"; continue; }
  for c in "${CASES[@]}"; do
    echo "#### $arm/$c $(date +%H:%M:%S)"
    run $arm "$IMG" "$c" || echo "   (no result after 3 tries)"
  done
done
echo "#### ALL DONE $(date +%H:%M:%S)"
