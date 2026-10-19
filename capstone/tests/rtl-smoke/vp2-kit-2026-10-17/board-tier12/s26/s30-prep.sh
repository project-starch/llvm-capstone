#!/bin/bash
# S-30's matched pair, prepared (2026-10-19 evening, pre-registered in tier3-smoke/README.md before the boots):
#  1. the S-30 image: the PAIR's stage (rev7's record with the pair's bounded vm-contract.sh, exactly as image-s26) with
#     vmrun.sh extended by ONE case, s30-read (vmrun-s30.case); the stage restored exactly to its record afterwards;
#  2. two firmwares on it that differ only in the wrapper: s30-old (capstone-sbi 5b5e86b + wrapper aab63d5, fix1's) and
#     s30-new (capstone-sbi 5b5e86b + wrapper 9bee361, S-30's held carve).
# Usage: s30-prep.sh <log>. Ends by itself. SP_T, SP_X, SP_BAKE and SP_FW stub the stage root, the output root and the two
# external steps for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${SP_T:-/tmp/capstone/tier1} X=${SP_X:-/tmp/capstone/s26} I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${SP_BAKE:-bash /tmp/capstone/b0/b0-bake-tier1.sh} FW=${SP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local A=$T/stage-attic/rev8-1019-1626 K=$X/s30-stage-keep
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000"
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "SP STOP: the stage is not its record" >> "$O"; return 1; }
  [ -e $X/image-s30 ] && { echo "SP STOP: $X/image-s30 exists" >> "$O"; return 1; }
  [ -e $K ] && { echo "SP STOP: $K exists" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-fix rev-parse --short=7 HEAD)" = 5b5e86b ] && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s30 rev-parse --short=7 HEAD)" = 9bee361 ] || { echo "SP STOP: a source tree is not at its pin" >> "$O"; return 1; }
  mkdir -p $K/vm; cp -p $T/stage.SHA256SUMS $K/; cp -p $T/stage/vmrun.sh $K/; cp -p $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh $T/stage/vm/contract-dense.dom $K/vm/
  cp -p $A/vm/vm-suite-inner.sh $A/vm/vm-contract.sh $T/stage/vm/; rm -f "${T:?}"/stage/vm/contract-dense.dom
  ( cd $T/stage && sha256sum -c --quiet $X/stage-rev7.SHA256SUMS ) || { echo "SP STOP: the stage did not become rev7's record" >> "$O"; restore; return 1; }
  cp $S/vm-contract.sh $T/stage/vm/vm-contract.sh
  # vmrun.sh + the s30-read case, inserted before its catch-all, checked by count
  python3 - $T/stage/vmrun.sh $S/vmrun-s30.case <<'PY' || { echo "SP STOP: vmrun.sh did not take the case" >> "$O"; restore; return 1; }
import sys
p, c = sys.argv[1], open(sys.argv[2]).read(); s = open(p).read()
o = '  *) exec /test-domains/b0run.sh "$@" ;;'
assert s.count(o) == 1 and "s30-read)" not in s
open(p, "w").write(s.replace(o, c + o))
PY
  sh -n $T/stage/vmrun.sh || { echo "SP STOP: vmrun.sh does not parse" >> "$O"; restore; return 1; }
  echo "SP BAKE $(date +%T): the pair's stage, vm-contract.sh $(sha256sum $T/stage/vm/vm-contract.sh | cut -c1-16), vmrun.sh $(sha256sum $T/stage/vmrun.sh | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-s30.out 2>&1; local rc=$?
  restore || return 1
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-s30.out && grep -q "cpio test-domains/vmrun.sh: present" $X/bake-s30.out \
    || { echo "SP STOP: the bake failed (rc $rc)" >> "$O"; return 1; }
  mkdir -p $X/image-s30; cp $I/Image $I/caplifive.dtb $X/image-s30/; chmod a-w $X/image-s30/*
  echo "SP IMAGE $(sha256sum $X/image-s30/Image | cut -c1-16) dtb $(sha256sum $X/image-s30/caplifive.dtb | cut -c1-16)" >> "$O"
  local a w
  for a in s30-old s30-new; do
    case $a in s30-old) w=/tmp/capstone/wt-wrap-s26;; s30-new) w=/tmp/capstone/wt-wrap-s30;; esac
    [ -e $X/fw-$a ] && { echo "SP STOP: $X/fw-$a exists" >> "$O"; return 1; }
    FW_WRAP=$w FW_MON=/tmp/capstone/csbi-fix FW_IMAGE_DIR=$X/image-s30 nice -n 10 $FW $X/fw-$a "$D" > $X/fw-$a.log 2>&1
    grep -q "^gate: PASS" $X/fw-$a.log && [ -f $X/fw-$a/fw_payload.bin ] || { echo "SP STOP: firmware $a failed" >> "$O"; return 1; }
    echo "SP FW $a $(sha256sum $X/fw-$a/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-$a/fw_payload.bin) B wrapper $(git -C $w rev-parse --short=7 HEAD)" >> "$O"
  done
  echo "SP DONE $(date +%T)" >> "$O"
}
restore() {
  cp -p $K/vmrun.sh $T/stage/vmrun.sh; cp -p $K/vm/vm-contract.sh $K/vm/vm-suite-inner.sh $K/vm/contract-dense.dom $T/stage/vm/; cp -p $K/stage.SHA256SUMS $T/stage.SHA256SUMS
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "SP STOP: the stage did not restore to its record" >> "$O"; return 1; }
  echo "SP stage restored to its record $(date +%T)" >> "$O"
}
main "$@"; exit $?
