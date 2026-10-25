#!/bin/bash
# S-26's firming arms, prepared (2026-10-19; the runtime lane's two arms, after sql0-treeA):
#  arm 1 "cur-esc": today's monitor with the pair's rung plus diag_escapes=1. That needs a bake of the CURRENT Tier 1 stage
#    (rev8's) with only vm/vm-contract.sh swapped for vm-contract-esc.sh, the stage restored to its record afterwards;
#    then the cur firmware (QNT0 print only) on that image.
#  arm 2 "var1k": the var firmware with CAPSTONE_BOOT_QUANTUM=1000 instead of 2,000,000, on the PAIR's image
#    89baf46451448872, so it differs from var by that one value.
# Both firmwares come from wrapper aab63d5 (wt-wrap-s26) and monitor ba9a0de (csbi-s26), as the pair's did.
# Usage: firming-prep.sh <log>. Ends by itself. FP_T, FP_BAKE and FP_FW stub the stage root and the two external steps.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${FP_T:-/tmp/capstone/tier1} X=/tmp/capstone/s26 I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${FP_BAKE:-bash /tmp/capstone/b0/b0-bake-tier1.sh} FW=${FP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT"
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "FP STOP: the stage is not its record" >> "$O"; return 1; }
  [ "$(sha256sum $X/image-s26/Image | cut -c1-16)" = 89baf46451448872 ] || { echo "FP STOP: the pair's image is not 89baf464" >> "$O"; return 1; }
  [ -e $X/image-esc ] && { echo "FP STOP: $X/image-esc exists" >> "$O"; return 1; }
  cp $T/stage.SHA256SUMS $X/stage-before-esc.SHA256SUMS; cp $T/stage/vm/vm-contract.sh $X/vm-contract-before-esc.sh
  cp $S/vm-contract-esc.sh $T/stage/vm/vm-contract.sh
  echo "FP BAKE $(date +%T): vm-contract.sh $(sha256sum $T/stage/vm/vm-contract.sh | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-esc.out 2>&1; local rc=$?
  cp $X/vm-contract-before-esc.sh $T/stage/vm/vm-contract.sh; cp $X/stage-before-esc.SHA256SUMS $T/stage.SHA256SUMS
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "FP STOP: the stage did not restore to its record" >> "$O"; return 1; }
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-esc.out && grep -q "cpio test-domains/vm/vm-contract.sh: present" $X/bake-esc.out \
    || { echo "FP STOP: the bake failed (rc $rc)" >> "$O"; return 1; }
  mkdir -p $X/image-esc; cp $I/Image $I/caplifive.dtb $X/image-esc/; chmod a-w $X/image-esc/*
  echo "FP IMAGE esc $(sha256sum $X/image-esc/Image | cut -c1-16); stage restored to its record" >> "$O"
  local a x im
  for a in cur-esc var1k; do
    case $a in cur-esc) x="$D"; im=$X/image-esc;; var1k) x="$D -DCAPSTONE_BOOT_QUANTUM=1000"; im=$X/image-s26;; esac
    [ -e $X/fw-s26-$a ] && { echo "FP STOP: $X/fw-s26-$a exists" >> "$O"; return 1; }
    FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-s26 FW_IMAGE_DIR=$im nice -n 10 $FW $X/fw-s26-$a "$x" > $X/fw-s26-$a.log 2>&1
    grep -q "^gate: PASS" $X/fw-s26-$a.log && [ -f $X/fw-s26-$a/fw_payload.bin ] || { echo "FP STOP: firmware $a failed" >> "$O"; return 1; }
    echo "FP FW $a $(sha256sum $X/fw-s26-$a/fw_payload.bin | cut -c1-16) on image $(sha256sum $im/Image | cut -c1-16)" >> "$O"
  done
  echo "FP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
