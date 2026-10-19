#!/bin/bash
# S-26's silicon bracket, prepared (2026-10-19; the RTL lane's ask, reshaped as three cold boots, see tier3-smoke/README.md):
#  1. the bracket image: the PAIR's stage (rev7's record, as /tmp/capstone/s26/stage-rev7.SHA256SUMS) with ONLY
#     vm/vm-contract.sh swapped for vm-contract-long.sh (the pair's rung, bound 600 s, reads every 60 s). Today's stage
#     is rev8's, so rev7's two differing files come back from the attic and contract-dense.dom steps aside; the stage is
#     then restored exactly to its current record, whatever the bake did.
#  2. three firmwares on that Image, var1k's recipe (wrapper aab63d5, monitor ba9a0de) with CAPSTONE_BOOT_QUANTUM 600,
#     300 and 100 in place of 1000.
# Usage: bracket-prep.sh <log>. Ends by itself. BP_T, BP_X, BP_BAKE and BP_FW stub the stage root, the output root and the
# two external steps for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${BP_T:-/tmp/capstone/tier1} X=${BP_X:-/tmp/capstone/s26} I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${BP_BAKE:-bash /tmp/capstone/b0/b0-bake-tier1.sh} FW=${BP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local A=$T/stage-attic/rev8-1019-1626 K=$X/bracket-stage-keep
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT"
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage is not its record" >> "$O"; return 1; }
  [ -e $X/image-bracket ] && { echo "BP STOP: $X/image-bracket exists" >> "$O"; return 1; }
  [ -e $K ] && { echo "BP STOP: $K exists" >> "$O"; return 1; }
  mkdir -p $K/vm; cp -p $T/stage.SHA256SUMS $K/; cp -p $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh $T/stage/vm/contract-dense.dom $K/vm/
  # the pair's stage: rev7's vm-suite-inner.sh, no contract-dense.dom, rev7's vm-contract.sh, checked against rev7's record
  cp -p $A/vm/vm-suite-inner.sh $A/vm/vm-contract.sh $T/stage/vm/; rm -f "${T:?}"/stage/vm/contract-dense.dom
  ( cd $T/stage && sha256sum -c --quiet $X/stage-rev7.SHA256SUMS ) && [ -z "$(cd $T/stage && find . -type f | sed 's|^\./||' | sort | comm -23 - <(awk '{print $2}' $X/stage-rev7.SHA256SUMS | sort))" ] \
    || { echo "BP STOP: the stage did not become the pair's (rev7's) record" >> "$O"; restore; return 1; }
  cp $S/vm-contract-long.sh $T/stage/vm/vm-contract.sh
  echo "BP BAKE $(date +%T): the pair's stage, vm-contract.sh $(sha256sum $T/stage/vm/vm-contract.sh | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-bracket.out 2>&1; local rc=$?
  restore || return 1
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-bracket.out && grep -q "cpio test-domains/vm/vm-contract.sh: present" $X/bake-bracket.out \
    && ! grep -q "cpio test-domains/vm/contract-dense.dom: present" $X/bake-bracket.out \
    || { echo "BP STOP: the bake failed (rc $rc): $(grep -E 'BAKE_DONE|FAILED|STOP' $X/bake-bracket.out | tail -2 | tr '\n' ' ')" >> "$O"; return 1; }
  mkdir -p $X/image-bracket; cp $I/Image $I/caplifive.dtb $X/image-bracket/; chmod a-w $X/image-bracket/*
  echo "BP IMAGE $(sha256sum $X/image-bracket/Image | cut -c1-16) dtb $(sha256sum $X/image-bracket/caplifive.dtb | cut -c1-16)" >> "$O"
  local q
  for q in 600 300 100; do
    [ -e $X/fw-s26-q$q ] && { echo "BP STOP: $X/fw-s26-q$q exists" >> "$O"; return 1; }
    FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-s26 FW_IMAGE_DIR=$X/image-bracket nice -n 10 $FW $X/fw-s26-q$q "$D -DCAPSTONE_BOOT_QUANTUM=$q" > $X/fw-s26-q$q.log 2>&1
    grep -q "^gate: PASS" $X/fw-s26-q$q.log && [ -f $X/fw-s26-q$q/fw_payload.bin ] || { echo "BP STOP: firmware q$q failed" >> "$O"; return 1; }
    echo "BP FW q$q $(sha256sum $X/fw-s26-q$q/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-s26-q$q/fw_payload.bin) B on image $(sha256sum $X/image-bracket/Image | cut -c1-16)" >> "$O"
  done
  echo "BP DONE $(date +%T)" >> "$O"
}
# today's stage back, exactly: the three kept files and the record, checked
restore() {
  cp -p $K/vm/vm-contract.sh $K/vm/vm-suite-inner.sh $K/vm/contract-dense.dom $T/stage/vm/; cp -p $K/stage.SHA256SUMS $T/stage.SHA256SUMS
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage did not restore to its record" >> "$O"; return 1; }
  echo "BP stage restored to its record $(date +%T)" >> "$O"
}
main "$@"; exit $?
