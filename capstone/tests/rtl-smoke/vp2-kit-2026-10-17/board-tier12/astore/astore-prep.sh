#!/bin/bash
# The alias-store test, prepared (2026-10-20; pre-registered in tier3-smoke/README.md before the boot): image-astore, rev7's
# Tier 1 stage with its OWN module (39d8492d, no swap) and vmrun.sh extended by ONE case, alias-store
# (board-tier12/astore/vmrun-astore.case), baked by alias/b0-bake-astore.sh (the Tier 1 bake plus alias-store.dom); the stage
# restored exactly afterwards; then ONE firmware, fw-astore, with the retry monitor line's recipe (f13dbc5, aab63d5, the D
# defines), as tree A's kits. Derived from ff/ff-prep.sh. Usage: astore-prep.sh <log>. Ends by itself. BP_T, BP_X, BP_BAKE,
# BP_FW stub for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${BP_T:-/tmp/capstone/tier1} X=${BP_X:-/tmp/capstone/s26} I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${BP_BAKE:-bash $S/b0-bake-astore.sh} FW=${BP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local A=$T/stage-attic/rev8-1019-1626 K=$X/astore-stage-keep
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000 -DCAPSTONE_S23_REFUSAL_READ -DCAPSTONE_S23_DATA_RETRY"
  [ "$(sha256sum $X/image-rev7/Image | cut -c1-16)" = dcfc2e15bf1bbd77 ] || { echo "BP STOP: image-rev7 is not dcfc2e15" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-drt rev-parse --short=7 HEAD)" = f13dbc5 ] && [ -z "$(git -C /tmp/capstone/csbi-drt status --porcelain)" ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "BP STOP: monitor or wrapper not at f13dbc5 / aab63d5" >> "$O"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage is not its record" >> "$O"; return 1; }
  [ -e $X/image-astore ] && { echo "BP STOP: $X/image-astore exists" >> "$O"; return 1; }
  [ "$(sha256sum /mnt/hdd/$USER/capstone/aliasstore/alias-store.dom | cut -c1-16)" = b68b3d2586a91d07 ] \
    || { echo "BP STOP: the alias-store image is not as recorded" >> "$O"; return 1; }
  [ -e $K ] && { echo "BP STOP: $K exists" >> "$O"; return 1; }
  mkdir -p $K/vm; cp -p $T/stage.SHA256SUMS $K/; cp -p $T/stage/vmrun.sh $K/; cp -p $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh $T/stage/vm/contract-dense.dom $K/vm/
  # the pair's stage: rev7's vm-suite-inner.sh, no contract-dense.dom, rev7's vm-contract.sh, checked against rev7's record
  cp -p $A/vm/vm-suite-inner.sh $A/vm/vm-contract.sh $T/stage/vm/; rm -f "${T:?}"/stage/vm/contract-dense.dom
  ( cd $T/stage && sha256sum -c --quiet $X/stage-rev7.SHA256SUMS ) && [ -z "$(cd $T/stage && find . -type f | sed 's|^\./||' | sort | comm -23 - <(awk '{print $2}' $X/stage-rev7.SHA256SUMS | sort))" ] \
    || { echo "BP STOP: the stage did not become the pair's (rev7's) record" >> "$O"; restore; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] || { echo "BP STOP: the staged module is not 39d8492d" >> "$O"; restore; return 1; }
  python3 - $T/stage/vmrun.sh $S/vmrun-astore.case <<'PY' || { echo "BP STOP: vmrun.sh did not take the case" >> "$O"; restore; return 1; }
import sys
p, c = sys.argv[1], open(sys.argv[2]).read(); s = open(p).read()
o = '  *) exec /test-domains/b0run.sh "$@" ;;'
assert s.count(o) == 1 and "alias-store)" not in s
open(p, "w").write(s.replace(o, c + o))
PY
  sh -n $T/stage/vmrun.sh || { echo "BP STOP: vmrun.sh does not parse" >> "$O"; restore; return 1; }
  echo "BP BAKE $(date +%T): rev7's stage, capstone_vm.ko $(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-astore.out 2>&1; local rc=$?
  restore || return 1
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-astore.out && grep -q "cpio test-domains/vm/capstone_vm.ko: present" $X/bake-astore.out && grep -q "cpio test-domains/vm/alias-store.dom: present" $X/bake-astore.out \
    && ! grep -q "cpio test-domains/vm/contract-dense.dom: present" $X/bake-astore.out \
    || { echo "BP STOP: the bake failed (rc $rc): $(grep -E 'BAKE_DONE|FAILED|STOP' $X/bake-astore.out | tail -2 | tr '\n' ' ')" >> "$O"; return 1; }
  mkdir -p $X/image-astore; cp $I/Image $I/caplifive.dtb $X/image-astore/; chmod a-w $X/image-astore/*
  echo "BP IMAGE alias $(sha256sum $X/image-astore/Image | cut -c1-16) dtb $(sha256sum $X/image-astore/caplifive.dtb | cut -c1-16)" >> "$O"
  [ -e $X/fw-astore ] && { echo "BP STOP: $X/fw-astore exists" >> "$O"; return 1; }
  FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-drt FW_IMAGE_DIR=$X/image-astore nice -n 10 $FW $X/fw-astore "$D" > $X/fw-astore.log 2>&1
  grep -q "^gate: PASS" $X/fw-astore.log && [ -f $X/fw-astore/fw_payload.bin ] || { echo "BP STOP: firmware alias failed" >> "$O"; return 1; }
  echo "BP FW alias $(sha256sum $X/fw-astore/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-astore/fw_payload.bin) B on image $(sha256sum $X/image-astore/Image | cut -c1-16)" >> "$O"
  echo "BP DONE $(date +%T)" >> "$O"
}
# today's stage back, exactly: the three kept files and the record, checked
restore() {
  cp -p $K/vm/vm-contract.sh $K/vm/vm-suite-inner.sh $K/vm/contract-dense.dom $T/stage/vm/; cp -p $K/stage.SHA256SUMS $T/stage.SHA256SUMS
  [ -f $K/vmrun.sh ] && cp -p $K/vmrun.sh $T/stage/vmrun.sh
  [ -f $K/vm/capstone_vm.ko.39d8492d ] && { rm -f "${T:?}"/stage/vm/capstone_vm.ko; cp -p $K/vm/capstone_vm.ko.39d8492d $T/stage/vm/capstone_vm.ko; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage did not restore to its record" >> "$O"; return 1; }
  echo "BP stage restored to its record $(date +%T)" >> "$O"
}
main "$@"; exit $?
