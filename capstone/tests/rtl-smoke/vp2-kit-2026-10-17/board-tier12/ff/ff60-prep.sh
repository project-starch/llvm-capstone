#!/bin/bash
# The fetch-face sweep, prepared (2026-10-20; pre-registered in tier3-smoke/README.md before the boot): ff-prep.sh with the
# case ff-sweep (board-tier12/ff/vmrun-ff60.case: INNER 300, node_flat_records 65536) in place of ff-pair, output names
# image-ff60 / fw-ff60 / ff60-stage-keep / bake-ff60.out, and the firmware on the fetch-slot monitor (capstone-sbi 9697173,
# the D defines plus CAPSTONE_S23_FETCH_SLOTS) instead of f13dbc5. Usage: ff60-prep.sh <log>. Ends by itself.
# BP_T, BP_X, BP_BAKE, BP_FW stub for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${BP_T:-/tmp/capstone/tier1} X=${BP_X:-/tmp/capstone/s26} I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${BP_BAKE:-bash $S/b0-bake-ff.sh} FW=${BP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local A=$T/stage-attic/rev8-1019-1626 K=$X/ff60-stage-keep M=/mnt/hdd/$USER/capstone/modfix/srz-board-flat/module/capstone_vm.ko
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000 -DCAPSTONE_S23_REFUSAL_READ -DCAPSTONE_S23_DATA_RETRY -DCAPSTONE_S23_FETCH_SLOTS"
  [ "$(sha256sum $M | cut -c1-16)" = 607164b9c1407c7c ] || { echo "BP STOP: the srz module is not 607164b9" >> "$O"; return 1; }
  [ "$(sha256sum $X/image-rev7/Image | cut -c1-16)" = dcfc2e15bf1bbd77 ] || { echo "BP STOP: image-rev7 is not dcfc2e15" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-frs rev-parse --short=7 HEAD)" = 9697173 ] && [ -z "$(git -C /tmp/capstone/csbi-frs status --porcelain)" ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "BP STOP: monitor or wrapper not at 9697173 / aab63d5" >> "$O"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage is not its record" >> "$O"; return 1; }
  [ -e $X/image-ff60 ] && { echo "BP STOP: $X/image-ff60 exists" >> "$O"; return 1; }
  [ "$(sha256sum /mnt/hdd/$USER/capstone/fetchface/revoke-loop.dom | cut -c1-16)" = 252c16ca3c63cec5 ] && [ "$(sha256sum /mnt/hdd/$USER/capstone/fetchface/revoke-loop-ops.dom | cut -c1-16)" = cf8be80bec87bc03 ] \
    || { echo "BP STOP: the fetch-face images are not as recorded" >> "$O"; return 1; }
  [ -e $K ] && { echo "BP STOP: $K exists" >> "$O"; return 1; }
  mkdir -p $K/vm; cp -p $T/stage.SHA256SUMS $K/; cp -p $T/stage/vmrun.sh $K/; cp -p $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh $T/stage/vm/contract-dense.dom $K/vm/
  # the pair's stage: rev7's vm-suite-inner.sh, no contract-dense.dom, rev7's vm-contract.sh, checked against rev7's record
  cp -p $A/vm/vm-suite-inner.sh $A/vm/vm-contract.sh $T/stage/vm/; rm -f "${T:?}"/stage/vm/contract-dense.dom
  ( cd $T/stage && sha256sum -c --quiet $X/stage-rev7.SHA256SUMS ) && [ -z "$(cd $T/stage && find . -type f | sed 's|^\./||' | sort | comm -23 - <(awk '{print $2}' $X/stage-rev7.SHA256SUMS | sort))" ] \
    || { echo "BP STOP: the stage did not become the pair's (rev7's) record" >> "$O"; restore; return 1; }
  # The staged module is read-only (mode 0444), so it is replaced by removal, and the result is checked by hash: a failed
  # copy must never bake the old module under the new label (the dry run's finding, 2026-10-20).
  cp -p $T/stage/vm/capstone_vm.ko $K/vm/capstone_vm.ko.39d8492d && rm -f "${T:?}"/stage/vm/capstone_vm.ko && cp $M $T/stage/vm/capstone_vm.ko
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = 607164b9c1407c7c ] || { echo "BP STOP: the staged module is not srz after the swap" >> "$O"; restore; return 1; }
  python3 - $T/stage/vmrun.sh $S/vmrun-ff60.case <<'PY' || { echo "BP STOP: vmrun.sh did not take the case" >> "$O"; restore; return 1; }
import sys
p, c = sys.argv[1], open(sys.argv[2]).read(); s = open(p).read()
o = '  *) exec /test-domains/b0run.sh "$@" ;;'
assert s.count(o) == 1 and "ff-sweep)" not in s
open(p, "w").write(s.replace(o, c + o))
PY
  sh -n $T/stage/vmrun.sh || { echo "BP STOP: vmrun.sh does not parse" >> "$O"; restore; return 1; }
  echo "BP BAKE $(date +%T): rev7's stage, capstone_vm.ko $(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-ff60.out 2>&1; local rc=$?
  restore || return 1
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-ff60.out && grep -q "cpio test-domains/vm/capstone_vm.ko: present" $X/bake-ff60.out && grep -q "cpio test-domains/vm/revoke-loop-ops.dom: present" $X/bake-ff60.out \
    && ! grep -q "cpio test-domains/vm/contract-dense.dom: present" $X/bake-ff60.out \
    || { echo "BP STOP: the bake failed (rc $rc): $(grep -E 'BAKE_DONE|FAILED|STOP' $X/bake-ff60.out | tail -2 | tr '\n' ' ')" >> "$O"; return 1; }
  mkdir -p $X/image-ff60; cp $I/Image $I/caplifive.dtb $X/image-ff60/; chmod a-w $X/image-ff60/*
  echo "BP IMAGE ff60 $(sha256sum $X/image-ff60/Image | cut -c1-16) dtb $(sha256sum $X/image-ff60/caplifive.dtb | cut -c1-16)" >> "$O"
  [ -e $X/fw-ff60 ] && { echo "BP STOP: $X/fw-ff60 exists" >> "$O"; return 1; }
  FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-frs FW_IMAGE_DIR=$X/image-ff60 nice -n 10 $FW $X/fw-ff60 "$D" > $X/fw-ff60.log 2>&1
  grep -q "^gate: PASS" $X/fw-ff60.log && [ -f $X/fw-ff60/fw_payload.bin ] || { echo "BP STOP: firmware ff60 failed" >> "$O"; return 1; }
  echo "BP FW ff60 $(sha256sum $X/fw-ff60/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-ff60/fw_payload.bin) B on image $(sha256sum $X/image-ff60/Image | cut -c1-16)" >> "$O"
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
