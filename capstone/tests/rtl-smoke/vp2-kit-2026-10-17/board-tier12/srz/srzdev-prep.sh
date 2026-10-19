#!/bin/bash
# dev's srevroot-zero module (90bf036f, dev 178e005899b0, banner srevroot-zero-2): its own Tier 1 image and firmware, prepared
# (2026-10-20) as srz-prep.sh did for 607164b9, the old arm's firmware reused (t1-old). srz-prep.sh's header follows.
# The srz module's Tier 1 pair, prepared (2026-10-20; pre-registered in tier3-smoke/README.md before the boots):
#  1. image-rev7-srzdev: rev7's Tier 1 stage (rev7's record, as image-rev7 dcfc2e15bf1bbd77 was baked) with ONLY
#     vm/capstone_vm.ko swapped for the compiler lane's srevroot-zero module 90bf036f59da1231 (lane module-srevroot-zero);
#     the stage restored exactly to its record afterwards;
#  2. two firmwares with the retry monitor line's recipe (capstone-sbi f13dbc5, wrapper aab63d5, the D kits' defines):
#     fw-t1-old on image-rev7 (module 39d8492d) and fw-t1-srz on image-rev7-srzdev.
# Usage: srz-prep.sh <log>. Ends by itself. BP_T, BP_X, BP_BAKE and BP_FW stub the stage root, the output root and the
# two external steps for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${BP_T:-/tmp/capstone/tier1} X=${BP_X:-/tmp/capstone/s26} I=/tmp/capstone/b0/fpga-image-tier1
  local BAKE=${BP_BAKE:-bash /tmp/capstone/b0/b0-bake-tier1.sh} FW=${BP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local A=$T/stage-attic/rev8-1019-1626 K=$X/srzdev-stage-keep M=/mnt/hdd/$USER/capstone/modfix/srzdev-board-flat/module/capstone_vm.ko
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000 -DCAPSTONE_S23_REFUSAL_READ -DCAPSTONE_S23_DATA_RETRY"
  [ "$(sha256sum $M | cut -c1-16)" = 90bf036f59da1231 ] || { echo "BP STOP: the module is not 90bf036f" >> "$O"; return 1; }
  [ "$(sha256sum $X/image-rev7/Image | cut -c1-16)" = dcfc2e15bf1bbd77 ] || { echo "BP STOP: image-rev7 is not dcfc2e15" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-drt rev-parse --short=7 HEAD)" = f13dbc5 ] && [ -z "$(git -C /tmp/capstone/csbi-drt status --porcelain)" ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "BP STOP: monitor or wrapper not at f13dbc5 / aab63d5" >> "$O"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage is not its record" >> "$O"; return 1; }
  [ -e $X/image-rev7-srzdev ] && { echo "BP STOP: $X/image-rev7-srzdev exists" >> "$O"; return 1; }
  [ -e $K ] && { echo "BP STOP: $K exists" >> "$O"; return 1; }
  mkdir -p $K/vm; cp -p $T/stage.SHA256SUMS $K/; cp -p $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh $T/stage/vm/contract-dense.dom $K/vm/
  # the pair's stage: rev7's vm-suite-inner.sh, no contract-dense.dom, rev7's vm-contract.sh, checked against rev7's record
  cp -p $A/vm/vm-suite-inner.sh $A/vm/vm-contract.sh $T/stage/vm/; rm -f "${T:?}"/stage/vm/contract-dense.dom
  ( cd $T/stage && sha256sum -c --quiet $X/stage-rev7.SHA256SUMS ) && [ -z "$(cd $T/stage && find . -type f | sed 's|^\./||' | sort | comm -23 - <(awk '{print $2}' $X/stage-rev7.SHA256SUMS | sort))" ] \
    || { echo "BP STOP: the stage did not become the pair's (rev7's) record" >> "$O"; restore; return 1; }
  # The staged module is read-only (mode 0444), so it is replaced by removal, and the result is checked by hash: a failed
  # copy must never bake the old module under the new label (the dry run's finding, 2026-10-20).
  cp -p $T/stage/vm/capstone_vm.ko $K/vm/capstone_vm.ko.39d8492d && rm -f "${T:?}"/stage/vm/capstone_vm.ko && cp $M $T/stage/vm/capstone_vm.ko
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = 90bf036f59da1231 ] || { echo "BP STOP: the staged module is not srz after the swap" >> "$O"; restore; return 1; }
  echo "BP BAKE $(date +%T): rev7's stage, capstone_vm.ko $(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-srzdev.out 2>&1; local rc=$?
  restore || return 1
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-srzdev.out && grep -q "cpio test-domains/vm/capstone_vm.ko: present" $X/bake-srzdev.out \
    && ! grep -q "cpio test-domains/vm/contract-dense.dom: present" $X/bake-srzdev.out \
    || { echo "BP STOP: the bake failed (rc $rc): $(grep -E 'BAKE_DONE|FAILED|STOP' $X/bake-srzdev.out | tail -2 | tr '\n' ' ')" >> "$O"; return 1; }
  mkdir -p $X/image-rev7-srzdev; cp $I/Image $I/caplifive.dtb $X/image-rev7-srzdev/; chmod a-w $X/image-rev7-srzdev/*
  echo "BP IMAGE srz $(sha256sum $X/image-rev7-srzdev/Image | cut -c1-16) dtb $(sha256sum $X/image-rev7-srzdev/caplifive.dtb | cut -c1-16)" >> "$O"
  local a im
  for a in t1-srzdev; do
    case $a in t1-old) im=$X/image-rev7;; t1-srzdev) im=$X/image-rev7-srzdev;; esac
    [ -e $X/fw-$a ] && { echo "BP STOP: $X/fw-$a exists" >> "$O"; return 1; }
    FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-drt FW_IMAGE_DIR=$im nice -n 10 $FW $X/fw-$a "$D" > $X/fw-$a.log 2>&1
    grep -q "^gate: PASS" $X/fw-$a.log && [ -f $X/fw-$a/fw_payload.bin ] || { echo "BP STOP: firmware $a failed" >> "$O"; return 1; }
    echo "BP FW $a $(sha256sum $X/fw-$a/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-$a/fw_payload.bin) B on image $(sha256sum $im/Image | cut -c1-16)" >> "$O"
  done
  echo "BP DONE $(date +%T)" >> "$O"
}
# today's stage back, exactly: the three kept files and the record, checked
restore() {
  cp -p $K/vm/vm-contract.sh $K/vm/vm-suite-inner.sh $K/vm/contract-dense.dom $T/stage/vm/; cp -p $K/stage.SHA256SUMS $T/stage.SHA256SUMS
  [ -f $K/vm/capstone_vm.ko.39d8492d ] && { rm -f "${T:?}"/stage/vm/capstone_vm.ko; cp -p $K/vm/capstone_vm.ko.39d8492d $T/stage/vm/capstone_vm.ko; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "BP STOP: the stage did not restore to its record" >> "$O"; return 1; }
  echo "BP stage restored to its record $(date +%T)" >> "$O"
}
main "$@"; exit $?
