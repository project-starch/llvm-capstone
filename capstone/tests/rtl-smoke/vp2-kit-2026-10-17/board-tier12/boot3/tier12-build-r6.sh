#!/bin/bash
# vp2 Tier 1+2 boot r6: bake BL's image plus the re-staged /test-domains/vm (the module back to the compiler lane's
# 0cc4760d21fde17a), then build the firmware by BL's recipe on capstone-sbi vp2/fetch-retry 007d81d (the CPMX diagnostic
# plus the fetch-retry workaround; wrapper 1269c77, BL's defines + CAPSTONE_VP2_CSMINT + CAPSTONE_CPMX_DIAG +
# CAPSTONE_VP2_FETCH_RETRY). Run after stage-module.sh. The r5 image directory is copied aside first. Ends by itself.
main() {
  set -u
  local T=/tmp/capstone/tier1 L=/tmp/capstone/tier1/BUILD-R6.log I=/tmp/capstone/b0/fpga-image-tier1
  local MON=/tmp/capstone/csbi-retry WRAP=/tmp/capstone/wt-wrap-exact FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
  echo "TIER12-BUILD-R6 START $(date +%T)" >> $L
  [ "$(git -C $MON rev-parse --short=7 HEAD)" = 007d81d ] && [ -z "$(git -C $MON status --porcelain)" ] || { echo "STOP: the monitor worktree is not a clean 007d81d" >> $L; return 1; }
  [ "$(git -C $WRAP rev-parse --short=7 HEAD)" = 1269c77 ] || { echo "STOP: the wrapper is not 1269c77" >> $L; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage does not match stage.SHA256SUMS" >> $L; return 1; }
  [ -e $T/fw3r6 ] && { echo "STOP: $T/fw3r6 exists" >> $L; return 1; }
  [ -e ${I}-r5 ] || cp -a $I ${I}-r5
  echo "shared buildroot $(git -C $FB rev-parse --short=12 HEAD); linux $(git -C $FB/build-fpga/build/linux-6.4.14 rev-parse --short=12 HEAD 2>/dev/null || echo n/a); the r5 image kept in ${I}-r5 (Image $(sha256sum ${I}-r5/Image | cut -c1-16))" >> $L
  bash /tmp/capstone/b0/b0-bake-tier1.sh > $T/bake-r6.out 2>&1; local rc=$?
  echo "BAKE rc=$rc $(date +%T): $(grep -E 'saved Image|BAKE_DONE|FAILED|WRONG' $T/bake-r6.out | tr '\n' ' ')" >> $L
  [ $rc -eq 0 ] && grep -q BAKE_DONE $T/bake-r6.out || { echo "STOP: bake failed" >> $L; return 1; }
  grep -q "cpio test-domains/vm/capstone_vm.ko: present" $T/bake-r6.out || { echo "STOP: the bake's cpio check does not show the vm module present" >> $L; return 1; }
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY"
  FW_WRAP=$WRAP FW_MON=$MON FW_IMAGE_DIR=$I nice -n 10 bash /tmp/capstone/r55-table/build-fpga-fw.sh $T/fw3r6 "$D" > $T/fw3r6.log 2>&1; rc=$?
  echo "FW rc=$rc $(date +%T): $(tail -1 $T/fw3r6.log)" >> $L
  [ $rc -eq 0 ] && [ -f $T/fw3r6/fw_payload.bin ] || { echo "STOP: firmware build failed" >> $L; return 1; }
  grep -q "^gate: PASS" $T/fw3r6.log || { echo "STOP: the dom_stack gate did not pass" >> $L; return 1; }
  grep -q "defs \[.*-DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY\].*monitor 007d81d" $T/fw3r6.log || { echo "STOP: the firmware's summary does not name the defines and monitor 007d81d" >> $L; return 1; }
  local X=$T/fw3r6/opensbi/lib/sbi/capstone-sbi
  echo "extracted monitor: c $(sha256sum $X/sbi_capstone.c | cut -c1-16) S $(sha256sum $X/sbi_capstone.S | cut -c1-16) csmint $(sha256sum $X/capstone_csmint.S | cut -c1-16) h $(sha256sum $X/sbi_capstone.h | cut -c1-16)" >> $L
  local sz; sz=$(stat -c %s $T/fw3r6/fw_payload.bin)
  echo "fw_payload $(sha256sum $T/fw3r6/fw_payload.bin | cut -c1-16) $sz B; Image $(sha256sum $I/Image | cut -c1-16)" >> $L
  [ $sz -le $((20*1024*1024)) ] || echo "WARN: firmware $sz B is over the 20 MB working budget (32 MB hard cap)" >> $L
  local elf; elf=$(find $T/fw3r6/opensbi/build -name 'fw_payload.elf' | head -1)
  [ -n "$elf" ] && echo "symbols in the firmware ELF: csmint_dispatch $(nm "$elf" 2>/dev/null | grep -c ' csmint_dispatch$'), _inject_s_trap $(nm "$elf" 2>/dev/null | grep -c ' _inject_s_trap$')" >> $L
  echo TIER12_BUILD_R6_DONE >> $L
}
main "$@"; exit
