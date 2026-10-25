#!/bin/bash
# vp2 Tier 1/2 boot 3: bake BL's image plus the re-staged /test-domains/vm (b0-bake-tier1.sh, unchanged: shared buildroot,
# machine lock, EXIT-trap restore, cpio checked present/absent by hash), then build the firmware by BL's recipe on the
# combined monitor capstone-sbi vp2/tier1-monitor 1e9451a (wrapper 1269c77, BL's defines + CAPSTONE_VP2_CSMINT).
# Run after stage-boot3.sh. Boot 2's image directory is copied aside first. Ends by itself.
main() {
  set -u
  local T=/tmp/capstone/tier1 L=/tmp/capstone/tier1/BUILD3.log I=/tmp/capstone/b0/fpga-image-tier1
  local MON=/tmp/capstone/csbi-t1mon WRAP=/tmp/capstone/wt-wrap-exact FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
  echo "TIER1-BUILD3 START $(date +%T)" >> $L
  [ "$(git -C $MON rev-parse --short=7 HEAD)" = 1e9451a ] && [ -z "$(git -C $MON status --porcelain)" ] || { echo "STOP: the monitor worktree is not a clean 1e9451a" >> $L; return 1; }
  [ "$(git -C $WRAP rev-parse --short=7 HEAD)" = 1269c77 ] || { echo "STOP: the wrapper is not 1269c77" >> $L; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage does not match stage.SHA256SUMS" >> $L; return 1; }
  [ -e $T/fw3 ] && { echo "STOP: $T/fw3 exists" >> $L; return 1; }
  [ -e ${I}-boot2 ] || cp -a $I ${I}-boot2
  echo "shared buildroot $(git -C $FB rev-parse --short=12 HEAD); linux $(git -C $FB/build-fpga/build/linux-6.4.14 rev-parse --short=12 HEAD 2>/dev/null || echo n/a); boot 2 image kept in ${I}-boot2 (Image $(sha256sum ${I}-boot2/Image | cut -c1-16))" >> $L
  bash /tmp/capstone/b0/b0-bake-tier1.sh > $T/bake3.out 2>&1; local rc=$?
  echo "BAKE rc=$rc $(date +%T): $(grep -E 'saved Image|BAKE_DONE|FAILED|WRONG' $T/bake3.out | tr '\n' ' ')" >> $L
  [ $rc -eq 0 ] && grep -q BAKE_DONE $T/bake3.out || { echo "STOP: bake failed" >> $L; return 1; }
  grep -q "cpio test-domains/vm/capstone_vm.ko: present" $T/bake3.out || { echo "STOP: the bake's cpio check does not show the vm module present" >> $L; return 1; }
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT"
  FW_WRAP=$WRAP FW_MON=$MON FW_IMAGE_DIR=$I nice -n 10 bash /tmp/capstone/r55-table/build-fpga-fw.sh $T/fw3 "$D" > $T/fw3.log 2>&1; rc=$?
  echo "FW rc=$rc $(date +%T): $(tail -1 $T/fw3.log)" >> $L
  [ $rc -eq 0 ] && [ -f $T/fw3/fw_payload.bin ] || { echo "STOP: firmware build failed" >> $L; return 1; }
  grep -q "^gate: PASS" $T/fw3.log || { echo "STOP: the dom_stack gate did not pass" >> $L; return 1; }
  grep -q "defs \[.*-DCAPSTONE_VP2_CSMINT\].*monitor 1e9451a" $T/fw3.log || { echo "STOP: the firmware's summary does not name the defines and monitor 1e9451a" >> $L; return 1; }
  local X=$T/fw3/opensbi/lib/sbi/capstone-sbi
  echo "extracted monitor: c $(sha256sum $X/sbi_capstone.c | cut -c1-16) S $(sha256sum $X/sbi_capstone.S | cut -c1-16) csmint $(sha256sum $X/capstone_csmint.S | cut -c1-16) h $(sha256sum $X/sbi_capstone.h | cut -c1-16)" >> $L
  local sz; sz=$(stat -c %s $T/fw3/fw_payload.bin)
  echo "fw_payload $(sha256sum $T/fw3/fw_payload.bin | cut -c1-16) $sz B; Image $(sha256sum $I/Image | cut -c1-16)" >> $L
  [ $sz -le $((20*1024*1024)) ] || echo "WARN: firmware $sz B is over the 20 MB working budget (32 MB hard cap)" >> $L
  local elf; elf=$(find $T/fw3/opensbi/build -name 'fw_payload.elf' | head -1)
  [ -n "$elf" ] && echo "symbols in the firmware ELF: csmint_dispatch $(nm "$elf" 2>/dev/null | grep -c ' csmint_dispatch$'), _inject_s_trap $(nm "$elf" 2>/dev/null | grep -c ' _inject_s_trap$')" >> $L
  echo TIER1_BUILD3_DONE >> $L
}
main "$@"; exit
