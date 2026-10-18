#!/bin/bash
# vp2 arena-pin Tier 1 boot: as r13's build (monitor 007d81d) with module 3005ed52e860c7af (arena-pin-1 on a2a0b2f4, arena_pin off by
# default). The stage's suite is boot 4b's and is not run (Tier 1 alone).
#
# Ends by itself.
main() {
  set -u
  local T=/tmp/capstone/tier1 L=/tmp/capstone/tier1/BUILD-AP1.log I=/tmp/capstone/b0/fpga-image-tier1
  local MON=/tmp/capstone/csbi-retry WRAP=/tmp/capstone/wt-wrap-exact FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
  echo "TIER12-BUILD-AP1 START $(date +%T)" >> $L
  [ "$(git -C $MON rev-parse --short=7 HEAD)" = 007d81d ] && [ -z "$(git -C $MON status --porcelain)" ] || { echo "STOP: the monitor worktree is not a clean 007d81d" >> $L; return 1; }
  [ "$(git -C $WRAP rev-parse --short=7 HEAD)" = 1269c77 ] || { echo "STOP: the wrapper is not 1269c77" >> $L; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage does not match stage.SHA256SUMS" >> $L; return 1; }
  [ -e $T/fw3ap1 ] && { echo "STOP: $T/fw3ap1 exists" >> $L; return 1; }
  [ -e ${I}-be ] || cp -a $I ${I}-be
  echo "shared buildroot $(git -C $FB rev-parse --short=12 HEAD); linux $(git -C $FB/build-fpga/build/linux-6.4.14 rev-parse --short=12 HEAD 2>/dev/null || echo n/a); the be image kept in ${I}-be (Image $(sha256sum ${I}-be/Image | cut -c1-16))" >> $L
  bash /tmp/capstone/b0/b0-bake-tier1.sh > $T/bake-ap1.out 2>&1; local rc=$?
  echo "BAKE rc=$rc $(date +%T): $(grep -E 'saved Image|BAKE_DONE|FAILED|WRONG' $T/bake-ap1.out | tr '\n' ' ')" >> $L
  [ $rc -eq 0 ] && grep -q BAKE_DONE $T/bake-ap1.out || { echo "STOP: bake failed" >> $L; return 1; }
  grep -q "cpio test-domains/vm/capstone_vm.ko: present" $T/bake-ap1.out || { echo "STOP: the bake's cpio check does not show the vm module present" >> $L; return 1; }
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY"
  FW_WRAP=$WRAP FW_MON=$MON FW_IMAGE_DIR=$I nice -n 10 bash /tmp/capstone/r55-table/build-fpga-fw.sh $T/fw3ap1 "$D" > $T/fw3ap1.log 2>&1; rc=$?
  echo "FW rc=$rc $(date +%T): $(tail -1 $T/fw3ap1.log)" >> $L
  [ $rc -eq 0 ] && [ -f $T/fw3ap1/fw_payload.bin ] || { echo "STOP: firmware build failed" >> $L; return 1; }
  grep -q "^gate: PASS" $T/fw3ap1.log || { echo "STOP: the dom_stack gate did not pass" >> $L; return 1; }
  grep -q "defs \[.*-DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY\].*monitor 007d81d" $T/fw3ap1.log || { echo "STOP: the firmware's summary does not name the defines and monitor 007d81d" >> $L; return 1; }
  local X=$T/fw3ap1/opensbi/lib/sbi/capstone-sbi
  echo "extracted monitor: c $(sha256sum $X/sbi_capstone.c | cut -c1-16) S $(sha256sum $X/sbi_capstone.S | cut -c1-16) csmint $(sha256sum $X/capstone_csmint.S | cut -c1-16) h $(sha256sum $X/sbi_capstone.h | cut -c1-16)" >> $L
  local sz; sz=$(stat -c %s $T/fw3ap1/fw_payload.bin)
  echo "fw_payload $(sha256sum $T/fw3ap1/fw_payload.bin | cut -c1-16) $sz B; Image $(sha256sum $I/Image | cut -c1-16)" >> $L
  [ $sz -le $((20*1024*1024)) ] || echo "WARN: firmware $sz B is over the 20 MB working budget (32 MB hard cap)" >> $L
  local elf; elf=$(find $T/fw3ap1/opensbi/build -name 'fw_payload.elf' | head -1)
  [ -n "$elf" ] && echo "symbols in the firmware ELF: csmint_dispatch $(nm "$elf" 2>/dev/null | grep -c ' csmint_dispatch$'), _inject_s_trap $(nm "$elf" 2>/dev/null | grep -c ' _inject_s_trap$')" >> $L
  echo TIER12_BUILD_AP1_DONE >> $L
}
main "$@"; exit
