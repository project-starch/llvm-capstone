#!/bin/bash
# R-55: the acceptance chain's four pinned firmwares rebuilt with the fix, each beside an unpatched CONTROL rebuild that
# must reproduce the pinned image (else the recipe, not the fix, is what differs). Outputs under $P.
P=/tmp/capstone/r55-fix/fw; FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
XC=$FB/build-fpga/host/bin/riscv64-buildroot-linux-gnu-; CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
WW=/tmp/capstone/wt-wrap-supcall
restore() { git -C $WW checkout -q wrapper/supcall-fpga 2>/dev/null; }
h() { sha256sum "$1" | cut -c1-12; }
main() {
  set -u
  trap restore EXIT
  rm -rf $P; mkdir -p $P
  # 1. s06agg: the board line (shared opensbi-custom = 49b0f93 + monitor 2dcd3a5) on the pinned image's own payload.
  #    Control: /tmp/capstone/fpb-diag/fw-ctl.bin (equals pinned 2f49d5b257f7 but the payload-end word and 16 pad bytes).
  python3 -c "d=open('/tmp/capstone/s06agg-776/fw_payload.bin','rb').read(); open('$P/s06agg.Image','wb').write(d[0x200000:])"
  rsync -a --exclude build --exclude .git $FB/build-fpga/build/opensbi-custom/ $P/s06agg-fix/
  python3 /tmp/capstone/r55-fix/apply.py $P/s06agg-fix/lib/sbi/sbi_capstone_dom.c || return 1
  $CC --abi capstone $P/s06agg-fix/lib/sbi/sbi_capstone_dom.c -- -I$P/s06agg-fix/lib/sbi/capstone-sbi -D__riscv_xlen=64 \
    -DCAPSTONE_TARGET_FPGA > $P/s06agg-fix/lib/sbi/sbi_capstone_dom.c.S 2> $P/s06agg-fix.cc.err || return 1
  make -s -C $P/s06agg-fix -j8 PLATFORM=fpga/ariane CROSS_COMPILE=$XC LINUX_PAYLOAD=1 FW_PAYLOAD_PATH=$P/s06agg.Image \
    FW_FDT_PATH=$FB/build-fpga/images/caplifive.dtb FW_PAYLOAD_FDT_PATH=$FB/build-fpga/images/caplifive.dtb > $P/s06agg-fix.make.log 2>&1 || return 1
  cp $P/s06agg-fix/build/platform/fpga/ariane/firmware/fw_payload.bin $P/fw-s06agg-fix.bin
  echo "s06agg: control $(h /tmp/capstone/fpb-diag/fw-ctl.bin) (pinned 2f49d5b257f7, recorded equal but the payload-end word) fix $(h $P/fw-s06agg-fix.bin)"
  # 2. C5u: the archived tree of fw-c5u (d07761ded4ce); the fix is the assembly-level patch (asm-patch.py).
  for v in ctl fix; do
    rsync -a --exclude build /tmp/capstone/mon-c0/fw-c5u/opensbi/ $P/c5u-$v/
    [ $v = fix ] && { python3 /tmp/capstone/r55-fix/asm-patch.py $P/c5u-$v/lib/sbi/sbi_capstone_dom.c.S || return 1; }
    touch $P/c5u-$v/lib/sbi/*.c.S
    make -s -C $P/c5u-$v -j8 PLATFORM=fpga/ariane CROSS_COMPILE=$XC FW_PAYLOAD_PATH=/tmp/capstone/mon-c0/fw-c5u/images/Image \
      FW_FDT_PATH=/tmp/capstone/mon-c0/fw-c5u/images/caplifive.dtb FW_PAYLOAD_FDT_PATH=/tmp/capstone/mon-c0/fw-c5u/images/caplifive.dtb \
      > $P/c5u-$v.make.log 2>&1 || { echo "c5u $v make failed"; tail -3 $P/c5u-$v.make.log; return 1; }
    cp $P/c5u-$v/build/platform/fpga/ariane/firmware/fw_payload.bin $P/fw-c5u-$v.bin
  done
  echo "c5u: control $(h $P/fw-c5u-ctl.bin) (pinned d07761ded4ce) fix $(h $P/fw-c5u-fix.bin)"
  # 3./4. boot C+F and boot W: build-fpga-fw.sh on the wrapper/supcall-fpga worktree, control at 882892f, fix at d7489e3.
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET"
  for v in ctl fix; do
    if [ $v = ctl ]; then git -C $WW checkout -q --detach 882892f || return 1; else git -C $WW checkout -q wrapper/supcall-fpga || return 1; fi
    echo "  wrapper for $v: $(git -C $WW log --oneline -1 | cut -c1-60)"
    FW_IMAGE_DIR=/tmp/capstone/b0/fpga-image-fpb bash /tmp/capstone/b4/build-fpga-fw.sh $P/cf-$v "$D" > $P/cf-$v.log 2>&1 || { echo "cf $v failed"; tail -3 $P/cf-$v.log; return 1; }
    FW_IMAGE_DIR=/tmp/capstone/b0/fpga-image-cw bash /tmp/capstone/b4/build-fpga-fw.sh $P/w-$v "$D" > $P/w-$v.log 2>&1 || { echo "w $v failed"; tail -3 $P/w-$v.log; return 1; }
  done
  echo "bootCF: control $(h $P/cf-ctl/fw_payload.bin) (pinned abdb54361c4f) fix $(h $P/cf-fix/fw_payload.bin)"
  echo "bootW:  control $(h $P/w-ctl/fw_payload.bin) (pinned ff1e132b1084) fix $(h $P/w-fix/fw_payload.bin)"
  grep -h "gate:" $P/cf-fix.log $P/w-fix.log
  echo BUILD_ALL_DONE
}
main "$@"; exit $?
