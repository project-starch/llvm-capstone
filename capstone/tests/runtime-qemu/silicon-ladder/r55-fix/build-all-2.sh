#!/bin/bash
# R-55 (both halves: wrapper df0e9a8/d7489e3 + monitor 6dad416/fc5e3c6): the re-run's four firmwares.
# s06agg: the board line on the pinned image's payload (control: /tmp/capstone/fpb-diag/fw-ctl.bin).
# boot C+F, boot W: build-fpga-fw.sh on the supcall worktrees (controls cf-ctl/w-ctl reproduce the pinned images).
# C5u: the current supcall line with C5u's own payload (the 2026-10-03 monitor cannot be regenerated: functional row).
P=/tmp/capstone/r55-fix/fw2; FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
XC=$FB/build-fpga/host/bin/riscv64-buildroot-linux-gnu-; CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
h() { sha256sum "$1" | cut -c1-12; }
main() {
  set -u
  rm -rf $P; mkdir -p $P
  [ "$(git -C /tmp/capstone/wt-wrap-supcall rev-parse --short=7 HEAD)" = d7489e3 ] && [ "$(git -C /tmp/capstone/wt-mon-supcall rev-parse --short=7 HEAD)" = fc5e3c6 ] \
    && [ "$(git -C /tmp/capstone/wt-mon-r55 rev-parse --short=7 HEAD)" = 6dad416 ] || { echo "worktree HEADs not as expected"; return 1; }
  python3 -c "d=open('/tmp/capstone/s06agg-776/fw_payload.bin','rb').read(); open('$P/s06agg.Image','wb').write(d[0x200000:])"
  rsync -a --exclude build --exclude .git $FB/build-fpga/build/opensbi-custom/ $P/s06agg/
  python3 /tmp/capstone/r55-fix/apply.py $P/s06agg/lib/sbi/sbi_capstone_dom.c || return 1
  rm -rf $P/s06agg/lib/sbi/capstone-sbi; mkdir -p $P/s06agg/lib/sbi/capstone-sbi
  git -C /tmp/capstone/wt-mon-r55 archive HEAD | tar -x -C $P/s06agg/lib/sbi/capstone-sbi
  $CC --abi capstone $P/s06agg/lib/sbi/sbi_capstone_dom.c -- -I$P/s06agg/lib/sbi/capstone-sbi -D__riscv_xlen=64 -DCAPSTONE_TARGET_FPGA \
    > $P/s06agg/lib/sbi/sbi_capstone_dom.c.S 2> $P/s06agg.cc.err || return 1
  make -s -C $P/s06agg -j8 PLATFORM=fpga/ariane CROSS_COMPILE=$XC LINUX_PAYLOAD=1 FW_PAYLOAD_PATH=$P/s06agg.Image \
    FW_FDT_PATH=$FB/build-fpga/images/caplifive.dtb FW_PAYLOAD_FDT_PATH=$FB/build-fpga/images/caplifive.dtb > $P/s06agg.make.log 2>&1 || return 1
  cp $P/s06agg/build/platform/fpga/ariane/firmware/fw_payload.bin $P/fw-s06agg.bin
  echo "s06agg $(h $P/fw-s06agg.bin)"
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET"
  FW_IMAGE_DIR=/tmp/capstone/b0/fpga-image-fpb bash /tmp/capstone/b4/build-fpga-fw.sh $P/cf "$D" > $P/cf.log 2>&1 || { echo "cf failed"; tail -3 $P/cf.log; return 1; }
  FW_IMAGE_DIR=/tmp/capstone/b0/fpga-image-cw bash /tmp/capstone/b4/build-fpga-fw.sh $P/w "$D" > $P/w.log 2>&1 || { echo "w failed"; tail -3 $P/w.log; return 1; }
  FW_IMAGE_DIR=/tmp/capstone/mon-c0/fw-c5u/images bash /tmp/capstone/b4/build-fpga-fw.sh $P/c5u "$D" > $P/c5u.log 2>&1 || { echo "c5u failed"; tail -3 $P/c5u.log; return 1; }
  echo "bootCF $(h $P/cf/fw_payload.bin)  bootW $(h $P/w/fw_payload.bin)  c5u $(h $P/c5u/fw_payload.bin)"
  grep -h "gate:" $P/cf.log $P/w.log $P/c5u.log
  python3 - <<'PY'
pairs = [("s06agg", "/tmp/capstone/fpb-diag/fw-ctl.bin", "/tmp/capstone/r55-fix/fw2/fw-s06agg.bin"),
         ("bootCF", "/tmp/capstone/r55-fix/fw/cf-ctl/fw_payload.bin", "/tmp/capstone/r55-fix/fw2/cf/fw_payload.bin"),
         ("bootW", "/tmp/capstone/r55-fix/fw/w-ctl/fw_payload.bin", "/tmp/capstone/r55-fix/fw2/w/fw_payload.bin")]
for name, c, f in pairs:
    a, b = open(c, "rb").read(), open(f, "rb").read()
    d = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
    print(f"{name:7s} vs control: same length {len(a) == len(b)}; {len(d)} bytes differ in [{d[0]:#x}, {d[-1]:#x}]; payload identical: {a[0x200000:] == b[0x200000:]}")
c5 = open("/tmp/capstone/r55-fix/fw2/c5u/fw_payload.bin", "rb").read(); orig = open("/tmp/capstone/mon-c0/fw-c5u/fw_payload.bin", "rb").read()
print(f"c5u     payload identical to d07761ded4ce's: {c5[0x200000:] == orig[0x200000:]}")
PY
  echo BUILD2_DONE
}
main "$@"; exit $?
