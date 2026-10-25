#!/bin/bash
# Private FPGA firmware for the FP-respin Linux hang (2026-10-15). Two OpenSBI copies from the shared build dir's
# opensbi-custom (read only): CTL (unchanged) and MTIME (cap_env_init sets the mtime capability's cursor to
# SBI_MTIME_ADDR and reports its decoded base and cursor). Both link the PINNED firmware's own payload
# (fw_2f49d5b257f7 [0x200000:], kernel #894 + its initramfs with k800/s06agg) and today's caplifive.dtb.
# Gate: CTL's regenerated .c.S must equal the shared one byte for byte, and CTL's fw_payload.bin must equal the
# pinned image outside known build-stamp bytes -- else the recipe does not reproduce the board build.
main() {
  set -u
  local P=/tmp/capstone/fpb-diag SRC=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga
  local CC=$HOME/dev/llvm-capstone/capstone/capstone-c PIN=/tmp/capstone/s06agg-776/fw_payload.bin
  local XC=$SRC/host/bin/riscv64-buildroot-linux-gnu-
  python3 -c "d=open('$PIN','rb').read(); open('$P/payload.Image','wb').write(d[0x200000:])" || return 1
  cp -f $SRC/images/caplifive.dtb $P/caplifive.dtb || return 1
  for v in ctl mtime; do
    rm -rf $P/$v; mkdir -p $P/$v
    rsync -a --exclude build --exclude .git $SRC/build/opensbi-custom/ $P/$v/ || return 1
  done
  python3 "$(dirname "$0")/patch.py" $P/mtime/lib/sbi/sbi_capstone_dom.c || return 1
  for v in ctl mtime; do
    ( cd $CC && cargo run -q -- --abi capstone $P/$v/lib/sbi/sbi_capstone_dom.c -- -I$P/$v/lib/sbi/capstone-sbi \
        -D__riscv_xlen=64 -DCAPSTONE_TARGET_FPGA > $P/$v/lib/sbi/sbi_capstone_dom.c.S.new 2> $P/$v.capstone-c.log ) \
      || { echo "capstone-c failed for $v"; return 1; }
  done
  cmp $P/ctl/lib/sbi/sbi_capstone_dom.c.S.new $SRC/build/opensbi-custom/lib/sbi/sbi_capstone_dom.c.S \
    && echo "GATE1 OK: regenerated control .c.S == shared .c.S" || { echo "GATE1 FAIL: recipe does not reproduce the shared .c.S"; return 1; }
  for v in ctl mtime; do
    mv -f $P/$v/lib/sbi/sbi_capstone_dom.c.S.new $P/$v/lib/sbi/sbi_capstone_dom.c.S
    make -C $P/$v -j8 PLATFORM=fpga/ariane CROSS_COMPILE=$XC LINUX_PAYLOAD=1 FW_PAYLOAD_PATH=$P/payload.Image \
      FW_FDT_PATH=$P/caplifive.dtb FW_PAYLOAD_FDT_PATH=$P/caplifive.dtb > $P/$v.make.log 2>&1 \
      || { echo "make failed for $v"; tail -5 $P/$v.make.log; return 1; }
    cp -f $P/$v/build/platform/fpga/ariane/firmware/fw_payload.bin $P/fw-$v.bin
    echo "$v: $(sha256sum $P/fw-$v.bin | cut -c1-12) $(stat -c %s $P/fw-$v.bin) B"
  done
  python3 - "$PIN" "$P/fw-ctl.bin" "$P/fw-mtime.bin" <<'PY'
import sys
pin, ctl, mt = (open(f, "rb").read() for f in sys.argv[1:4])
def runs(a, b):
    d = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
    out, s = [], None
    for i in d:
        if s is None or i != p + 1:
            if s is not None: out.append((s, p))
            s = i
        p = i
    if s is not None: out.append((s, p))
    return len(a) == len(b), len(d), out
for name, (a, b) in (("ctl vs pinned", (pin, ctl)), ("mtime vs ctl", (ctl, mt))):
    same, n, r = runs(a, b)
    print(f"{name}: same length {same}, {n} bytes differ in {len(r)} runs: " + ", ".join(f"{s:#x}-{e:#x}" for s, e in r[:12]))
PY
}
main "$@"; exit $?
