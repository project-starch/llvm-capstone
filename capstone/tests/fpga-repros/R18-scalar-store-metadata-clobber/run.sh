#!/usr/bin/env bash
# R-18: a plain scalar store in the UPPER half of a 16-byte cache row is silently zeroed.
#
# The images are FROZEN and checksummed on purpose. The effect depends entirely on where the -O0
# allocator places the accumulator, so rebuilding can move it out of the affected slot and cure the
# fault -- which is how several instruments in this investigation destroyed what they measured.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1

echo "== verifying frozen images =="
( cd "$D" && sha256sum -c SHA256SUMS ) || { echo "CHECKSUM MISMATCH -- do not trust a run"; exit 1; }

echo
echo "== layouts: the accumulator's row offset is the whole variable =="
python3 capstone/tests/runtime-qemu/silicon-ladder/extract-frame-layout.py \
        "$D"/src/c8.dom "$D"/src/c0.dom "$D"/src/sn8.dom "$D"/src/sn0.dom

echo
echo "== expected on silicon =="
echo "   c0   0x04090240  (p=64 k=9 qc=576)  CORRECT   -- accumulator in the LOWER half"
echo "   c8   0x04090237  (p=64 k=9 qc=567)  DEFECT    -- accumulator in the UPPER half"
echo "   sn0  1000576                        CORRECT   -- sentinel start, lower half"
echo "   sn8  567                            DEFECT    -- sentinel start, upper half"
echo
echo "   sn8 is decisive: it starts at 1,000,000. Lost increments would give 1000567."
echo "   Returning 567 means the slot was OVERWRITTEN and counted up from zero."

: "${FPGA_URL:?set FPGA_URL from the secrets file (never commit or echo it)}"
: "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
: "${FPGA_BITSTREAM:=caplifive_65536_nodes.bit}"; export FPGA_BITSTREAM

O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
echo
echo "== staging the four frozen images (plus the known-good control k800) =="
cp -f "$D"/src/c0.dom "$D"/src/c8.dom "$D"/src/sn0.dom "$D"/src/sn8.dom "$O/" || exit 1
mkdir -p "$T" && cp -f "$D"/src/c0.dom "$D"/src/c8.dom "$D"/src/sn0.dom "$D"/src/sn8.dom "$T/"
export LADDER_FPGA_DIR=${LADDER_FPGA_DIR:-/tmp/capstone/ladder-fpga}; mkdir -p "$LADDER_FPGA_DIR"
python3 -c "print((64<<20)|(9<<16)|576)" > "$LADDER_FPGA_DIR/c0.oracle"
python3 -c "print((64<<20)|(9<<16)|576)" > "$LADDER_FPGA_DIR/c8.oracle"
echo 1000576 > "$LADDER_FPGA_DIR/sn0.oracle"
echo 1000576 > "$LADDER_FPGA_DIR/sn8.oracle"

echo
echo "NOTE: rebuild the firmware before running, in this order, or the OLD initramfs ships:"
echo "  make build LINUX_PAYLOAD=1 A=linux-rebuild  ...   # FIRST"
echo "  make build LINUX_PAYLOAD=1 A=opensbi-rebuild ...   # THEN"
echo "  then confirm the firmware HASH changed -- never the size."
echo
echo "== board run: control first, then lower/upper pairs =="
( cd capstone/tests/rtl-smoke \
  && BAKED_RUNGS="k800 c0 c8 sn8" BAKED_TIMEOUT=150 \
     python3 -m fpga_driver.run_baked_rungs_fpga )

echo
echo "PASS(defect present) = c0 correct AND c8 returns qc=567 AND sn8 returns 567."
echo "If c8 and sn8 both return the correct value, the defect did NOT reproduce -- check the"
echo "bitstream and confirm the accumulator still lands in the upper half of its row."
