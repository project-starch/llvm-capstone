#!/usr/bin/env bash
# Build both images, run the QEMU differential, then the board pair in ONE boot.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1

echo "== build uc (unmodified) and dp0 (uc + one dead never-called function) =="
SQLITE_STATIC_BUILTINS=0 DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_STAGE=0" \
  OUT_DIR=/tmp/capstone/uc  bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh \
  >/tmp/capstone/uc.log 2>&1 || { echo "uc build failed"; exit 1; }
SQLITE_STATIC_BUILTINS=0 CAPSTONE_DOMAIN_PAD=0 DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_STAGE=0" \
  OUT_DIR=/tmp/capstone/dp0 bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh \
  >/tmp/capstone/dp0.log 2>&1 || { echo "dp0 build failed"; exit 1; }
sha256sum /tmp/capstone/uc/sqlite_silicon.dom /tmp/capstone/dp0/sqlite_silicon.dom
echo "  (IMAGE-HASHES.txt records what these were; a mismatch means the toolchain moved --"
echo "   re-record it, that alone does not invalidate the repro)"

echo "== QEMU differential: BOTH must return obs=1517161237 =="
SHARE=/tmp/capstone/s01-share; mkdir -p "$SHARE"
cp -f /tmp/capstone/uc/sqlite_silicon.dom  "$SHARE/uc.dom"
cp -f /tmp/capstone/dp0/sqlite_silicon.dom "$SHARE/dp0.dom"
cp -f "$CAPSTONE_TMP_ROOT/sqlite-build/sqlite_host.user" "$SHARE/"
for D in uc dp0; do
  timeout 900 python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
    --share-dir "$SHARE" --log-file /tmp/capstone/s01-qemu-$D.log --timeout-multiplier 6 \
    --guest-command "cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && /tmp/h.user /mnt/host/$D.dom 11" \
    --success-marker 'SQ: obs=' >/dev/null 2>&1
  printf "  QEMU %-4s %s\n" "$D" "$(grep -oE 'SQ: obs=[0-9]+' /tmp/capstone/s01-qemu-$D.log 2>/dev/null | tail -1)"
done

: "${FPGA_URL:?set FPGA_URL from ~/.claude-c/secrets/fpga-console-url}"
: "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
echo "== board: uc FIRST as the live control, then dp0, in ONE boot =="
export SQLITE_STAGE_TIMEOUT=150 SQLITE_IDLE_S=30
export PROBE_SCOPED_OUT=/tmp/capstone/s01-board.txt
export SQLITE_STAGE_DOMS="/test-domains/uc.dom:11,/test-domains/dp0.dom:11"
rm -f "$PROBE_SCOPED_OUT"
python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py >/tmp/capstone/s01-board.log 2>&1
awk '/^===== /{if(l)printf "  %-26s %s\n",l,(g?(o?"RETURN "o:"NO RETURN"):"ENTRY-STALL"); l=$2; g=0; o=""}
     /SQ: G\/enter/{g=1} /SQ: obs=/{o=$2}
     END{if(l)printf "  %-26s %s\n",l,(g?(o?"RETURN "o:"NO RETURN"):"ENTRY-STALL")}' "$PROBE_SCOPED_OUT"
echo "REPRODUCED = uc RETURN and dp0 NO RETURN."
echo "Both returning means the divergence is gone -- check the resident bitstream first."
