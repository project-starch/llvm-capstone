#!/usr/bin/env bash
# Build both arms, take the QEMU reference, then run them on the board in the required order.
# tagr MUST run first: it is what makes tagf's no-result readable as "the lcc trapped".
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1
unset LINKER_SCRIPT
for g in tagr tagf; do
  DOMAIN_GLUE=interp DOMAIN_WINDOW=0x150000 OUT_DIR=/tmp/capstone/$g \
    DOMAIN_EXTRA_CFLAGS="-DLADDER_NO_MINSTRET=1" \
    bash capstone/tests/rtl-smoke/build-ladder-fpga.sh $g || { echo "build failed: $g"; exit 1; }
done
echo "== QEMU reference: tagr returns 1017, tagf ABORTS in helper_cslcc =="
( cd capstone/tests/runtime-qemu/silicon-ladder
  for g in tagr tagf; do
    printf "  %-5s " "$g"
    DOMAIN_GLUE=interp DOMAIN_OPT_LEVEL=-O1 bash run-ladder-qemu.sh $g 2>&1 \
      | grep -oE "retval = [0-9]+|Assertion .* failed" | tail -1
  done )
: "${FPGA_URL:?set FPGA_URL from ~/.claude-c/secrets/fpga-console-url}"
: "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
export LADDER_FPGA_DIR=/tmp/capstone/ladder-fpga BAKED_TIMEOUT=150
echo 1017 > "$LADDER_FPGA_DIR/tagr.oracle"; echo 1011 > "$LADDER_FPGA_DIR/tagf.oracle"
( cd capstone/tests/rtl-smoke && BAKED_RUNGS="tagr tagf" python3 -m fpga_driver.run_baked_rungs_fpga )
echo "PASS = tagr 1017 and tagf NO RESULT (tag cleared). tagf RETURNING a value is the defect."
