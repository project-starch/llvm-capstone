#!/usr/bin/env bash
# Stage-0 QEMU proof for ladder rung 1 (matmult-int): build in the silicon config
# and run in a pure-cap domain on QEMU with the gp fabrication OFF. The domain must
# return the same checksum the native oracle computes.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/silicon-ladder}
mkdir -p "$OUT_DIR"
DOM="$OUT_DIR/matmult_int.dom"

# Native oracle (same kernel header) -> expected checksum.
cc -O0 -o "$OUT_DIR/matmult_int_host" "$SCRIPT_DIR/matmult_int_host.c"
EXPECT_DEC=$("$OUT_DIR/matmult_int_host")
echo "oracle: matmult-int checksum = $EXPECT_DEC"

bash "$SCRIPT_DIR/build-ladder-domain.sh" "$SCRIPT_DIR/matmult_int_app.c" "$DOM"

# Run with the gp fabrication disabled; the glue builds gp from cscratch itself.
: "${CAPSTONE_GP_FABRICATE:=0}"; export CAPSTONE_GP_FABRICATE
[[ -n "${CAPSTONE_GP_STANDIN:-}" ]] && export CAPSTONE_GP_STANDIN
python3 "$SCRIPT_DIR/../run-domain-smoke.py" "$DOM"

LOG="$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
EXPECT="retval = $EXPECT_DEC"
if grep -aqF "$EXPECT" "$LOG"; then
  echo "__CAPSTONE_LADDER_MATMULT_INT_PASSED__ ($EXPECT)"
else
  echo "FAIL: expected '$EXPECT' not found" >&2
  grep -aE 'retval|Cap mem|halt|fault' "$LOG" | tail -8 >&2 || true
  exit 1
fi
