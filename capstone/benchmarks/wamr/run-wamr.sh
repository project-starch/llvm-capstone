#!/usr/bin/env bash
# Nightly gate for WAMR: build the domain, run it, and require that the value the
# WebAssembly module computes comes back.
#
# The marker is DERIVED from the generated module rather than written down, so
# changing the summands moves the gate with them and cannot leave it asserting a
# number nothing produces any more. WD_STAGE=4 is the full path: init, load,
# instantiate, call.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/wamr-silicon}
SHARE_DIR=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-wamr.log}
DOM=${DOM_NAME:-wamr-nightly}

HDR="$SCRIPT_DIR/port/wamr_test_module.h"
EXPECTED=$(sed -n 's/^#define WAMR_TEST_MODULE_EXPECTED \([0-9]*\).*/\1/p' "$HDR")
[ -n "$EXPECTED" ] || { echo "no WAMR_TEST_MODULE_EXPECTED in $HDR" >&2; exit 1; }
# The domain tags its answer with 0x5741 ("WA") so a bare 42 from anywhere else
# cannot be mistaken for a result. WD_MARKER is what the serial log must show.
WD_MARKER="retval = $(( 0x57410000 + EXPECTED ))"

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/$DOM.dom"

WD_STAGE=4 DOM_NAME="$DOM" OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/build-wamr-silicon.sh"
cp -f "$OUT_DIR/$DOM.dom" "$SHARE_DIR/$DOM.dom"

echo "== expecting $EXPECTED from the module, i.e. '$WD_MARKER'"
CAPSTONE_QUIET_GP=1 python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 6 \
  --guest-command "/mnt/host/capstone-test.user /mnt/host/$DOM.dom 2" \
  --success-marker "$WD_MARKER" \
  "$SHARE_DIR/$DOM.dom"

echo "__CAPSTONE_WAMR_PASSED__"
echo "run-wamr.sh completed. Full serial log: $LOG_FILE"
