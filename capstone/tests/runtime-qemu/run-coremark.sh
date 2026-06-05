#!/usr/bin/env bash
set -euo pipefail

# End-to-end CoreMark correctness run on Capstone PureCap.
# Builds both the domain (.dom) and the Linux host binary (.user), then boots
# QEMU and checks that CoreMark prints "Correct operation validated."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/coremark-build}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-coremark.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/coremark_capstone.dom "$SHARE_DIR"/coremark_host.user

COREMARK_DIR="$SCRIPT_DIR/../../benchmarks/coremark"

bash "$COREMARK_DIR/build-coremark-capstone.sh"
bash "$COREMARK_DIR/build-coremark-host.sh"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    'cp /mnt/host/coremark_host.user /tmp/coremark_host.user && chmod 0755 /tmp/coremark_host.user && /tmp/coremark_host.user /mnt/host/coremark_capstone.dom' \
  --success-marker 'Correct operation validated'

echo 'run-coremark.sh: CoreMark CRC validated'
echo '__COREMARK_PASSED__'
echo "run-coremark.sh completed. Full serial log: $LOG_FILE"
