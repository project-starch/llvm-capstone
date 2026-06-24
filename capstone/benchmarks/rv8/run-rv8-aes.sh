#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
SHARE_DIR=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/rv8-build}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-rv8-aes.log}
mkdir -p "$SHARE_DIR"
rm -f "$SHARE_DIR"/rv8_aes_capstone.dom "$SHARE_DIR"/rv8_aes_host.user
bash "$SCRIPT_DIR/build-rv8-aes-capstone.sh"
bash "$SCRIPT_DIR/build-rv8-aes-host.sh"
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    "cp /mnt/host/rv8_aes_host.user /tmp/rv8_aes_host.user && chmod 0755 /tmp/rv8_aes_host.user && /tmp/rv8_aes_host.user aes /mnt/host/rv8_aes_capstone.dom" \
  --success-marker "beebs-aes-host: correctness marker validated"
echo "run-rv8-aes.sh: RV8 aes correctness marker validated"
echo "__RV8_AES_PASSED__"
echo "run-rv8-aes.sh completed. Full serial log: $LOG_FILE"
