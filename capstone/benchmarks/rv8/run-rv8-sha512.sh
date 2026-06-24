#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
SHARE_DIR=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/rv8-build}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-rv8-sha512.log}
mkdir -p "$SHARE_DIR"
rm -f "$SHARE_DIR"/rv8_sha512_capstone.dom "$SHARE_DIR"/rv8_sha512_host.user
bash "$SCRIPT_DIR/build-rv8-sha512-capstone.sh"
bash "$SCRIPT_DIR/build-rv8-sha512-host.sh"
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    "cp /mnt/host/rv8_sha512_host.user /tmp/rv8_sha512_host.user && chmod 0755 /tmp/rv8_sha512_host.user && /tmp/rv8_sha512_host.user sha512 /mnt/host/rv8_sha512_capstone.dom" \
  --success-marker "beebs-sha512-host: correctness marker validated"
echo "run-rv8-sha512.sh: RV8 sha512 correctness marker validated"
echo "__RV8_SHA512_PASSED__"
echo "run-rv8-sha512.sh completed. Full serial log: $LOG_FILE"
