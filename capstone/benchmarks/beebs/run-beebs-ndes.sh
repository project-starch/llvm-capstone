#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/beebs-build}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-beebs-ndes.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/beebs_ndes_capstone.dom \
      "$SHARE_DIR"/beebs_ndes_host.user

bash "$SCRIPT_DIR/build-beebs-ndes-capstone.sh"
bash "$SCRIPT_DIR/build-beebs-ndes-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    'cp /mnt/host/beebs_ndes_host.user /tmp/beebs_ndes_host.user && chmod 0755 /tmp/beebs_ndes_host.user && /tmp/beebs_ndes_host.user ndes /mnt/host/beebs_ndes_capstone.dom' \
  --success-marker 'beebs-ndes-host: correctness marker validated'

echo 'run-beebs-ndes.sh: BEEBS ndes correctness marker validated'
echo '__BEEBS_NDES_PASSED__'
echo "run-beebs-ndes.sh completed. Full serial log: $LOG_FILE"
