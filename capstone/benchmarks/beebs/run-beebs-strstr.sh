#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/beebs-build}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-beebs-strstr.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/beebs_strstr_capstone.dom \
      "$SHARE_DIR"/beebs_strstr_host.user

bash "$SCRIPT_DIR/build-beebs-strstr-capstone.sh"
bash "$SCRIPT_DIR/build-beebs-strstr-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    'cp /mnt/host/beebs_strstr_host.user /tmp/beebs_strstr_host.user && chmod 0755 /tmp/beebs_strstr_host.user && /tmp/beebs_strstr_host.user strstr /mnt/host/beebs_strstr_capstone.dom' \
  --success-marker 'beebs-strstr-host: correctness marker validated'

echo 'run-beebs-strstr.sh: BEEBS strstr correctness marker validated'
echo '__BEEBS_STRSTR_PASSED__'
echo "run-beebs-strstr.sh completed. Full serial log: $LOG_FILE"
