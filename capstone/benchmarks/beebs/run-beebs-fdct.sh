#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/beebs-build}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-beebs-fdct.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/beebs_fdct_capstone.dom \
      "$SHARE_DIR"/beebs_fdct_host.user

bash "$SCRIPT_DIR/build-beebs-fdct-capstone.sh"
bash "$SCRIPT_DIR/build-beebs-fdct-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    'cp /mnt/host/beebs_fdct_host.user /tmp/beebs_fdct_host.user && chmod 0755 /tmp/beebs_fdct_host.user && /tmp/beebs_fdct_host.user fdct /mnt/host/beebs_fdct_capstone.dom' \
  --success-marker 'beebs-fdct-host: correctness marker validated'

echo 'run-beebs-fdct.sh: BEEBS fdct correctness marker validated'
echo '__BEEBS_FDCT_PASSED__'
echo "run-beebs-fdct.sh completed. Full serial log: $LOG_FILE"
