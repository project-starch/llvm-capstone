#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

if [[ -z "${BEEBS_BENCHMARK:-}" ]]; then
  echo "BEEBS_BENCHMARK must be set before sourcing this file" >&2
  exit 1
fi

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/beebs-build}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-beebs-${BEEBS_BENCHMARK}.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/beebs_${BEEBS_BENCHMARK}_capstone.dom \
      "$SHARE_DIR"/beebs_${BEEBS_BENCHMARK}_host.user

bash "$SCRIPT_DIR/build-beebs-${BEEBS_BENCHMARK}-capstone.sh"
bash "$SCRIPT_DIR/build-beebs-${BEEBS_BENCHMARK}-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 4 \
  --guest-command \
    "cp /mnt/host/beebs_${BEEBS_BENCHMARK}_host.user /tmp/beebs_${BEEBS_BENCHMARK}_host.user && chmod 0755 /tmp/beebs_${BEEBS_BENCHMARK}_host.user && /tmp/beebs_${BEEBS_BENCHMARK}_host.user ${BEEBS_BENCHMARK} /mnt/host/beebs_${BEEBS_BENCHMARK}_capstone.dom" \
  --success-marker "beebs-${BEEBS_BENCHMARK}-host: correctness marker validated"

upper_name=$(printf "%s" "$BEEBS_BENCHMARK" | tr '[:lower:]-' '[:upper:]_')
echo "run-beebs-${BEEBS_BENCHMARK}.sh: BEEBS ${BEEBS_BENCHMARK} correctness marker validated"
echo "__BEEBS_${upper_name}_PASSED__"
echo "run-beebs-${BEEBS_BENCHMARK}.sh completed. Full serial log: $LOG_FILE"
