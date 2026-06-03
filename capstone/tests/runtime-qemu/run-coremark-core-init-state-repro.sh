#!/usr/bin/env bash
set -euo pipefail

# Validation probe for the core_init_state() fix.
# Builds with core_state_capstone.c (flat 2D pattern arrays) and expects the
# domain to return successfully rather than hitting the old
# helper_cscincoffset/rs1_v->tag assertion.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-coremark-core-init-state-repro.log}
WRAPPER_LOG=${WRAPPER_LOG:-$TMP_ROOT/capstone-runtime-qemu-coremark-core-init-state-repro-wrapper.txt}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/coremark_core_init_state_repro.dom "$SHARE_DIR"/capstone-test.user

bash "$SCRIPT_DIR/build-capstone-test-user.sh" \
  "$SHARE_DIR/capstone-test.user"

bash "$SCRIPT_DIR/build-coremark-core-init-state-repro.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 2 \
  --guest-command '/mnt/host/capstone-test.user /mnt/host/coremark_core_init_state_repro.dom' \
  --success-marker 'Created domain ID = 0' \
  --success-marker 'Called dom (1-th time) retval =' \
  > "$WRAPPER_LOG" 2>&1

echo 'coremark-core-init-state-repro: core_init_state fix validated'
echo '__COREMARK_CORE_INIT_STATE_REPRO_PASSED__'
echo "run-coremark-core-init-state-repro.sh wrapper completed. Logs: wrapper=$WRAPPER_LOG log=$LOG_FILE"
