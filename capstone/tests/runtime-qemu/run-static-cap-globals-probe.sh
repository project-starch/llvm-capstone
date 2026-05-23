#!/usr/bin/env bash
set -euo pipefail

# Diagnostic wrapper for the current static/global capability blocker.
# It confirms that:
# - a direct-use control case succeeds, and
# - the reduced file-scope static const case reproduces the current failure.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
DIRECT_LOG=${DIRECT_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-direct.log}
STATIC_LOG=${STATIC_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-static.log}
WRAPPER_LOG=${WRAPPER_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-wrapper.txt}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/static_cap_globals_direct.dom \
      "$SHARE_DIR"/static_cap_globals_static.dom \
      "$SHARE_DIR"/capstone-test.user

bash "$SCRIPT_DIR/build-capstone-test-user.sh" \
  "$SHARE_DIR/capstone-test.user"

bash "$SCRIPT_DIR/build-static-cap-globals-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$DIRECT_LOG" \
  --guest-command '/mnt/host/capstone-test.user /mnt/host/static_cap_globals_direct.dom' \
  --success-marker 'Created domain ID = 0' \
  --success-marker 'Called dom (1-th time) retval = 305397871'

set +e
python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$STATIC_LOG" \
  --guest-command '/mnt/host/capstone-test.user /mnt/host/static_cap_globals_static.dom' \
  --success-marker 'Created domain ID = 0' \
  --success-marker 'Called dom (1-th time) retval = 305397871' \
  > "$WRAPPER_LOG" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
  echo "static-cap-globals-probe: the static const reproducer unexpectedly succeeded" >&2
  echo "  wrapper: $WRAPPER_LOG" >&2
  echo "  direct:  $DIRECT_LOG" >&2
  echo "  static:  $STATIC_LOG" >&2
  exit 1
fi

if grep -qF '[CAPSTONE] cs.cjalr requires capability in rs1' "$WRAPPER_LOG" "$STATIC_LOG"; then
  echo 'static-cap-globals-probe: reproduced current static/global capability failure after control-case success'
  echo '__STATIC_CAP_GLOBALS_REPRODUCED__'
  echo "run-static-cap-globals-probe.sh wrapper completed. Logs: direct=$DIRECT_LOG static=$STATIC_LOG"
  exit 0
fi

echo "static-cap-globals-probe: unexpected failure; inspect logs:" >&2
echo "  wrapper: $WRAPPER_LOG" >&2
echo "  direct:  $DIRECT_LOG" >&2
echo "  static:  $STATIC_LOG" >&2
exit 1

