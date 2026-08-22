#!/usr/bin/env bash
set -euo pipefail

# Regression guard for file-scope static const capability globals.
# It confirms that all four cases succeed:
# - a direct-use control case,
# - a runtime-side materialization POC,
# - a descriptor-driven materialization POC,
# - the reduced file-scope static const case.
#
# The last one used to be a reproducer: it asserted the cs.cjalr fault that
# static const capability globals produced. The gp cap table resolved that --
# kPair is reached through gp and its fields load with ldc as tagged
# capabilities -- and the assertion has been stale ever since, passing exactly
# when the bug was present. Nobody saw it, because a boot flake in any of the
# four domains aborted the probe first, which it did in every nightly.
# Each domain is retried, so the flake costs a retry rather than the verdict.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
DIRECT_LOG=${DIRECT_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-direct.log}
RUNTIME_LOG=${RUNTIME_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-runtime-materialize.log}
DESCRIPTOR_LOG=${DESCRIPTOR_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-descriptor-materialize.log}
STATIC_LOG=${STATIC_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-static.log}
WRAPPER_LOG=${WRAPPER_LOG:-$TMP_ROOT/capstone-runtime-qemu-static-cap-globals-wrapper.txt}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/static_cap_globals_direct.dom \
      "$SHARE_DIR"/static_cap_globals_runtime_materialize.dom \
      "$SHARE_DIR"/static_cap_globals_descriptor_materialize.dom \
      "$SHARE_DIR"/static_cap_globals_static.dom \
      "$SHARE_DIR"/capstone-test.user

bash "$SCRIPT_DIR/build-capstone-test-user.sh" \
  "$SHARE_DIR/capstone-test.user"

bash "$SCRIPT_DIR/build-static-cap-globals-probe.sh" "$SHARE_DIR"

# 75 is the shared infra-flake code: the guest never reached login, which says
# nothing about the domain. Retry it rather than let it decide the verdict.
run_domain() { # $1=log  $2=dom
  local rc
  for _ in 1 2 3; do
    set +e
    python3 "$SCRIPT_DIR/run-domain-smoke.py" \
      --share-dir "$SHARE_DIR" \
      --log-file "$1" \
      --timeout-multiplier 2 \
      --guest-command "/mnt/host/capstone-test.user /mnt/host/$2" \
      --success-marker 'Created domain ID = 0' \
      --success-marker 'Called dom (1-th time) retval = 305397871'
    rc=$?
    set -e
    [ "$rc" -eq 75 ] || return "$rc"
  done
  return "$rc"
}

run_domain "$DIRECT_LOG" static_cap_globals_direct.dom

run_domain "$RUNTIME_LOG" static_cap_globals_runtime_materialize.dom

run_domain "$DESCRIPTOR_LOG" static_cap_globals_descriptor_materialize.dom

set +e
run_domain "$STATIC_LOG" static_cap_globals_static.dom > "$WRAPPER_LOG" 2>&1
status=$?
set -e

# 75 means the guest never reached login even after the retries, so nothing was
# measured. Propagate it rather than folding it into exit 1: the nightly reports
# 75 as FLAKE, and calling an infra flake a FAIL is exactly the masquerade this
# probe was rewritten to stop.
if [ "$status" -eq 75 ]; then
  echo "static-cap-globals-probe: the static const case never booted; no verdict" >&2
  echo "  static:  $STATIC_LOG" >&2
  exit 75
fi

if [ "$status" -eq 0 ]; then
  echo 'static-cap-globals-probe: all four cases succeeded, including the file-scope static const'
  echo '__STATIC_CAP_GLOBALS_OK__'
  echo "run-static-cap-globals-probe.sh wrapper completed. Logs: direct=$DIRECT_LOG runtime=$RUNTIME_LOG desc=$DESCRIPTOR_LOG static=$STATIC_LOG"
  exit 0
fi

if grep -qF '[CAPSTONE] cs.cjalr requires capability in rs1' "$WRAPPER_LOG" "$STATIC_LOG"; then
  echo 'static-cap-globals-probe: REGRESSION -- the static const case faults on cs.cjalr again,' >&2
  echo '  so a static capability global is reaching the call untagged.' >&2
else
  echo "static-cap-globals-probe: the static const case failed for some other reason:" >&2
fi
echo "  wrapper: $WRAPPER_LOG" >&2
echo "  direct:  $DIRECT_LOG" >&2
echo "  runtime: $RUNTIME_LOG" >&2
echo "  desc:    $DESCRIPTOR_LOG" >&2
echo "  static:  $STATIC_LOG" >&2
exit 1




