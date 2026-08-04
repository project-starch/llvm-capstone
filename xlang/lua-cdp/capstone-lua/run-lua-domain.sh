#!/usr/bin/env bash
# Run the real-Lua Capstone domain under QEMU and show its serial output.
#
#   ./build-lua-domain.sh && ./run-lua-domain.sh
#
# Expects "LUA-OK result=400" in the payload. A capability fault instead means the
# domain halted — the log will carry the "[CAPSTONE] domain halted by capability
# fault: cause=..." line, which is the honest failure to report.
#
# Uses --success-marker '__never__' so run-domain-smoke.py never fails the run on a
# missing marker; we classify from the captured log ourselves. Exit 75 = infra
# flake -> retry (up to ATTEMPTS).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/lua-domain}
SMOKE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
LOG="$SHARE/run.log"
ATTEMPTS=${ATTEMPTS:-3}

[ -f "$SHARE/lua_domain.dom" ] || { echo "no domain — run build-lua-domain.sh first" >&2; exit 1; }
[ -f "$SHARE/lua_host.user" ] || { echo "no host — run build-lua-domain.sh first" >&2; exit 1; }

for attempt in $(seq 1 "$ATTEMPTS"); do
  timeout 520 python3 "$SMOKE" \
    --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier 8 \
    --guest-command "cp /mnt/host/lua_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/lua_domain.dom" \
    --success-marker '__never__' >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq 75 ]; then
    echo "attempt $attempt: infra flake (exit 75), retrying" >&2
    continue
  fi
  break
done

echo "===== domain serial output ====="
grep -aE 'LUA-OK|LUA-FAIL|capability fault|cause ?=|Assertion|lua-host: call retval' "$LOG" || true
echo "================================"
if grep -qa 'LUA-OK result=400' "$LOG"; then
  echo "RESULT: PASS (Lua ran; realloc-moved table returned 400)"
  exit 0
elif grep -qa 'capability fault\|Assertion' "$LOG"; then
  echo "RESULT: CAPABILITY FAULT — see $LOG"
  exit 1
else
  echo "RESULT: no marker (see $LOG)"
  exit 1
fi
