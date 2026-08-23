#!/usr/bin/env bash
set -euo pipefail

# Phase-0 feasibility probe for the HIERARCHICAL revoke-on-free allocator
# (task 010, checkpoint H).
#
# One domain, one monitor-granted linear arena. The allocator carves a
# per-connection SUB-ARENA (a fresh SPLIT node) and MREVs it; child allocations
# are SPLIT descendants of that senior node. Closing the connection REVOKEs the
# senior node and must sweep the whole subtree (the child), while a sibling
# connection's sub-arena survives. This proves the tree primitive before any
# SQLite (checkpoint H, row7).
#
# Assertion style and the -O-dependent cause are exactly as in
# run-revoke-on-free-probe.sh, which this is modelled on: OK probes return a
# value; FAULT probes halt the domain and QEMU exits, evidence in the serial log;
# each probe gets its own boot. -O0 spills the child alias (cause 24, with
# hier_no_close_ok as control); -O1/-O2 keep it in a register (cause 25).
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/infra-retry.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

primary_cause() { [[ "$1" == "-O0" ]] && echo 24 || echo 25; }

smoke() { # $1=probe  rest: extra harness args
  local name="$1"; shift
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
    --share-dir "$SHARE" \
    --log-file "$SHARE/$name.log" \
    --guest-command "/mnt/host/hier_revoke_probe.user /mnt/host/$name.dom" \
    "$@"
}

run_ok() { # $1=probe  $2=expected retval
  local name="$1" retval="$2"
  local marker="hier-revoke-probe: call retval = $retval"
  local log="$SHARE/$name.log"
  local attempt=0 rc
  while :; do
    attempt=$((attempt + 1))
    set +e
    smoke "$name" --success-marker "$marker" >/dev/null 2>&1
    rc=$?
    set -e
    [[ $rc -eq 0 ]] && { echo "PASS  $name  (retval $retval)"; return 0; }
    if [[ $rc -eq $INFRA_FLAKE_EXIT && $attempt -le $RETRIES ]]; then
      echo "  ...infra flake on $name (attempt $attempt), retrying" >&2; continue
    fi
    if [[ $attempt -le $RETRIES ]] &&
       ! grep -q "hier-revoke-probe: call retval" "$log" 2>/dev/null; then
      echo "  ...no boot/retval for $name (attempt $attempt), retrying" >&2; continue
    fi
    capstone_domain_ran "$log" "hier-revoke-probe:" || {
      echo "FLAKE $name  (guest never ran; no verdict; see $log)" >&2; return 75; }
    echo "FAIL  $name  (rc=$rc; see $log)" >&2; return 1
  done
}

run_fault() { # $1=probe  $2=expected diagnostic  $3=expected cause
  local name="$1" msg="$2" want="$3"
  local log="$SHARE/$name.log"
  local attempt=0 cause
  while :; do
    attempt=$((attempt + 1))
    set +e
    smoke "$name" >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if [[ "$cause" == "$want" ]] && grep -q "$msg" "$log" 2>/dev/null; then
        echo "PASS  $name  (fault: '$msg', cause = $cause)"; return 0
      fi
      echo "FAIL  $name  (cause $cause, expected $want; see $log)" >&2; return 1
    fi
    if grep -q "hier-revoke-probe: call retval" "$log" 2>/dev/null; then
      echo "FAIL  $name  (returned instead of faulting -- parent revoke missed the child; see $log)" >&2; return 1
    fi
    [[ $attempt -le $RETRIES ]] && { echo "  ...no boot/fault for $name (attempt $attempt), retrying" >&2; continue; }
    capstone_domain_ran "$log" "hier-revoke-probe:" || {
      echo "FLAKE $name  (guest never ran; no verdict; see $log)" >&2; return 75; }
    echo "FAIL  $name  (no fault after $attempt attempts; see $log)" >&2; return 1
  done
}

fail=0 flaked=0

# A probe that never ran is not a probe that failed. Counting both as `fail=1`
# is how this suite reported FAIL(1) on a night when the guest simply never
# reached login. A real failure still outranks a flake.
record() { # $1 = a probe's exit code
  case "$1" in
    0)  ;;
    75) flaked=$((flaked + 1)) ;;
    *)  fail=1 ;;
  esac
}
for opt in $OPT_LEVELS; do
  SHARE="$TMP_ROOT/hier-revoke-share$opt"
  rm -rf "$SHARE"; mkdir -p "$SHARE"

  echo "== building controller + domains at $opt =="
  DOMAIN_OPT_LEVEL="$opt" bash "$SCRIPT_DIR/build-hier-revoke-probe.sh" "$SHARE" >/dev/null

  echo "== running at $opt (one boot each) =="
  want=$(primary_cause "$opt")
  [[ "$want" == 25 ]] && msg="$REVOKED" || msg="$UNTAGGED"
  run_fault hier_child_revoked_fault "$msg" "$want" || record $?
  run_ok   hier_no_close_ok 0x0872005e || record $?
  run_ok   hier_sibling_conn_survives_ok 0x0873003c || record $?
done

if [[ $fail -ne 0 ]]; then
  echo "one or more probes FAILED" >&2
  exit 1
elif [[ $flaked -ne 0 ]]; then
  echo "$flaked probe(s) never ran -- infra flake, no verdict" >&2
  exit 75
fi
echo "__CAPSTONE_HIER_REVOKE_PASSED__"
