#!/usr/bin/env bash
set -euo pipefail

# Phase-0 feasibility probe for the revoke-on-free allocator (task 008).
#
# One domain, one monitor-granted linear arena. The allocator carves each
# allocation as its own SPLIT sub-capability (own revocation node), MREVs it, and
# hands back the delin'd alias; free revokes it. This probe proves the primitive
# before any SQLite: allocate two buffers, free the first, and show its cached
# alias faults while the second survives and a third allocation works.
#
# Assertion style and the -O-dependent cause are exactly as in
# run-intra-domain-mrev-revoke-probe.sh (which this is modelled on): OK probes
# return a value; FAULT probes halt the domain and QEMU exits, evidence in the
# serial log; each probe gets its own boot. -O0 spills the alias (cause 24, with
# alloc_no_free_ok as control); -O1/-O2 keep it in a register (cause 25).
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

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
    --guest-command "/mnt/host/revoke_on_free_probe.user /mnt/host/$name.dom" \
    "$@"
}

run_ok() { # $1=probe  $2=expected retval
  local name="$1" retval="$2"
  local marker="revoke-on-free-probe: call retval = $retval"
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
       ! grep -q "revoke-on-free-probe: call retval" "$log" 2>/dev/null; then
      echo "  ...no boot/retval for $name (attempt $attempt), retrying" >&2; continue
    fi
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
    if grep -q "revoke-on-free-probe: call retval" "$log" 2>/dev/null; then
      echo "FAIL  $name  (returned instead of faulting; see $log)" >&2; return 1
    fi
    [[ $attempt -le $RETRIES ]] && { echo "  ...no boot/fault for $name (attempt $attempt), retrying" >&2; continue; }
    echo "FAIL  $name  (no fault after $attempt attempts; see $log)" >&2; return 1
  done
}

fail=0
for opt in $OPT_LEVELS; do
  SHARE="$TMP_ROOT/revoke-on-free-share$opt"
  rm -rf "$SHARE"; mkdir -p "$SHARE"

  echo "== building controller + domains at $opt =="
  DOMAIN_OPT_LEVEL="$opt" bash "$SCRIPT_DIR/build-revoke-on-free-probe.sh" "$SHARE" >/dev/null

  echo "== running at $opt (one boot each) =="
  want=$(primary_cause "$opt")
  [[ "$want" == 25 ]] && msg="$REVOKED" || msg="$UNTAGGED"
  run_fault alloc_use_after_free_fault "$msg" "$want" || fail=1
  run_ok   alloc_no_free_ok 0x0812005e || fail=1
  run_ok   alloc_sibling_survives_ok 0x0813003c || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_REVOKE_ON_FREE_PASSED__"
else
  echo "one or more probes FAILED" >&2
fi
exit $fail
