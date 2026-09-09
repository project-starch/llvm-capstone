#!/usr/bin/env bash
set -euo pipefail

# row2 -- the LITERAL matched pair for cve-repros/row2_rusqlite_hook_closure_uaf
# (SEALED-CALLBACK / UAF of the callback context). Real SQLite, one domain.
# SQLite's whole heap is the revoke-on-free allocator (row3 B2). A SQL function
# is registered with a context pointer `app`; the host frees `app` (revoke) while
# the function stays registered; sqlite3_exec drives the callback, which
# dereferences the revoked context via sqlite3_user_data and FAULTS. See
# sqlite_row2_domain.c and the history note.
#
# TWO domains, ONE boot each:
#   - fault variant : SQLite invokes read_context, which reads the revoked pApp
#     context and FAULTS. Domain halts, QEMU exits, harness non-zero BY DESIGN.
#   - no-revoke control (-DROW2_NO_REVOKE): identical program, the free does NOT
#     revoke, so the callback reads 42, exec succeeds and the domain RETURNS.
#     Proves SQLite invokes the real callback on the allocator; disambiguates.
#
# Cause is 24 at ALL opt levels here (unlike the register-held rows): SQLite keeps
# pApp in its function table and reloads it on every invocation, so the revoked
# context is always reached through a MEMORY reload (tag gone -> cause 24). The
# no-revoke control is the disambiguation. The SEAL-proper half (a domain-crossing
# sealed entry) is NOT exercised single-domain -- see the domain comment.
#
# Reuses the generic B2 host (sqlite_host_row3_b2.c): 3 regions, prints payload.
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O2}
RETRIES=${RETRIES:-3}
INFRA_FLAKE_EXIT=75

UNTAGGED="Cap mem access requires capability" # cause 24: tag gone on reload

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row2-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row2 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row2.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row2_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row2_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row2_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW2_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row2_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 8 \
      --guest-command \
        "cp /mnt/host/sqlite_row2_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row2 no-revoke control at $opt (callback reads 42; must RETURN) =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row2_norevoke \
      --success-marker 'row2 registered function, freed context' \
      --success-marker 'row2 NOTRAP exec rc=0' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (callback ran; exec rc=0)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row2 registered function' "$share/sqlite_row2_norevoke.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row2_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  echo "== row2 fault variant at $opt (callback deref of revoked context must FAULT cause 24) =="
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row2.log"
    set +e
    smoke sqlite_row2 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if [[ "$cause" == "24" ]] && grep -q "$UNTAGGED" "$log" 2>/dev/null; then
        echo "PASS  fault (callback deref of revoked context faults: '$UNTAGGED', cause = $cause)"
        fault_ok=1; break
      fi
      echo "FAIL  fault (cause $cause, expected 24; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-b2-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting -- callback did not fault; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW2_MATCHED_PASSED__"
else
  echo "one or more row2 checks FAILED" >&2
fi
exit $fail
