#!/usr/bin/env bash
set -euo pipefail

# row14 -- the LITERAL matched pair for cve-repros/row14_cpython_uninit_connection
# (UNINIT / use-before-init). Real SQLite, one domain. The connection handle `db`
# is modelled as a genuine UNINIT capability (task-009: revoke a still-linear
# lineage -> UNINIT). before.c reads `*(unsigned char *)c->db` before open; here
# a pre-open read through the UNINIT db FAULTS with cause 26. See
# sqlite_row14_domain.c and history/<date>_row11-14-...md.
#
# The matched pair is the ORDER of the two operations (exactly the before.c
# defect: db used before open assigns it):
#   - fault variant : read db, THEN sqlite3_open -> the pre-open read FAULTS
#     (cause 26). Domain halts, QEMU exits, harness non-zero BY DESIGN.
#   - correct control (-DROW14_OPEN_FIRST): sqlite3_open FIRST (real SQLite
#     overwrites db with a valid handle), THEN read db -> succeeds, RETURNS.
#     Carries the "real SQLite opened the connection" evidence.
#
# Cause 26 does NOT move with -O (task-009): the UNINIT handle keeps its tag and
# rev node, so it faults on TYPE, not tag-gone, at every opt level. Self-proving.
#
# Reuses the generic B2 host (sqlite_host_row3_b2.c): 3 regions, prints payload.
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

UNINIT="Cap mem load through uninitialised capability" # cause 26: no read authority

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row14-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row14 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row14.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row14_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row14_openfirst.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row14_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW14_OPEN_FIRST" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row14_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 6 \
      --guest-command \
        "cp /mnt/host/sqlite_row14_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row14 correct control at $opt (open FIRST, must RETURN) =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row14_openfirst \
      --success-marker 'row14 opened connection ok' \
      --success-marker 'row14 NOTRAP post-open read byte=' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (opened real SQLite connection; post-open read ok)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row14 opened connection ok' "$share/sqlite_row14_openfirst.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row14_openfirst.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  echo "== row14 fault variant at $opt (pre-open read of UNINIT db must FAULT cause 26) =="
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row14.log"
    set +e
    smoke sqlite_row14 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if [[ "$cause" == "26" ]] && grep -q "$UNINIT" "$log" 2>/dev/null; then
        echo "PASS  fault (pre-open read of UNINIT db faults: '$UNINIT', cause = $cause)"
        fault_ok=1; break
      fi
      echo "FAIL  fault (cause $cause, expected 26; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-b2-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW14_MATCHED_PASSED__"
else
  echo "one or more row14 checks FAILED" >&2
fi
exit $fail
