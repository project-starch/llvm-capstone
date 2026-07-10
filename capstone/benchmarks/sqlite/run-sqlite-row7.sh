#!/usr/bin/env bash
set -euo pipefail

# row7 -- the LITERAL matched pair for cve-repros/row7_cpython_cursor_dealloc
# (HIERARCHICAL-REVOKE). Real SQLite, one domain. The connection gets its own
# MREV'd sub-arena; the statement is a SPLIT descendant of it. Closing the
# connection REVOKEs the sub-arena's senior node, sweeping the connection AND its
# live statement; a later sqlite3_step on the child statement FAULTS. See
# sqlite_row7_domain.c, revoke_on_free_hier_alloc.h and the history note.
#
# TWO domains, ONE boot each:
#   - fault variant : post-close sqlite3_step of the revoked child statement
#     FAULTS. Domain halts, QEMU exits, harness non-zero BY DESIGN.
#   - no-revoke control (-DROW7_NO_REVOKE): identical program and allocator, the
#     close does NOT fire the subtree revoke. SQLite's zombie close leaves the
#     statement usable, so the post-close step succeeds and the domain RETURNS.
#     This is BOTH the -O0 cause-24 disambiguation control AND the proof that
#     SQLite runs on the hierarchical allocator.
#
# The SCOPING property (a sibling connection survives one connection's close) is
# proven at the primitive level by tests/runtime-qemu/hier-revoke-probe; this
# script demonstrates the literal cascade on real SQLite.
#
# Cause is opt-level dependent: -O0 spills the statement handle across the close
# so the reload clears the tag -> cause 24; -O1/-O2 keep it register-held ->
# cause 25, self-proving. SQLite (the engine) is always built -O0.
#
# Reuses the generic B2 host (sqlite_host_row3_b2.c): 3 regions (metadata,
# payload, 4 MiB arena), prints payload.
# Requires the rootfs.ext2 write lock: announce in agent-handoff/COORDINATION.md.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-3}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

primary_cause() { [[ "$1" == "-O0" ]] && echo 24 || echo 25; }

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row7-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row7 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row7.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row7_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row7_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row7_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW7_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row7_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 8 \
      --guest-command \
        "cp /mnt/host/sqlite_row7_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row7 no-revoke control at $opt (zombie close; step must RETURN) =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row7_norevoke \
      --success-marker 'row7 closed connection' \
      --success-marker 'row7 NOTRAP post-close step rc=' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (zombie close; post-close step returned)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row7 prepared child statement ok' "$share/sqlite_row7_norevoke.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row7_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  # The statement handle crosses the sqlite3_close_v2 call, so it is spilled and
  # the reload after the revoke comes back untagged -> cause 24; a register-held
  # build would be cause 25 (self-proving). Accept either -- both are the SAME
  # revoked-child deref -- and report which. The no-revoke control disambiguates.
  echo "== row7 fault variant at $opt (post-close step of revoked child must FAULT) =="
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row7.log"
    set +e
    smoke sqlite_row7 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if { [[ "$cause" == "24" ]] && grep -q "$UNTAGGED" "$log" 2>/dev/null; } ||
         { [[ "$cause" == "25" ]] && grep -q "$REVOKED" "$log" 2>/dev/null; }; then
        echo "PASS  fault (post-close step of revoked child statement faults: cause = $cause)"
        fault_ok=1; break
      fi
      echo "FAIL  fault (cause $cause not a revoked-handle deref; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-b2-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting -- cascade missed; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW7_MATCHED_PASSED__"
else
  echo "one or more row7 checks FAILED" >&2
fi
exit $fail
