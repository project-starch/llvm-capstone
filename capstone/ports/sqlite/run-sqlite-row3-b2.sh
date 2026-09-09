#!/usr/bin/env bash
set -euo pipefail

# row3 fork B2 -- the LITERAL matched pair. Real SQLite, one domain, SQLite's
# WHOLE heap is a revoke-on-free linear allocator (revoke_on_free_alloc.h). The
# pointer that faults post-finalize is the exact value sqlite3_column_name
# returned -- no wrapper, no carved copy (that was B1). See
# sqlite_row3_b2_domain.c and history/<date>_row3-b2-literal-revoke-on-free.md.
#
# TWO domains, ONE boot each:
#   - fault variant : post-finalize read of SQLite's own column-name pointer
#     FAULTS. Domain halts, QEMU exits, harness non-zero BY DESIGN; evidence is
#     the monitor fault line. A fault never flushes the host-call payload, so the
#     "row3-b2 live name=..." markers do NOT appear in a fault log -- the
#     no-revoke control carries that evidence.
#   - no-revoke control (-DROW3_B2_NO_REVOKE): identical program and allocator,
#     the free path recycles the slot but does NOT revoke, so the post-finalize
#     read succeeds and the domain RETURNS, reading colname 'c' live and
#     post-finalize. This is BOTH the row3 disambiguation control (cause 24 =
#     "tag gone", which a plain reload also yields) AND the proof that SQLite runs
#     correctly on the allocator.
#
# Cause is opt-level dependent (task-007): -O0 spills `name` so the reload clears
# the tag -> cause 24; -O1/-O2 keep it register-held across finalize -> cause 25,
# self-proving. The domain TU opt level drives this; SQLite (the engine) is always
# built -O0 (its amalgamation is not -O2-selectable on this backend).
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

primary_cause() { [[ "$1" == "-O0" ]] && echo 24 || echo 25; }

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row3-b2-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row3-b2 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row3_b2.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row3_b2_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row3_b2_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row3_b2_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW3_B2_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row3_b2_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 6 \
      --guest-command \
        "cp /mnt/host/sqlite_row3_b2_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row3-b2 no-revoke control at $opt (must RETURN, colname 'c') =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row3_b2_norevoke \
      --success-marker 'row3-b2 live name[0]=c' \
      --success-marker 'row3-b2 post-finalize NOTRAP name[0]=c' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (returned; colname 'c' live and post-finalize)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row3-b2 live name' "$share/sqlite_row3_b2_norevoke.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row3_b2_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  echo "== row3-b2 fault variant at $opt (post-finalize read of SQLite's own ptr must FAULT) =="
  want=$(primary_cause "$opt")
  [[ "$want" == 25 ]] && msg="$REVOKED" || msg="$UNTAGGED"
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row3_b2.log"
    set +e
    smoke sqlite_row3_b2 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if [[ "$cause" == "$want" ]] && grep -q "$msg" "$log" 2>/dev/null; then
        echo "PASS  fault (post-finalize read of revoked SQLite pointer faults: '$msg', cause = $cause)"
        fault_ok=1; break
      fi
      echo "FAIL  fault (cause $cause, expected $want; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-b2-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting -- revoke missed; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW3_B2_MATCHED_PASSED__"
else
  echo "one or more row3-b2 checks FAILED" >&2
fi
exit $fail
