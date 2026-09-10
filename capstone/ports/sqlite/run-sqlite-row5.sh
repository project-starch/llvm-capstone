#!/usr/bin/env bash
set -euo pipefail

# row5 -- the LITERAL matched pair for cve-repros/row5_php_destruction_order
# (HIERARCHICAL-REVOKE, PHP binding). Real SQLite, one domain. The connection gets
# its own MREV'd sub-arena; its db_object wrapper AND statement are SPLIT
# descendants. The DB object's free handler (which runs FIRST, the #69971 wrong
# order) REVOKEs the sub-arena's senior node, sweeping wrapper + connection +
# child statement; the statement free handler then faults on the revoked child.
# See sqlite_row5_domain.c and revoke_on_free_hier_alloc.h.
#
# THREE domains, ONE boot each per opt level:
#   - fault variant : the statement free handler's sqlite3_finalize of the revoked
#     child statement FAULTS. Domain halts, QEMU exits, harness non-zero BY DESIGN.
#   - no-revoke control (-DROW5_NO_REVOKE): identical program and allocator, the
#     db-object free handler does NOT fire the subtree revoke. The wrapper stays
#     intact, the statement free handler (finalize + free_list walk) returns and
#     the domain RETURNS. BOTH the -O0 cause-24 disambiguation control AND proof
#     SQLite runs on the hierarchical allocator.
#   - sibling scoping (-DROW5_SIBLING): opens two connections, frees A's db object
#     (revoke), then steps B's statement -- B SURVIVES and the domain RETURNS.
#     Proves the revoke is scoped to A's subtree (hierarchical, not a global heap
#     wipe), the same property tests/runtime-qemu/hier-revoke-probe proves at the
#     primitive level -- here on real SQLite.
#
# Cause is opt-level dependent: -O0 spills the statement handle so the reload
# clears the tag -> cause 24; -O1/-O2 keep it register-held -> cause 25,
# self-proving. SQLite (the engine) is always built -O0.
#
# Reuses the generic B2 host (sqlite_host_row3_b2.c): 3 regions, prints payload.
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
SIBLING_OPT_LEVELS=${SIBLING_OPT_LEVELS:--O2}
RETRIES=${RETRIES:-3}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

want_sibling() { # $1=opt -> 0 if this opt is in SIBLING_OPT_LEVELS
  local o
  for o in $SIBLING_OPT_LEVELS; do [[ "$o" == "$1" ]] && return 0; done
  return 1
}

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row5-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row5 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row5.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row5_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row5_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row5_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW5_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  if want_sibling "$opt"; then
    OUT_DIR="$share" OUT_DOM="$share/sqlite_row5_sibling.dom" \
      DOMAIN_SRC="$SCRIPT_DIR/sqlite_row5_domain.c" \
      DOMAIN_EXTRA_FLAGS="-DROW5_SIBLING" \
      DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
      bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  fi
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row5_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 8 \
      --guest-command \
        "cp /mnt/host/sqlite_row5_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row5 no-revoke control at $opt (wrapper intact; stmt free handler must RETURN) =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row5_norevoke \
      --success-marker 'row5 freed db object' \
      --success-marker 'row5 NOTRAP stmt free handler rc=' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (wrapper intact; stmt free handler returned)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row5 prepared child statement ok' "$share/sqlite_row5_norevoke.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row5_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  # The statement handle is spilled across sqlite3_close_v2/finalize, so the
  # reload after the revoke comes back untagged -> cause 24; a register-held build
  # would be cause 25 (self-proving). Accept either -- both are the SAME
  # revoked-child deref -- and report which. The no-revoke control disambiguates.
  echo "== row5 fault variant at $opt (post-teardown stmt free handler must FAULT) =="
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row5.log"
    set +e
    smoke sqlite_row5 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if { [[ "$cause" == "24" ]] && grep -q "$UNTAGGED" "$log" 2>/dev/null; } ||
         { [[ "$cause" == "25" ]] && grep -q "$REVOKED" "$log" 2>/dev/null; }; then
        echo "PASS  fault (post-teardown stmt free handler faults on revoked child: cause = $cause)"
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

  if want_sibling "$opt"; then
    echo "== row5 sibling scoping at $opt (free A; B's statement must SURVIVE) =="
    sib_ok=0
    for attempt in $(seq 1 $((RETRIES + 1))); do
      set +e
      smoke sqlite_row5_sibling \
        --success-marker 'row5 freed db object A' \
        --success-marker 'row5 SIBLING survived free rc=' >/dev/null 2>&1
      rc=$?
      set -e
      if [[ $rc -eq 0 ]]; then echo "PASS  sibling (connection B survived connection A's revoke)"; sib_ok=1; break; fi
      if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
      if ! grep -q 'row5 prepared two sibling connections ok' "$share/sqlite_row5_sibling.log" 2>/dev/null; then
        echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
      echo "FAIL  sibling (rc=$rc; B did not survive -- see $share/sqlite_row5_sibling.log)" >&2; break
    done
    [[ $sib_ok -eq 1 ]] || fail=1
  fi
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW5_MATCHED_PASSED__"
else
  echo "one or more row5 checks FAILED" >&2
fi
exit $fail
