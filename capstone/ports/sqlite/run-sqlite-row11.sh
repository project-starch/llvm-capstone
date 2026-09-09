#!/usr/bin/env bash
set -euo pipefail

# row11 -- the LITERAL matched pair for cve-repros/row11_go_double_finalize
# (LINEAR / double-free). Real SQLite, one domain, SQLite's WHOLE heap is the
# revoke-on-free linear allocator (revoke_on_free_alloc.h), exactly as row3 B2.
# The statement handle sqlite3_prepare_v2 returns is a pointer into an rof
# allocation (the Vdbe block); the FIRST sqlite3_finalize frees -> REVOKEs it, so
# the SECOND sqlite3_finalize dereferences SQLite's own revoked handle and
# FAULTS. See sqlite_row11_domain.c and history/<date>_row11-14-...md.
#
# TWO domains, ONE boot each:
#   - fault variant : second finalize dereferences the revoked stmt and FAULTS.
#     Domain halts, QEMU exits, harness non-zero BY DESIGN; evidence is the
#     monitor fault line. A fault never flushes the host-call payload.
#   - no-revoke control (-DROW11_NO_REVOKE): identical program and allocator, the
#     free path recycles the slot but does NOT revoke, so the second finalize
#     succeeds and the domain RETURNS. This is BOTH the -O0 cause-24
#     disambiguation control AND the proof that SQLite runs on the allocator.
#
# Cause is opt-level dependent (task-007/008): -O0 spills `stmt` across the first
# finalize so the reload clears the tag -> cause 24; -O1/-O2 keep it register-held
# -> cause 25, self-proving. The domain TU opt level drives this; SQLite (the
# engine) is always built -O0.
#
# Reuses the generic B2 host (sqlite_host_row3_b2.c): 3 regions, prints payload.
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
  share="$TMP_ROOT/sqlite-row11-share$opt"
  rm -rf "$share"; mkdir -p "$share"

  echo "== building row11 domains (domain TU $opt, SQLite -O0) + host =="
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row11.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row11_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row11_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row11_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW11_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row11_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3_b2.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 6 \
      --guest-command \
        "cp /mnt/host/sqlite_row11_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row11 no-revoke control at $opt (must RETURN, second finalize rc=0) =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row11_norevoke \
      --success-marker 'row11 first finalize ok' \
      --success-marker 'row11 NOTRAP second finalize rc=0' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (returned; second finalize rc=0)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    if ! grep -q 'row11 prepared stmt ok' "$share/sqlite_row11_norevoke.log" 2>/dev/null; then
      echo "  ...no boot (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row11_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  # The statement handle must cross the first sqlite3_finalize call, so the
  # compiler spills it: the reload before the second finalize comes back
  # untagged -> cause 24 (tag gone) at every opt level observed. If a build ever
  # keeps it register-held across the call the fault is cause 25 (self-proving);
  # accept either -- both are the SAME revoked-handle deref -- and report which.
  # The no-revoke control is the cause-24 disambiguation and runs every opt.
  echo "== row11 fault variant at $opt (second finalize of revoked stmt must FAULT) =="
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row11.log"
    set +e
    smoke sqlite_row11 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if { [[ "$cause" == "24" ]] && grep -q "$UNTAGGED" "$log" 2>/dev/null; } ||
         { [[ "$cause" == "25" ]] && grep -q "$REVOKED" "$log" 2>/dev/null; }; then
        echo "PASS  fault (second finalize of revoked stmt faults: cause = $cause)"
        fault_ok=1; break
      fi
      echo "FAIL  fault (cause $cause not a revoked-handle deref; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-b2-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting -- revoke missed; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW11_MATCHED_PASSED__"
else
  echo "one or more row11 checks FAILED" >&2
fi
exit $fail
