#!/usr/bin/env bash
set -euo pipefail

# row3 matched pair -- real SQLite, single domain, revoke at sqlite3_finalize.
#
# The SAME program as cve-repros/row3_diesel_colname_cached/before.c (open ->
# prepare "SELECT a AS colname" -> step -> column_name -> finalize -> read
# name[0]), compiled into a real-SQLite Capstone domain. A thin wrapper carves an
# independently revocable copy of the real column name out of a monitor-granted
# linear arena and REVOKEs it at finalize; the post-finalize read then faults.
# See sqlite_row3_domain.c for what is and isn't literal (fork B1 vs B2).
#
# TWO domains, ONE boot each (a faulted domain poisons later create_dom in the
# same guest session):
#   - fault variant : post-finalize read FAULTS. Domain halts, QEMU exits, the
#     harness returns non-zero BY DESIGN; the evidence is the monitor fault line.
#   - no-revoke control (-DROW3_NO_REVOKE): identical program, no revoke, so the
#     read succeeds and the domain RETURNS. Disambiguates the -O0 cause-24 fault
#     ("tag gone", which a plain spill-reload also yields) from a real revoke.
#
# Cause is opt-level dependent (task-007): -O0 spills the alias so the reload
# clears the tag -> cause 24; -O1/-O2 keep it in a register -> cause 25, which is
# self-proving (tag intact, node revoked). -O1/-O2 need the C1 fastcc+cap-arg fix.
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)
# and confirm the other agent is not mid-run before invoking this.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
# Default -O0 only: the matched pair is validated GREEN at -O0 (fault cause 24 +
# no-revoke control). At -O1/-O2 the domain TU currently aborts QEMU with
# `helper_csdelin: Assertion rd_v->tag failed` BEFORE the revoke (it aborts the
# no-revoke control too, so it is NOT the mechanism): a `.insn`-derived capability
# loses its tag across a call spill once the optimiser keeps it out of -O0's
# tag-safe stack slots. That is a domain-TU codegen robustness bug (B's lane),
# tracked separately. The -O2 ASM still proves the alias is register-held (the
# self-proving path); only the -O1+ QEMU boot is blocked. Override to reproduce:
#   OPT_LEVELS="-O0 -O1 -O2" bash run-sqlite-row3.sh
OPT_LEVELS=${OPT_LEVELS:--O0}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

primary_cause() { [[ "$1" == "-O0" ]] && echo 24 || echo 25; }

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/sqlite-row3-share$opt"
  rm -rf "$share"
  mkdir -p "$share"

  # SQLite (the engine) is always built at -O0: the amalgamation does not compile
  # at -O2 on this backend (an i128 CapstoneISD::SELECT_CC in sqlite3_str_vappendf
  # is not selectable -- a pre-existing codegen limitation, unrelated to row3).
  # The fault-cause evidence depends only on the DOMAIN TU's opt level, since the
  # held alias lives in run_row3(): -O0 spills it (cause 24), -O1+ keeps it in a
  # register across the revoke (cause 25, self-proving). So `opt` drives only the
  # domain TU.
  echo "== building row3 domains (domain TU $opt, SQLite -O0) + host =="
  # Fault variant.
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row3.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row3_domain.c" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  # No-revoke control (reuses the cached sqlite3.o in the same OUT_DIR).
  OUT_DIR="$share" OUT_DOM="$share/sqlite_row3_norevoke.dom" \
    DOMAIN_SRC="$SCRIPT_DIR/sqlite_row3_domain.c" \
    DOMAIN_EXTRA_FLAGS="-DROW3_NO_REVOKE" \
    DOMAIN_OPT_LEVEL="$opt" SQLITE_OPT_LEVEL="-O0" \
    bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
  OUT_DIR="$share" OUT_HOST="$share/sqlite_row3_host.user" \
    HOST_SRC="$SCRIPT_DIR/sqlite_host_row3.c" \
    bash "$SCRIPT_DIR/build-sqlite-host.sh" >/dev/null

  smoke() { # $1=dom basename  rest: extra harness args
    local dom="$1"; shift
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$share" \
      --log-file "$share/$dom.log" \
      --timeout-multiplier 6 \
      --guest-command \
        "cp /mnt/host/sqlite_row3_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      "$@"
  }

  echo "== row3 no-revoke control at $opt (must RETURN, read colname 'c') =="
  ctrl_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    set +e
    smoke sqlite_row3_norevoke \
      --success-marker 'row3 live name[0]=c' \
      --success-marker 'row3 post-finalize NOTRAP name[0]=c' >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then echo "PASS  control (returned; colname 'c' live and post-finalize)"; ctrl_ok=1; break; fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT ]]; then echo "  ...infra flake (attempt $attempt), retrying" >&2; continue; fi
    echo "FAIL  control (rc=$rc; see $share/sqlite_row3_norevoke.log)" >&2; break
  done
  [[ $ctrl_ok -eq 1 ]] || fail=1

  echo "== row3 fault variant at $opt (post-finalize read must FAULT) =="
  want=$(primary_cause "$opt")
  [[ "$want" == 25 ]] && msg="$REVOKED" || msg="$UNTAGGED"
  fault_ok=0
  for attempt in $(seq 1 $((RETRIES + 1))); do
    log="$share/sqlite_row3.log"
    set +e
    smoke sqlite_row3 --success-marker '__never__' >/dev/null 2>&1
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      # NOTE: the fault halts the domain, so call_dom never returns and the host
      # never flushes the payload region -- the "row3 live name=..." markers
      # CANNOT appear in a fault log. The no-revoke CONTROL (identical program,
      # asserted just above) is what proves the same code reads colname 'c' live
      # and post-finalize; here we assert only that flipping on the revoke turns
      # that same read into a fault of the expected cause. Together = the matched
      # pair on one program.
      if [[ "$cause" == "$want" ]] && grep -q "$msg" "$log" 2>/dev/null; then
        echo "PASS  fault (post-finalize read of revoked alias faults: '$msg', cause = $cause)"
        fault_ok=1
        break
      fi
      echo "FAIL  fault (cause $cause, expected $want; see $log)" >&2; break
    fi
    if grep -q 'sqlite-row3-host: call retval' "$log" 2>/dev/null; then
      echo "FAIL  fault (domain returned instead of faulting -- revoke missed; see $log)" >&2; break
    fi
    echo "  ...no boot/fault (attempt $attempt), retrying" >&2
  done
  [[ $fault_ok -eq 1 ]] || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_SQLITE_ROW3_MATCHED_PASSED__"
else
  echo "row3 matched pair FAILED" >&2
fi
exit $fail
