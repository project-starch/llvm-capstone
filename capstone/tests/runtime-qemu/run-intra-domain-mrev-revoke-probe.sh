#!/usr/bin/env bash
set -euo pipefail

# Single-domain held-cap BORROW-REVOKE probe (row3 Option B, the faithful shape).
#
# ONE domain receives a real monitor-granted LINEAR capability over a region,
# mints its own MREV handle, uses a delinearised alias, REVOKEs at a lifecycle
# point, and the cached alias faults. No second entity, no monitor-mediated
# revoke, no csdebuggencap hand-mint -- the arena comes down the real
# REGION_SHARE delivery path and the revoke is intra-domain.
#
# ASSERTION STYLE (inherited from tests/capstone-mrev-codegen):
#   - OK probes RETURN a value; the controller prints it and we match the marker.
#   - FAULT probes halt the domain. A domain_main .dom runs in PRV_C, whose
#     capability faults have no delivery path, so QEMU prints the fault line and
#     exits; the guest never returns to the shell and the harness exits non-zero
#     BY DESIGN. The evidence is the fault line in the serial log.
#   Each probe gets its OWN boot: a faulted domain poisons later domain creation
#   in the same guest session, and --success-marker is per-invocation.
#
# THE EXPECTED CAUSE IS ASSERTED, NOT JUST REPORTED, and it is opt-level
# dependent for the primary probe. Cause 25 ("on revoked capability") means the
# tag was INTACT and the revoked rev-tree node stopped the deref -- self-proving.
# Cause 24 ("requires capability") only means the tag was gone, which a reload of
# a revoked capability AND a consumed linear capability both produce; every
# cause-24 expectation therefore carries a no-revoke control (held_no_revoke_ok).
#
# At -O0 clang spills the alias, so the post-revoke deref reloads it (ldc) and
# the reload clears the tag -> cause 24. At -O1/-O2 the alias stays in a register
# across the revoke -> cause 25. Both are real faults; only -O1+ is self-proving.
# (-O1/-O2 domain payloads need the CC_Capstone_FastCC capability-argument fix,
# task-006 C1.)
#
# Requires the rootfs.ext2 write lock: announce in agent-handoff/COORDINATION.md
# and confirm the other agent is not mid-run before invoking this.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

REVOKED="Cap mem access on revoked capability" # cause 25: tag intact, node revoked
UNTAGGED="Cap mem access requires capability"  # cause 24: tag gone

# Expected cause for the primary probe: it derefs a register-held alias only when
# the optimiser keeps it in a register.
primary_cause() { [[ "$1" == "-O0" ]] && echo 24 || echo 25; }

# $3 (optional) = "read-arena": let the controller read its Linux mapping of the
# region after the call. Only safe when the probe leaves the arena's revocation
# node live -- otherwise the monitor's swap_cpmp() reloads a revoked capability
# from regions[] and cap_base()'s lcc aborts QEMU. See README.md, "Gap found".
smoke() { # $1=share dir  $2=probe  $3=extra guest argv  rest: extra harness args
  local share="$1" name="$2" guest_arg="$3"
  shift 3
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
    --share-dir "$share" \
    --log-file "$share/$name.log" \
    --guest-command "/mnt/host/intra_domain_mrev_revoke_probe.user /mnt/host/$name.dom $guest_arg" \
    "$@"
}

run_ok() { # $1=share  $2=probe  $3=expected retval  $4=optional "read-arena"
  local share="$1" name="$2" retval="$3" guest_arg="${4:-}"
  local marker="intra-domain-mrev-revoke-probe: call retval = $retval"
  local attempt=0 rc
  while :; do
    attempt=$((attempt + 1))
    set +e
    smoke "$share" "$name" "$guest_arg" --success-marker "$marker" >/dev/null 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      echo "PASS  $name  (retval $retval)"
      return 0
    fi
    if [[ $rc -eq $INFRA_FLAKE_EXIT && $attempt -le $RETRIES ]]; then
      echo "  ...infra flake on $name (attempt $attempt), retrying" >&2
      continue
    fi
    echo "FAIL  $name  (rc=$rc; see $share/$name.log)" >&2
    return 1
  done
}

run_fault() { # $1=share  $2=probe  $3=expected diagnostic  $4=expected cause
  local share="$1" name="$2" msg="$3" want="$4"
  local log="$share/$name.log"
  local attempt=0 cause
  while :; do
    attempt=$((attempt + 1))
    set +e
    smoke "$share" "$name" "" >/dev/null 2>&1 # exit code ignored by design
    set -e
    if grep -q "domain halted by capability fault" "$log" 2>/dev/null; then
      cause=$(grep -oE 'cause = [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
      if [[ "$cause" == "$want" ]] && grep -q "$msg" "$log" 2>/dev/null; then
        echo "PASS  $name  (fault: '$msg', cause = $cause)"
        return 0
      fi
      echo "FAIL  $name  (faulted with cause $cause, expected $want -- wrong reason; see $log)" >&2
      return 1
    fi
    if grep -q "intra-domain-mrev-revoke-probe: call retval" "$log" 2>/dev/null; then
      echo "FAIL  $name  (domain returned instead of faulting -- revoke missed; see $log)" >&2
      return 1
    fi
    if [[ $attempt -le $RETRIES ]]; then
      echo "  ...no boot/fault for $name (attempt $attempt), retrying" >&2
      continue
    fi
    echo "FAIL  $name  (no fault line after $attempt attempts; see $log)" >&2
    return 1
  done
}

fail=0
for opt in $OPT_LEVELS; do
  share="$TMP_ROOT/intra-domain-mrev-revoke-share$opt"
  rm -rf "$share"
  mkdir -p "$share"

  echo "== building controller + domains at $opt =="
  DOMAIN_OPT_LEVEL="$opt" bash "$SCRIPT_DIR/build-intra-domain-mrev-revoke-probe.sh" \
    "$share" >/dev/null

  echo "== running probes at $opt (one boot each) =="
  # The mechanism: a cached alias of a revoked, monitor-granted capability faults.
  want_cause=$(primary_cause "$opt")
  [[ "$want_cause" == 25 ]] && want_msg="$REVOKED" || want_msg="$UNTAGGED"
  run_fault "$share" held_revoke_fault "$want_msg" "$want_cause" || fail=1
  # Memory-resident alias: the reload observes the revoked node and drops the tag.
  run_fault "$share" held_mem_alias_fault "$UNTAGGED" 24 || fail=1
  # Control for both cause-24 expectations: no revoke, and the deref succeeds.
  # read-arena: the host mmap independently sees the sentinel the domain wrote
  # through the granted capability -- the delivery really lands in the region.
  run_ok "$share" held_no_revoke_ok 0x2230005e read-arena || fail=1
  # The sweep is not over-broad: unrelated authority survives the domain's revoke.
  run_ok "$share" held_unrelated_ok 0x22310033 || fail=1
  # Provenance: the arena has NO ambient second path -- a forged address traps.
  run_fault "$share" held_ambient_miss "$UNTAGGED" 24 || fail=1
  # Sub-allocations of one grant are independently revocable (SPLIT, not SHRINK).
  # Only the high half is revoked, so the low half's node keeps regions[] valid.
  run_ok "$share" held_split_sibling_ok 0x22350044 read-arena || fail=1

  # Step-3 scaffold: the SQLite lifecycle. A carved protected value is revoked at
  # "finalize" and the caller's cached handle faults...
  run_fault "$share" held_protected_value_lifecycle "$want_msg" "$want_cause" || fail=1
  # ...while a second live value and the un-carved arena remainder keep working.
  run_ok "$share" held_arena_survives_revoke 0x22370077 read-arena || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_INTRA_DOMAIN_MREV_REVOKE_PASSED__"
else
  echo "one or more probes FAILED" >&2
fi
exit $fail
