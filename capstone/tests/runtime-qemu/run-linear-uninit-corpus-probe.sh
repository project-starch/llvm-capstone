#!/usr/bin/env bash
set -euo pipefail

# SQLite Stage-2 corpus rows 11 (LINEAR, double-free) and 14 (UNINIT,
# use-before-init), as mechanism probes on RTL.
#
# Both rows were deferred on 2026-07-08 as "blocked intra-domain"
# (agent-handoff/history/08-07-2026_13-01-23_linear-uninit-rows-blocked-intra-domain.md).
# Both blockers are gone:
#   row11 needed csdrop, which the emulator did not implement -- added since.
#   row14 needed a domain-held LINEAR capability to revoke, which the .smode
#         scaffold could not provide -- the domain_main receive protocol does
#         (../intra-domain-mrev-revoke-probe). No monitor op was needed: revoking
#         a still-linear lineage yields the UNINIT handle directly.
#
# ASSERTION STYLE (inherited from the held-cap probe):
#   - OK probes RETURN a value; the controller prints it and we match the marker.
#   - FAULT probes halt the domain. A domain_main .dom runs in PRV_C, whose
#     capability faults have no delivery path, so QEMU prints the fault line and
#     exits; the guest never returns to the shell and the harness exits non-zero
#     BY DESIGN. The evidence is the fault line in the serial log.
#   Each probe gets its OWN boot: a faulted domain poisons later domain creation
#   in the same guest session, and --success-marker is per-invocation.
#
# THE EXPECTED CAUSE IS ASSERTED, NOT JUST REPORTED, and -- unlike the held-cap
# probe -- it does not move with the optimisation level. Neither row depends on
# whether the optimiser keeps a capability in a register:
#   row14's UNINIT handle keeps its tag and its revocation node, so the fault is
#     always cause 26, raised by the capability's TYPE. Self-proving: nothing
#     else raises 26.
#   row11's dropped handle is untagged in a register and stays untagged through a
#     spill/reload, so the fault is always cause 24. Cause 24 only says "no
#     capability here", so linear_no_drop_ok is its control.
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)
# and confirm the other agent is not mid-run before invoking this.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OPT_LEVELS=${OPT_LEVELS:--O0 -O1 -O2}
RETRIES=${RETRIES:-2}
INFRA_FLAKE_EXIT=75

UNINIT="Cap mem load through uninitialised capability" # cause 26: no read authority
UNTAGGED="Cap mem access requires capability"          # cause 24: deref, tag gone
DROPPED="DROP requires capability"                     # cause 24: drop of a dropped handle

# $3 (optional) = "read-arena": let the controller read its Linux mapping of the
# region after the call. Only safe when the probe leaves the arena's revocation
# node live -- so never for a row14 probe, all of which revoke it.
smoke() { # $1=share dir  $2=probe  $3=extra guest argv  rest: extra harness args
  local share="$1" name="$2" guest_arg="$3"
  shift 3
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
    --share-dir "$share" \
    --log-file "$share/$name.log" \
    --guest-command "/mnt/host/linear_uninit_corpus_probe.user /mnt/host/$name.dom $guest_arg" \
    "$@"
}

run_ok() { # $1=share  $2=probe  $3=expected retval  $4=optional "read-arena"
  local share="$1" name="$2" retval="$3" guest_arg="${4:-}"
  local marker="linear-uninit-corpus-probe: call retval = $retval"
  local log="$share/$name.log"
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
    # A boot that dies before the domain returns leaves NO "call retval" line at
    # all; a domain that returned the wrong value leaves one. Retrying the former
    # cannot mask a real failure. The harness only reports its own flake exit for
    # the setup phases, so a truncated guest command lands here as rc=1.
    if [[ $attempt -le $RETRIES ]] &&
       ! grep -q "linear-uninit-corpus-probe: call retval" "$log" 2>/dev/null; then
      echo "  ...no boot/retval for $name (attempt $attempt), retrying" >&2
      continue
    fi
    echo "FAIL  $name  (rc=$rc; see $log)" >&2
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
    if grep -q "linear-uninit-corpus-probe: call retval" "$log" 2>/dev/null; then
      echo "FAIL  $name  (domain returned instead of faulting; see $log)" >&2
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
  share="$TMP_ROOT/linear-uninit-corpus-share$opt"
  rm -rf "$share"
  mkdir -p "$share"

  echo "== building controller + domains at $opt =="
  DOMAIN_OPT_LEVEL="$opt" bash "$SCRIPT_DIR/build-linear-uninit-corpus-probe.sh" \
    "$share" >/dev/null

  echo "== row14 UNINIT (use-before-init) at $opt =="
  # The mechanism: an uninitialised connection handle has no read authority.
  run_fault "$share" uninit_use_before_init_fault "$UNINIT" 26 || fail=1
  # ...and that is enforced by the capability's TYPE, not by where its cursor
  # sits: a negative offset is inside the bounds and still traps.
  run_fault "$share" uninit_negative_offset_fault "$UNINIT" 26 || fail=1
  # Control: csinit (sqlite3_open) reclaims the same bytes through the same
  # handle. Proves the trap was the type, not dead memory or a broken grant.
  run_ok "$share" uninit_init_then_use_ok 0x1412005e || fail=1

  echo "== row11 LINEAR (double-free) at $opt =="
  # The mechanism: the first finalize consumes the move-only statement handle,
  # and linearity leaves no second copy for a later use.
  run_fault "$share" linear_drop_use_fault "$UNTAGGED" 24 || fail=1
  # The literal shape: the second sqlite3_finalize itself faults, at the drop.
  run_fault "$share" linear_double_drop_fault "$DROPPED" 24 || fail=1
  # Control for both cause-24 expectations: the same carve, no drop, no fault.
  run_ok "$share" linear_no_drop_ok 0x11120033 || fail=1
  # csdrop consumes a handle; it does not revoke a lineage or free memory. The
  # connection the statement was carved from keeps working, and the host mmap
  # independently sees the byte written through it after the drop.
  run_ok "$share" linear_drop_sibling_ok 0x11130044 read-arena || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "__CAPSTONE_LINEAR_UNINIT_CORPUS_PASSED__"
else
  echo "one or more probes FAILED" >&2
fi
exit $fail
