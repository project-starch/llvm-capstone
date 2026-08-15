#!/usr/bin/env bash
# Row 10 (CVE-2022-1106) on the real interpreter: three allocator arms, ONE boot.
#
#   libc  libc-ext/malloc.c        no per-allocation bounds, no revocation
#   ctl   rof + xlang_set_no_revoke  exact bounds, revocation OFF   <- THE CONTROL
#   rev   rof, revoke-on-free        the shipped configuration
#
# THE CONTROL IS WHAT MAKES THIS A MEASUREMENT, not the fault. It is the same
# build and the same workload, differing from `rev` in one call. If it runs the
# whole trigger and returns, then mrb_range_new returned, so the stale write at
# vm.c:2822 was executed and nothing stopped it -- and only then does `rev`
# faulting mean revocation stopped it rather than the trigger never arming.
#
# `libc` is the third arm and is not redundant: it has NEITHER property, so it
# separates "revocation caught it" from "exact bounds caught it". A two-arm
# design cannot, because both of its arms share the bounds.
#
# ORDER: every arm expected to RETURN first, the expected-to-fault arm last.
# (A fault does not end the boot here -- only a wedge does -- but an arm that
# might wedge, such as anything using a LARGE host region, must go last or in
# its own boot. ISSUES.md I-2: a 16 MiB arena arm printed nothing at all and
# took four staged arms with it.)
#
# STACK_DOUBLING=1 (default) sets mruby's upstream MRB_STACK_EXTEND_DOUBLING.
# It is NOT mruby's default and must be reported with any number taken under it.
# Without it the revoking arena cannot hold the workload: mruby's linear stack
# growth makes the CUMULATIVE carve quadratic, 4,122,752 of a 4,194,304 arena,
# and the interpreter dies at 52 of 151 levels with SystemStackError. With it,
# 1,556,800. Set STACK_DOUBLING=0 to measure that wall instead.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

ROW=${ROW:-10}
STACK_DOUBLING=${STACK_DOUBLING:-1}
ARMS=${ARMS:-"libc ctl rev"}
SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/row$ROW-share}
LOG=${LOG_FILE:-$CAPSTONE_TMP_ROOT/row$ROW-arms.log}

# The LOG goes too, not just the share dir: a leftover log from an earlier
# session reads exactly like a result, complete with an older probe's arm
# labels, and was tailed for two minutes here before a timestamp gave it away.
rm -rf "$SHARE"; mkdir -p "$SHARE"; rm -f "$LOG"

arm_env() {
  case $1 in
    libc) echo "" ;;
    ctl)  echo "MRUBY_PROBE_REVOKE=1 MRUBY_PROBE_CDP_CONTROL=1" ;;
    rev)  echo "MRUBY_PROBE_REVOKE=1" ;;
    *)    echo "unknown arm $1" >&2; return 1 ;;
  esac
}

echo "building row $ROW arms (doubling=$STACK_DOUBLING):"
for a in $ARMS; do
  extra=$(arm_env "$a") || exit 2
  # The .dom is removed FIRST and the build log is NOT hidden: three runs in one
  # session executed a STALE image because a build failed quietly and the old
  # .dom was still in the share directory, reporting as if it were the new arm.
  rm -f "$SHARE/$a.dom"
  if ! env $extra MRUBY_WITH_PARSER=1 MRUBY_PROBE_ROW="$ROW" \
         ${STACK_DOUBLING:+MRUBY_PROBE_STACK_DOUBLING=$STACK_DOUBLING} \
         OUT_DIR="$CAPSTONE_TMP_ROOT/row$ROW-$a" \
         OUT_DOM="$SHARE/$a.dom" OUT_HOST="$SHARE/$a.user" \
         bash "$SCRIPT_DIR/build-mruby-probe.sh" \
         > "$CAPSTONE_TMP_ROOT/row$ROW-build-$a.log" 2>&1; then
    echo "BUILD FAILED for arm $a:" >&2
    grep -E "error:|Assertion|Cannot select" \
         "$CAPSTONE_TMP_ROOT/row$ROW-build-$a.log" | head -5 >&2
    exit 2
  fi
  # Not redundant with the exit status: a build can exit 0 and produce nothing.
  [[ -f "$SHARE/$a.dom" ]] || { echo "arm $a: exit 0 but no .dom" >&2; exit 2; }
  printf '  %-5s %s bytes\n' "$a" "$(stat -c%s "$SHARE/$a.dom")"
done

CMDS=(--guest-command 'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__')
for a in $ARMS; do
  # Copied to /tmp before running: read straight off the 9p share the loader
  # demand-pages a 2 MB image and it never finishes.
  CMDS+=(--guest-command "echo ===ARM_$a===; cp /mnt/host/$a.user /tmp/h_$a && chmod 0755 /tmp/h_$a; cp /mnt/host/$a.dom /tmp/d_$a && /tmp/h_$a /tmp/d_$a; echo ===END_$a===")
done

# NO --success-marker: the expected outcome is that one arm FAULTS, so any
# marker list would fail and say nothing about which arm did what. Read the log.
for attempt in 1 2 3; do
  CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300} python3 \
    "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro loglevel=1" \
    --share-dir "$SHARE" --log-file "$LOG" \
    --timeout-multiplier "${TIMEOUT_MULTIPLIER:-16}" "${CMDS[@]}"
  status=$?
  [[ $status -ne 75 ]] && break
  echo "infra flake on attempt $attempt (exit 75), retrying" >&2
done

echo "driver exit=$status"
echo "=== what each arm reported ==="
grep -aE "===ARM_|ROW ARM|ROW: |MRUBY ARENA after-row|capability fault" "$LOG" | grep -av '^#'
echo "Full serial log: $LOG"
