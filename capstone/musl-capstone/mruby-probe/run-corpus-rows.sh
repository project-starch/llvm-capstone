#!/usr/bin/env bash
# Corpus rows on the real interpreter: three allocator arms each, ONE boot.
#
#   ROWS="8 9 14" bash run-corpus-rows.sh
#
# Each row's mruby comes from its OWN pinned tree, xlang/repro/<n>/mruby, because
# the corpus pins a different vulnerable commit per row and building row 8's
# trigger against row 10's interpreter would measure the wrong program. The tree
# needs one `rake` first, to produce build/host.
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
# ONE ROW PER BOOT, and the faulting arm LAST. A real limit, not caution: when a
# domain takes a capability fault the servicer does not return -- it waits on a
# domain that will never answer -- so the guest never gets its prompt back and
# the DRIVER abandons the whole boot. A fault therefore ends a run exactly like a
# wedge. Measured 2026-08-15: a twelve-arm batch across four rows produced ONE.
#
# ROWS still accepts several rows and stages them correctly, which is useful for
# building a set in advance, but only the arms up to the first fault will run.
#
# A reaper around the servicer would lift this. `timeout` is the obvious one and
# busybox here is built WITHOUT it (CONFIG_TIMEOUT is not set), which was found
# the expensive way: the wrapper became "timeout: not found", every arm returned
# instantly having run nothing, and the batch read as if it had completed. Shell
# job control would work; not worth the fragility until batching pays.
#
# A WEDGE is unreapable in any case, so an arm that might wedge -- anything using
# a LARGE host region, ISSUES.md I-2 -- must go last or in its own boot.
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

ROWS=${ROWS:-${ROW:-10}}
STACK_DOUBLING=${STACK_DOUBLING:-1}
ARMS=${ARMS:-"libc ctl rev"}
SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/rows-share}
LOG=${LOG_FILE:-$CAPSTONE_TMP_ROOT/rows-arms.log}

# The LOG goes too, not just the share dir: a leftover log from an earlier
# session reads exactly like a result, complete with an older probe's arm
# labels, and was tailed for two minutes here before a timestamp gave it away.
rm -rf "$SHARE"; mkdir -p "$SHARE"; rm -f "$LOG"

# Gems a row's TRIGGER needs, beyond core mruby. Without them the trigger raises
# NoMethodError before reaching its offending access -- a run where every arm
# reports identically and none of them means anything. Row 14 spent a boot on
# exactly that ("undefined method '%' for \"%d\"", from mruby-sprintf).
# Overridable with GEMS=..., empty for rows that need none.
row_gems() {
  case $1 in
    14|15) echo "sprintf" ;;
    8|13)  echo "hash-ext" ;;
    9)     echo "eval print" ;;
    11)    echo "print" ;;
    6)     echo "print" ;;
    12)    echo "io print" ;;
    *)     echo "" ;;
  esac
}

arm_env() {
  case $1 in
    libc) echo "" ;;
    ctl)  echo "MRUBY_PROBE_REVOKE=1 MRUBY_PROBE_CDP_CONTROL=1" ;;
    rev)  echo "MRUBY_PROBE_REVOKE=1" ;;
    *)    echo "unknown arm $1" >&2; return 1 ;;
  esac
}

TAGS=()
for ROW in $ROWS; do
  # Each row against ITS OWN pinned tree. Overridable, but the default is the
  # corpus's, because a row measured against another row's interpreter is a
  # different program.
  row_src=${MRUBY_SRC:-$CAPSTONE_REPO_ROOT/xlang/repro/$ROW/mruby}
  [[ -f "$row_src/build/host/bin/mrbc" ]] || {
    echo "row $ROW: no host build in $row_src -- run 'rake' there first" >&2
    exit 2; }
  echo "building row $ROW arms from $(basename "$(dirname "$row_src")")/mruby (doubling=$STACK_DOUBLING):"
  for a in $ARMS; do
    extra=$(arm_env "$a") || exit 2
    tag="r${ROW}_$a"
    # The .dom is removed FIRST and the build log is NOT hidden: three runs in
    # one session executed a STALE image because a build failed quietly and the
    # old .dom was still in the share directory, reporting as if it were new.
    rm -f "$SHARE/$tag.dom"
    gems=${GEMS-$(row_gems "$ROW")}
    if ! env $extra MRUBY_SRC="$row_src" MRUBY_WITH_PARSER=1 MRUBY_PROBE_ROW="$ROW" \
           ${gems:+MRUBY_PROBE_GEMS="$gems"} \
           ${STACK_DOUBLING:+MRUBY_PROBE_STACK_DOUBLING=$STACK_DOUBLING} \
           OUT_DIR="$CAPSTONE_TMP_ROOT/$tag" \
           OUT_DOM="$SHARE/$tag.dom" OUT_HOST="$SHARE/$tag.user" \
           bash "$SCRIPT_DIR/build-mruby-probe.sh" \
           > "$CAPSTONE_TMP_ROOT/build-$tag.log" 2>&1; then
      echo "BUILD FAILED for row $ROW arm $a:" >&2
      grep -E "error:|Assertion|Cannot select|patch-embed-len|ERROR" \
           "$CAPSTONE_TMP_ROOT/build-$tag.log" | head -5 >&2
      exit 2
    fi
    # Not redundant with the exit status: a build can exit 0 and produce nothing.
    [[ -f "$SHARE/$tag.dom" ]] || { echo "$tag: exit 0 but no .dom" >&2; exit 2; }
    printf '  %-10s %s bytes\n' "$tag" "$(stat -c%s "$SHARE/$tag.dom")"
    TAGS+=("$tag")
  done
done

CMDS=(--guest-command 'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__')
for tag in "${TAGS[@]}"; do
  # Copied to /tmp before running: read straight off the 9p share the loader
  # demand-pages a 2 MB image and it never finishes. Removed again afterwards --
  # a dozen domains at ~2 MB each would otherwise fill the guest's tmpfs, and
  # THAT failure looks like a wedge.
  CMDS+=(--guest-command "echo ===ARM_$tag===; cp /mnt/host/$tag.user /tmp/h && chmod 0755 /tmp/h; cp /mnt/host/$tag.dom /tmp/d && /tmp/h /tmp/d; rm -f /tmp/d /tmp/h; echo ===END_$tag===")
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
