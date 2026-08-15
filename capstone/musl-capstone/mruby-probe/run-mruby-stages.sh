#!/usr/bin/env bash
# Run the staged mruby arms in ONE boot.
#
# WHY THIS EXISTS. mrb_open wedges: it neither returns nor faults. Nine boots
# were spent on it one hypothesis at a time, which is precisely what CLAUDE.md's
# "BATCH VARIANTS, and make every run RETURN" section was written to prevent.
# These arms replicate mruby's own init decomposition and each RETURNS a marker:
#
#   stage 1  allocate and zero mrb_state                    expected: returns
#   stage 2  + mrb_gc_init                                  expected: returns
#   stage 3  + mrb_init_core                                expected: WEDGES
#   stage 0  full mrb_open_allocf, then the bytecode        last, so a wedge
#                                                           above costs nothing
#
# ORDER IS THE SAFETY PROPERTY. A wedge ends the boot, so every arm expected to
# return goes first and at most ONE arm is expected not to. If stage 3 wedges,
# stages 1 and 2 have already reported and only stage 0 is lost -- and stage 0
# would have wedged in the same place anyway.
#
# Each arm also prints its allocation count, so the batch measures how much of
# mruby's init happens where. The heartbeat already brackets the wedge to
# allocations 257..320; this says which init call that is inside.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-mruby}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/stage-share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-mruby-stages.log}
STAGES=${STAGES:-"1 2 3 0"}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR"/*.dom "$SHARE_DIR"/*.user

for stage in $STAGES; do
  # THE .dom IS REMOVED FIRST, and the build's output is NOT hidden. Three runs
  # in this session executed a STALE image because a build failed while its
  # output went to /dev/null and the previous .dom was still lying in the share
  # directory -- and a stale arm reports as if it were the new one, which is the
  # worst possible failure mode for a bisection.
  rm -f "$SHARE_DIR/stage$stage.dom"
  MRUBY_PROBE_STAGE="$stage" OUT_DIR="$OUT_DIR" \
    OUT_DOM="$SHARE_DIR/stage$stage.dom" OUT_HOST="$SHARE_DIR/mruby_probe.user" \
    bash "$SCRIPT_DIR/build-mruby-probe.sh" > "$OUT_DIR/build-stage$stage.log" 2>&1 || {
      echo "BUILD FAILED for stage $stage:" >&2
      grep -E "error:|Assertion" "$OUT_DIR/build-stage$stage.log" | head -5 >&2
      exit 2
    }
  [[ -f "$SHARE_DIR/stage$stage.dom" ]] || {
    echo "build reported success but produced no stage$stage.dom" >&2; exit 2; }
  printf 'staged arm %s: %s\n' "$stage" \
    "$(stat -c%s "$SHARE_DIR/stage$stage.dom") bytes"
done

# One guest command per arm, all in the same boot. The .dom is copied to /tmp
# first: read straight off the 9p share the loader demand-pages it and a 1.3 MB
# image never finishes; copied, it loads in under a second (see
# run-mruby-probe.sh for the measurement).
GUEST_CMDS=()
GUEST_CMDS+=(--guest-command 'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/mruby_probe.user /tmp/h && chmod 0755 /tmp/h')
for stage in $STAGES; do
  GUEST_CMDS+=(--guest-command "date +S${stage}_START=%s; cp /mnt/host/stage$stage.dom /tmp/d$stage && /tmp/h /tmp/d$stage; date +S${stage}_END=%s")
done

run_smoke() {
  local attempt status
  for attempt in 1 2 3; do
    set +e
    "$@"
    status=$?
    set -e
    [[ $status -eq 0 ]] && return 0
    [[ $status -ne 75 ]] && return "$status"
    echo "run: infra flake on attempt $attempt (exit 75), retrying" >&2
  done
  return 75
}

# NO --success-marker on purpose. The batch is a MEASUREMENT, not a pass/fail
# gate: the expected outcome is that one arm wedges, which would make any marker
# list fail and say nothing about which arm. Read the log.
set +e
CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300} \
  run_smoke python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro loglevel=1" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-10}" \
  "${GUEST_CMDS[@]}"
set -e

echo "=== what each arm reported ==="
grep -aE "MRUBY STAGE|MRUBY TICK|S[0-9]_(START|END)=|capability fault" "$LOG_FILE" || true
echo "Full serial log: $LOG_FILE"
