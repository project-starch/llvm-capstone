#!/usr/bin/env bash
# Build the mruby image and walk the whole ladder in ONE boot.
#
#     tools/gen-specimen.sh cases/smoke.rb && ./run-ladder.sh
#     MRUBY_NO_NARROW=1 ./run-ladder.sh          # the wide-bounds arm
#
# ONE IMAGE, not one per stage. The stages live in a runtime counter inside
# domain_main (see port/mruby_domain.c), so the loader invoking the domain N times
# walks the ladder -- against five builds of an 85000-line translation unit at eight
# to ten minutes each, which is what the compile-time switch cost.
#
# One boot, because booting costs two to three minutes and dominates a short run.
# Ascending, because a fault takes the domain with it and everything after the
# first one is lost -- which is not a limitation to engineer around: the first call
# that fails to return IS the bisection point. Every call returns a marker, so a run
# always yields data, and a call that returns nothing is itself the result.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

CALLS=${CALLS:-59}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/mruby-silicon}
SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}
LOG=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-mruby.log}
DOM_NAME=${DOM_NAME:-mruby}
# The clamped build does far MORE work than the faulting one: mrb_open_core runs to
# completion instead of dying partway, which is thousands of allocations plus all of
# mrblib under a capability-checking QEMU. The first probe run timed out at 6 and the
# message called it "QEMU stopped", which reads like a crash.
TMULT=${TMULT:-30}
mkdir -p "$SHARE"

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
  echo "== building $DOM_NAME (narrowing: $([[ ${MRUBY_NO_NARROW:-0} == 1 ]] && echo OFF || echo on))"
  DOM_NAME="$DOM_NAME" bash "$SCRIPT_DIR/build-mruby-silicon.sh"
fi
cp -f "$OUT_DIR/$DOM_NAME.dom" "$SHARE/"

echo "== one boot, $CALLS calls ascending"
CAPSTONE_QUIET_GP=1 python3 "$REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier "$TMULT" \
  --guest-command "/mnt/host/capstone-test.user /mnt/host/$DOM_NAME.dom $CALLS; " \
  "$SHARE/$DOM_NAME.dom" || true

python3 "$SCRIPT_DIR/tools/report-ladder.py" "$LOG"
