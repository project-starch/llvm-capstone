#!/usr/bin/env bash
# Replay what real nginx asked its pool for, in a domain, on either level below.
#
#   run-nginx-replay.sh <trace.ngxt>
#   NGX_SUBLET=1 run-nginx-replay.sh <trace.ngxt>
#
# The trace comes from the recorder in the paper's experiments/a11/nginx, flattened and cut by its
# flatten.py. Both arms run the same file, so a difference between them is the discipline and not
# the workload.
#
# The gate is not "it returned". A replay that skipped every record would return. It is: no
# failure, no identity table that ran out, nothing left at the level below, and a count of
# executed records that matches what the file holds.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"
TRACE=${1:?usage: run-nginx-replay.sh <trace.ngxt>}
[ -f "$TRACE" ] || { echo "no trace at $TRACE" >&2; exit 1; }
DOM_NAME=${DOM_NAME:-ngx-replay}; export DOM_NAME   # or the build writes a different name
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/nginx-domain}
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}; mkdir -p "$SHARE"
ARENA=${NGX_ARENA_BYTES:-$((1<<20))}
if [ "${NGX_SUBLET:-0}" = 1 ]; then NGX_ARENA_LINEAR=1; fi
export NGX_SUBLET NGX_ARENA_LINEAR NGX_SUBLET_BLOCK

# Whatever a previous run left, BEFORE the build. A build that fails must not leave a stale image
# for the boot to pick up, and one that writes a different name must not leave the old one to be
# copied in its place. That has happened, and the stale image was from the other arm.
rm -f "$SHARE/$DOM_NAME.dom" "$SHARE/ngx-replay-guest" "$SHARE/trace.ngxt"
NGX_DOMAIN=replay bash "$SCRIPT_DIR/build-nginx-domain.sh" || exit 1
[ -f "$OUT_DIR/$DOM_NAME.dom" ] || { echo "the build produced no $DOM_NAME.dom" >&2; exit 1; }
cp "$OUT_DIR/$DOM_NAME.dom" "$SHARE/"
cp "$TRACE" "$SHARE/trace.ngxt"

GCC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
U=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace
"$GCC" -O2 -I "$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" -I "$U" \
    -o "$SHARE/ngx-replay-guest" "$SCRIPT_DIR/tools/ngx-replay-guest.c" "$U/lib/libcapstone.c" || exit 1

# cma= has to cover every region the guest makes, because a region above the buddy allocator's
# four megabytes comes from there, plus the kernel's own use of the area.
TRACE_MB=$(( ($(stat -c %s "$TRACE") + 1048575) / 1048576 + 1 ))
CMA=${NGX_CMA:-$(( TRACE_MB + ARENA / 1048576 + 1 + 64 ))M}

rm -f "$OUT_DIR/boot.log"
"${PYTHON:-python3}" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$OUT_DIR/boot.log" --timeout-multiplier 40 \
  --kernel-arg "cma=$CMA" \
  --guest-command "/mnt/host/ngx-replay-guest /mnt/host/$DOM_NAME.dom /mnt/host/trace.ngxt${NGX_ARENA_LINEAR:+ --arena-linear} --arena $ARENA" \
  --success-marker "replay tables" > "$OUT_DIR/replay.txt" 2>&1 || true

if ! grep -q 'replay tables' "$OUT_DIR/boot.log" 2>/dev/null; then
  echo "no result; $OUT_DIR/boot.log says why" >&2
  grep -m1 'domain halted\|ngx retval\|no result block' "$OUT_DIR/boot.log" >&2 || tail -3 "$OUT_DIR/replay.txt" >&2
  exit 1
fi
grep -E '^(trace |replay )' "$OUT_DIR/boot.log"

val() { grep -m1 "^replay $1 " "$OUT_DIR/boot.log" | awk '{print $NF}'; }
fail=0
[ "$(val failures)" = 0 ] || { echo "failures: $(val failures)" >&2; fail=1; }
[ "$(val tables)" = 0 ]   || { echo "an identity table ran out, so the run means nothing" >&2; fail=1; }
[ "$(val level0)" = 0 ]   || { echo "the level below still holds $(val level0)" >&2; fail=1; }
ex=$(grep -m1 '^replay executed' "$OUT_DIR/boot.log" | awk '{print $NF}')
rc=$(grep -m1 '^replay records' "$OUT_DIR/boot.log" | awk '{print $NF}')
sk=$(grep -m1 '^replay skipped' "$OUT_DIR/boot.log" | awk '{print $NF}')
[ "$ex" -gt 0 ] 2>/dev/null || { echo "nothing was executed" >&2; fail=1; }
[ $(( ex + sk )) -eq "$rc" ] || { echo "executed plus skipped is $(( ex + sk )), the file holds $rc" >&2; fail=1; }
exit $fail
