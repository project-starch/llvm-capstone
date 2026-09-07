#!/usr/bin/env bash
# The Q-03 module-consistency check, as one QEMU boot: N domains created through the check loader
# (q03_region_hole_check_host.c, built by build-q03-region-hole-check.sh), the last of them named
# `*chk*` so the loader runs its consistency block after it: the module's region count against the
# monitor's REGION_COUNT (holes included), a fresh region's query/mmap/write, an out-of-range share
# (must be rejected: retval 4294967295), and the module's own reuse/fetch-failure counters.
#
# This is the only instrument that caught the `||` non-short-circuit regression in the hole guards
# on 2026-09-05 (ISSUES.md Q-03): the manifest replays could not see it. Expected line shape:
#   Q03CHK count=N ... region id=M qlen=4096 mmap=ok write=1 ... oob_share id=... retval=4294967295
#   ... dmesg reuse=0 fetchfail=0
#
# Usage: run-q03-region-hole-check.sh [-n ITEMS] [DOMAIN.dom]
#   DOMAIN defaults to the smoke test's write_42 domain (built here if absent). It must be an image the
#   plain loader runs: a ladder rung built for the lpc host (k800 and the BEEBS rungs) faults at once
#   under create_dom/call_dom. The batch runner copies the image under ITEMS distinct names. Reads
#   CAPSTONE_BUILDROOT_DIR for the images and the guest toolchain. Take the QEMU lock yourself or set
#   CAPSTONE_QEMU_LOCK_HELD=1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
ITEMS=8
if [[ "${1:-}" == "-n" ]]; then ITEMS=$2; shift 2; fi
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
DOM=${1:-$TMP_ROOT/q03-region-hole-check/write_42.dom}
if [[ ! -f "$DOM" && -z "${1:-}" ]]; then mkdir -p "$(dirname "$DOM")"; bash "$SCRIPT_DIR/build-domain.sh" "$SCRIPT_DIR/domains/write_42.c" "$DOM" >/dev/null; fi
[[ -f "$DOM" ]] || { echo "no domain image at $DOM" >&2; exit 2; }
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/q03-region-hole-check-share}
OUT_DIR=${OUT_DIR:-$TMP_ROOT/q03-region-hole-check}
mkdir -p "$SHARE_DIR" "$OUT_DIR"
rm -f "$SHARE_DIR"/q03_region_hole_check.user "$OUT_DIR"/manifest.tsv "$OUT_DIR"/batch.log "$OUT_DIR"/results.tsv
bash "$SCRIPT_DIR/build-q03-region-hole-check.sh" "$SHARE_DIR" >/dev/null
for i in $(seq 1 $((ITEMS - 1))); do printf 'item%02d\t%s\n' "$i" "$DOM"; done > "$OUT_DIR/manifest.tsv"
printf 'chk%02d\t%s\n' "$ITEMS" "$DOM" >> "$OUT_DIR/manifest.tsv"
python3 "$SCRIPT_DIR/../fuzz/run-domain-batch.py" --manifest "$OUT_DIR/manifest.tsv" --share "$SHARE_DIR" \
  --log "$OUT_DIR/batch.log" --out "$OUT_DIR/results.tsv" --per-item-timeout 45 --max-reboots 2 \
  --loader /mnt/host/q03_region_hole_check.user >/dev/null 2>&1 || true
echo "items: $(grep -c . "$OUT_DIR/results.tsv") rows; RET: $(grep -c $'\tRET\t' "$OUT_DIR/results.tsv"); holes printed (0x1236): $(grep -a -c 'Scalar(0x1236)' "$OUT_DIR/batch.log")"
grep -a -o 'Q03CHK .*' "$OUT_DIR/batch.log" | sed 's/\r$//' | tr '\n' ';' | cut -c1-400; echo
# The check block ran, the out-of-range share was rejected, nothing was reused or refetched.
grep -a -q 'Q03CHK count=' "$OUT_DIR/batch.log" || { echo "__Q03_CHECK_DID_NOT_RUN__"; exit 1; }
grep -a -q 'oob_share id=[0-9]* retval=4294967295' "$OUT_DIR/batch.log" || { echo "__Q03_OOB_SHARE_NOT_REJECTED__"; exit 1; }
grep -a -q 'reuse=0 fetchfail=0' "$OUT_DIR/batch.log" || { echo "__Q03_MODULE_COUNTERS_NONZERO__"; exit 1; }
echo "__CAPSTONE_Q03_REGION_HOLE_CHECK_PASSED__"
