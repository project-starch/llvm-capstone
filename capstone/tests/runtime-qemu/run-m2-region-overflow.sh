#!/usr/bin/env bash
# M-2 positive control in one QEMU boot: the host program creates N regions (default 72 > the
# module's former 64-entry table, no domains), then queries/maps/writes the ones past 64 and counts
# kernel Oops/BUG/WARNING lines. PASS = every region past 64 has qlen 4096, maps, writes, and dmesg
# is clean; the program prints __M2_REGION_OVERFLOW_PASSED__. On an unfixed module this must FAIL
# (wrong lengths / NULL maps / a kernel oops) -- run it once BEFORE the fix to prove it fires.
#
# Needs the monitor's CAPSTONE_MAX_REGION_N >= N (Phase B item 3: 96 on both targets). Take the
# QEMU lock yourself or set CAPSTONE_QEMU_LOCK_HELD=1. Reads CAPSTONE_BUILDROOT_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
N=${1:-72}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/m2-region-overflow-share}
LOG=${LOG_FILE:-$TMP_ROOT/m2-region-overflow.log}
mkdir -p "$SHARE_DIR"
bash "$SCRIPT_DIR/build-m2-region-overflow.sh" "$SHARE_DIR" >/dev/null
set +e
python3 "$SCRIPT_DIR/run-domain-smoke.py" --share-dir "$SHARE_DIR" --log-file "$LOG" \
  --guest-command "/mnt/host/m2_region_overflow.user $N" \
  --success-marker "__M2_REGION_OVERFLOW_PASSED__" >/dev/null 2>&1; rc=$?
set -e
# (not `[^\r]*`: inside a bracket expression that is "not backslash, not r" and truncates at the first r)
grep -a '^M2 ' "$LOG" | tr -d '\r' | head -40
if grep -a -q '__M2_REGION_OVERFLOW_PASSED__' "$LOG"; then echo "__CAPSTONE_M2_REGION_OVERFLOW_PASSED__"; exit 0; fi
grep -a -q 'M2 created=' "$LOG" || { echo "__M2_CONTROL_DID_NOT_RUN__ (rc=$rc; see $LOG)"; exit 2; }
echo "__CAPSTONE_M2_REGION_OVERFLOW_FAILED__ (rc=$rc; see $LOG)"; exit 1
