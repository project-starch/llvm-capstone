#!/usr/bin/env bash
# Is a released region's mmap-offset window handed back, or leaked?
#
#   bash run-offsetcycle.sh [bytes]      # default 4 MiB -- at the buddy ceiling, no CMA needed
#
# THIS IS A TWO-DIRECTIONAL CHECK, which is why it is worth a QEMU boot rather than an assertion
# that the one-line fix is obviously right. Three create/query/release cycles at one size: with the
# fix every cycle reports the SAME mmap_offset; with the old module each reports one `size` higher,
# so a build carrying the unfixed module FAILS this program rather than passing it quietly.
#
# A release the monitor refuses frees nothing and restores nothing, so the probe treats any
# non-zero release as VOID and exits with its own marker instead of reporting a pass.
#
# Take the QEMU lock yourself or set CAPSTONE_QEMU_LOCK_HELD=1. Reads CAPSTONE_BUILDROOT_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
BYTES=${1:-4194304}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/offsetcycle-share}
LOG=${LOG_FILE:-$TMP_ROOT/offsetcycle-${BYTES}.log}
mkdir -p "$SHARE_DIR"
bash "$SCRIPT_DIR/build-offsetcycle.sh" "$SHARE_DIR" >/dev/null

set +e
python3 "$SCRIPT_DIR/run-domain-smoke.py" --share-dir "$SHARE_DIR" --log-file "$LOG" \
  --guest-command "/mnt/host/offsetcycle.user $BYTES" \
  --success-marker "__OFFSETCYCLE_PASSED__" >/dev/null 2>&1; rc=$?
set -e

grep -a -E '^OFFSETCYCLE' "$LOG" | tr -d '\r' | head -20
if grep -a -q '__OFFSETCYCLE_PASSED__' "$LOG"; then echo "__CAPSTONE_OFFSETCYCLE_PASSED__"; exit 0; fi
# A leak and a void run are different findings and must not be collapsed.
if grep -a -q '__OFFSETCYCLE_LEAKED__'           "$LOG"; then echo "__CAPSTONE_OFFSETCYCLE_LEAKED__ (see $LOG)"; exit 3; fi
if grep -a -q '__OFFSETCYCLE_RELEASE_REFUSED__'  "$LOG"; then echo "__CAPSTONE_OFFSETCYCLE_VOID_release_refused__ (see $LOG)"; exit 4; fi
if grep -a -q '__OFFSETCYCLE_WRONG_REGION__'     "$LOG"; then echo "__CAPSTONE_OFFSETCYCLE_VOID_wrong_region__ (see $LOG)"; exit 5; fi
grep -a -q 'OFFSETCYCLE size=' "$LOG" || { echo "__OFFSETCYCLE_DID_NOT_RUN__ (rc=$rc; see $LOG)"; exit 2; }
echo "__CAPSTONE_OFFSETCYCLE_FAILED__ (rc=$rc; see $LOG)"; exit 1
