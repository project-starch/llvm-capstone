#!/usr/bin/env bash
# Create a capability region of a given size in one QEMU boot, optionally with a CMA area.
#
#   bash run-bigregion.sh <bytes> [cma=<size>]
#   bash run-bigregion.sh 4194304              # 4 MiB, no CMA -- must PASS (the old ceiling)
#   bash run-bigregion.sh 8388608              # 8 MiB, no CMA -- must FAIL AT CREATE
#   bash run-bigregion.sh 8388608 cma=256M     # 8 MiB with CMA -- must PASS
#
# THE THREE ARMS ARE THE TEST; no single one of them is. A pass at 8 MiB with CMA means nothing
# unless the same size failed without it, because a passing arm is equally consistent with the size
# argument never reaching the allocator.
#
# The `cma=` flag needs no runner option: run-domain-smoke.py appends --qemu-extra-arg verbatim at
# the END of the qemu command line and qemu takes the LAST -append, so a second -append overrides
# the built-in one. (argparse rejects a bare `-append` as a value, hence the `=` form.)
#
# Take the QEMU lock yourself or set CAPSTONE_QEMU_LOCK_HELD=1. Reads CAPSTONE_BUILDROOT_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
BYTES=${1:?usage: run-bigregion.sh <bytes> [cma=<size>]}
CMA=${2:-}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/bigregion-share}
LOG=${LOG_FILE:-$TMP_ROOT/bigregion-${BYTES}${CMA:+-$CMA}.log}
mkdir -p "$SHARE_DIR"
bash "$SCRIPT_DIR/build-bigregion.sh" "$SHARE_DIR" >/dev/null

EXTRA=()
if [[ -n "$CMA" ]]; then EXTRA=(--qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro $CMA"); fi

set +e
python3 "$SCRIPT_DIR/run-domain-smoke.py" --share-dir "$SHARE_DIR" --log-file "$LOG" \
  --guest-command "/mnt/host/bigregion.user $BYTES; grep -i cma /proc/meminfo; dmesg | grep -i 'cma:'" \
  --success-marker "__BIGREGION_PASSED__" "${EXTRA[@]}" >/dev/null 2>&1; rc=$?
set -e

grep -a -E '^(BIGREGION|Cma|\[.*\] cma:)' "$LOG" | tr -d '\r' | head -20
if grep -a -q '__BIGREGION_PASSED__' "$LOG"; then echo "__CAPSTONE_BIGREGION_PASSED__"; exit 0; fi
# A CREATE failure and a MAP failure are different findings and must not be collapsed.
if grep -a -q '__BIGREGION_CREATE_FAILED__' "$LOG"; then echo "__CAPSTONE_BIGREGION_CREATE_FAILED__ (see $LOG)"; exit 3; fi
if grep -a -q '__BIGREGION_MAP_FAILED__'    "$LOG"; then echo "__CAPSTONE_BIGREGION_MAP_FAILED__ (see $LOG)"; exit 4; fi
grep -a -q 'BIGREGION request' "$LOG" || { echo "__BIGREGION_DID_NOT_RUN__ (rc=$rc; see $LOG)"; exit 2; }
echo "__CAPSTONE_BIGREGION_FAILED__ (rc=$rc; see $LOG)"; exit 1
