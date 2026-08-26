#!/usr/bin/env bash
# Wait for a run to finish, and ALWAYS terminate.
#
#   usage: bash wait-run.sh <logfile> [max_seconds] [stall_seconds]
#
# WHY THIS EXISTS. Five `until grep -qa "<-- TEST 3/3" ...; do sleep; done` loops span 7-11
# minutes each and never exited. Two distinct bugs, both worth naming:
#
#  1. THE TERMINAL PATTERN ASSUMED THE RUN REACHES ITS LAST ARM. The driver stops at the first
#     wedge or hard stop -- that is CORRECT behaviour, "everything after this is lost" -- so
#     `<-- TEST 3/3` never appears when arm 1 wedges, which is exactly the interesting case.
#     A waiter keyed to the success path hangs precisely when the run is most informative.
#  2. IT GREPPED FOR A STRING THAT IS NEVER WRITTEN TO THAT FILE. `EXIT=$?` is echoed to the
#     task's stdout, not into the driver log being polled. The pattern could never match.
#
# The fix is to key on markers the driver ALWAYS emits on every exit path, and to bound the
# wait two independent ways so no future pattern mistake can hang again:
#   - BOARD_RELEASED       the board driver's release, printed in its finally
#   - preflight BLOCKED    refused before spending a boot
#   - Traceback            crashed
#   - GUEST_RC / SLT_RAN__ the QEMU paths' own terminators
#   - STALL: no growth in the log for `stall_seconds` -> stop and say so. Per the standing
#     rule, watch LOG GROWTH rather than process exit: a hung driver never exits.
#   - MAX:   hard cap, unconditional.
#
# Exit status: 0 finished, 2 stalled, 3 hit the cap. Never hangs.
set -uo pipefail
LOG=${1:?usage: wait-run.sh <logfile> [max_s] [stall_s]}
MAX=${2:-2400}
STALL=${3:-420}
T0=$SECONDS
LAST_SIZE=-1
LAST_CHANGE=$SECONDS
while :; do
  if [[ -f "$LOG" ]] && grep -qaE 'BOARD_RELEASED|preflight BLOCKED|Traceback|GUEST_RC|__CAPSTONE_SQLITE_SLT_RAN__' "$LOG"; then
    echo "wait-run: finished after $((SECONDS-T0))s"; exit 0
  fi
  SZ=$( [[ -f "$LOG" ]] && wc -c < "$LOG" || echo 0 )
  if [[ "$SZ" != "$LAST_SIZE" ]]; then LAST_SIZE=$SZ; LAST_CHANGE=$SECONDS; fi
  if (( SECONDS - LAST_CHANGE > STALL )); then
    echo "wait-run: STALLED -- $LOG has not grown in ${STALL}s (size $SZ) after $((SECONDS-T0))s"; exit 2
  fi
  if (( SECONDS - T0 > MAX )); then
    echo "wait-run: CAP -- gave up after ${MAX}s (size $SZ)"; exit 3
  fi
  sleep 15
done
