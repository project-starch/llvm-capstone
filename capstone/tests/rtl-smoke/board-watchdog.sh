#!/usr/bin/env bash
# Board-session watchdog. Run it ALONGSIDE a board runner, as a second process.
#
# WHY THIS EXISTS. A board runner can sit for its whole per-domain budget with the FPGA doing
# nothing -- a wedged domain emits no UART at all, and a runner that died leaves a waiter
# spinning on a file that will never be written. Both look identical from outside: "still
# running". On 2026-08-02 that cost a stretch of wall-clock where nothing was on the board at
# all, while stale `tail -f` processes on finished logs made it look busy.
#
# The runner's own timeouts are not enough: they are per-domain and silent until they expire,
# so a 480 s budget spends 480 s before saying anything. This reports every interval.
#
# LIVENESS IS BY PID, NEVER BY PROCESS NAME. `pgrep -f <pattern>` matches the watchdog's OWN
# command line, because the pattern is one of its arguments -- so it reports the runner alive
# forever and the watchdog never fires. That is the self-matching-pgrep trap this project has
# hit three times; the first draft of THIS script hit it too and a unit test caught it.
#
# LIVE ENTRY-STALL (R-16) DETECTION. `SHA5` means the monitor handed off to the domain; `SHA6`
# means the domain came back. `SHA5` with no following `SHA6` and no further UART is an entry
# stall: the domain's code never ran, so THIS run carries no information and every later domain
# in the same boot is lost anyway (a wedged domain takes the core). Waiting out the per-domain
# timeouts after that is pure waste -- 3 domains x 200 s = 600 s spent to learn nothing.
# With ABORT_ON_ENTRY_STALL=1 the watchdog kills the runner as soon as the pattern is stable,
# turning a 600 s dead run into ~30 s.
#
# Usage:
#   bash board-watchdog.sh <uart-log> [idle-limit-s] [runner-pid]
#   ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=45 bash board-watchdog.sh ...
#   (runner-pid omitted -> liveness is not checked, only UART idle is reported)
#
# Emits ONE LINE PER CHECK on stdout, so it can be piped into a Monitor:
#   ALIVE   <elapsed>s  +<bytes> since last check
#   QUIET   <elapsed>s  no UART for <idle>s  (limit <limit>s)
#   STALE   <elapsed>s  no UART for <idle>s  -- EXCEEDED LIMIT
#   GONE    runner process no longer exists (log last grew <idle>s ago)
#   ENDED   runner finished
set -uo pipefail

LOG=${1:?usage: board-watchdog.sh <uart-log> [idle-limit-s] [runner-pattern]}
LIMIT=${2:-180}
RUNNER_PID=${3:-}
INTERVAL=${WATCHDOG_INTERVAL:-15}
ABORT_ON_ENTRY_STALL=${ABORT_ON_ENTRY_STALL:-0}
ENTRY_STALL_S=${ENTRY_STALL_S:-45}

start=$SECONDS
last_size=-1
last_change=$SECONDS

while true; do
  sleep "$INTERVAL"
  now_elapsed=$(( SECONDS - start ))
  size=$( [ -f "$LOG" ] && stat -c%s "$LOG" 2>/dev/null || echo 0 )

  # kill -0 tests existence of a specific process. No name matching, so nothing can match
  # this script itself. With no PID given, assume alive and rely on the idle limit alone.
  runner_alive=1
  if [ -n "$RUNNER_PID" ]; then
    kill -0 "$RUNNER_PID" 2>/dev/null || runner_alive=0
  fi

  if [ "$size" -ne "$last_size" ]; then
    delta=$(( size - (last_size < 0 ? size : last_size) ))
    last_size=$size
    last_change=$SECONDS
    echo "ALIVE   ${now_elapsed}s  +${delta}B"
  else
    idle=$(( SECONDS - last_change ))
    # Entry stall: the LAST capability-share marker in the log is SHA5 with no SHA6 after it.
    if [ "$ABORT_ON_ENTRY_STALL" = "1" ] && [ "$idle" -ge "$ENTRY_STALL_S" ] && [ -f "$LOG" ]; then
      lastmark=$(grep -ao 'SHA[56]:[0-9A-F]*' "$LOG" 2>/dev/null | tail -1)
      case "$lastmark" in
        SHA5:*)
          echo "ENTRY-STALL ${now_elapsed}s  last share marker=$lastmark, no SHA6 for ${idle}s"
          echo "  -> domain never ran; the rest of this boot is worthless. Aborting runner."
          [ -n "$RUNNER_PID" ] && kill -TERM "$RUNNER_PID" 2>/dev/null
          echo "ENDED   ${now_elapsed}s"
          exit 0 ;;
      esac
    fi
    if [ "$runner_alive" -eq 0 ]; then
      echo "GONE    ${now_elapsed}s  runner not running; log last grew ${idle}s ago"
      echo "ENDED   ${now_elapsed}s"
      exit 0
    fi
    if [ "$idle" -ge "$LIMIT" ]; then
      echo "STALE   ${now_elapsed}s  no UART for ${idle}s -- EXCEEDED LIMIT ${LIMIT}s"
    else
      echo "QUIET   ${now_elapsed}s  no UART for ${idle}s (limit ${LIMIT}s)"
    fi
  fi

  if [ "$runner_alive" -eq 0 ]; then
    echo "GONE    ${now_elapsed}s  runner PID ${RUNNER_PID:-?} is no longer running"
    echo "ENDED   ${now_elapsed}s"
    exit 0
  fi
done
