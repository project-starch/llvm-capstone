#!/usr/bin/env bash
# Wait for a sentinel string in a file, OR for the owning process to die -- whichever first.
#
# WHY. Waiting on a sentinel alone is a hang waiting to happen: if the producing script dies
# (crash, syntax error, kill), the sentinel never appears and the waiter blocks forever while
# the board sits idle. On 2026-08-02 that wasted ~9 minutes of board time -- a script was
# edited WHILE RUNNING (bash reads scripts incrementally, so the edit corrupted the tail it
# had not yet read), it died before printing its sentinel, and the waiter kept waiting.
#
# Two hard rules this encodes:
#   1. Never wait on a sentinel without also watching the producer's PID.
#   2. Never edit a script that is currently executing. Copy it, edit the copy.
#
# Usage: wait-for.sh <file> <sentinel> <pid> [timeout-s]
# Exit:  0 sentinel seen | 3 producer died without sentinel | 4 timeout
set -uo pipefail
FILE=${1:?file}; SENTINEL=${2:?sentinel}; PID=${3:?pid}; TIMEOUT=${4:-7200}
start=$SECONDS
while true; do
  if [ -f "$FILE" ] && grep -q -- "$SENTINEL" "$FILE" 2>/dev/null; then
    echo "SENTINEL '$SENTINEL' seen after $((SECONDS-start))s"; exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    sleep 2   # let any final buffered write land
    if [ -f "$FILE" ] && grep -q -- "$SENTINEL" "$FILE" 2>/dev/null; then
      echo "SENTINEL seen at exit after $((SECONDS-start))s"; exit 0
    fi
    echo "PRODUCER PID $PID DIED after $((SECONDS-start))s WITHOUT printing '$SENTINEL'"
    echo "  -> the job is over; do NOT keep waiting. Last lines of $FILE:"
    tail -5 "$FILE" 2>/dev/null | sed 's/^/     /'
    exit 3
  fi
  if [ $((SECONDS-start)) -ge "$TIMEOUT" ]; then
    echo "TIMEOUT after $((SECONDS-start))s waiting for '$SENTINEL'"; exit 4
  fi
  sleep 10
done
