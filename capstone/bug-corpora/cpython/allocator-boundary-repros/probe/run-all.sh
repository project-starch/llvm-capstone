#!/bin/sh
# Runs inside the domain. One verdict line per case, read from /mnt/host/cases.
#
# Each case gets a watchdog of its own: this guest has no timeout(1), and some
# CPython reproducers loop. A case that outlives the budget is TIMEOUT, which
# is not a detection and not a silence -- it is its own row.
BIN=$1
BUDGET=${2:-120}
echo "ARMS-BEGIN watchdog=$BUDGET"
while read n; do
  cd /mnt/host/cases/$n 2>/dev/null || { echo "CASE $n NOCASE"; continue; }
  PYTHONDONTWRITEBYTECODE=1 PYTHONHOME=/mnt/host "$BIN" trigger.py > /tmp/case.out 2>&1 &
  pid=$!
  ( sleep "$BUDGET"; kill -9 $pid 2>/dev/null ) >/dev/null 2>&1 &
  dog=$!
  wait $pid
  st=$?
  kill -9 $dog 2>/dev/null
  # The marker is the trigger's own last line, so a verdict can never be read
  # off the exit status alone -- a batch-level status standing in for a case's
  # own result has produced wrong rows in this project before.
  mark=$(tail -3 /tmp/case.out | tr -d '\r' | tr '\n' ' ')
  echo "CASE $n rc=$st LAST=$mark"
done
echo "ARMS-DONE"
