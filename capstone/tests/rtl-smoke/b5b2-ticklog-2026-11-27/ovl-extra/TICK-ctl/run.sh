#!/bin/sh
# tick-log controls (runtime lane, b5b2-ticklog-2026-11-27): an idle window, then a constant-CPI busy loop.
up() { cut -d' ' -f1 /proc/uptime; }
echo "@@TICK-CTL idle-begin up=$(up)"
sleep 10
echo "@@TICK-CTL idle-end up=$(up)"
awk 'BEGIN { for (;;) ; }' & p=$!
echo "@@TICK-CTL loop-begin up=$(up)"
sleep 60
echo "@@TICK-CTL loop-end up=$(up)"
kill -9 $p; wait $p 2>/dev/null
echo "@@TICK-CTL done"
