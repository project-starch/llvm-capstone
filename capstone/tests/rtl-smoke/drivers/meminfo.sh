#!/bin/sh
# /proc/meminfo on the board, read between arms (board-b80s.sh MEMINFO=1): the board's memory budget for the virtual
# profile's data window (docs/plans/2026-10-16-virtual-profile-on-silicon.md). Takes no part in any arm's verdict.
echo "MEMINFO BEGIN $*"
cat /proc/meminfo
echo "MEMINFO END $*"
