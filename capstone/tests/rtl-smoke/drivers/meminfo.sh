#!/bin/sh
# /proc/meminfo on the board, read between arms (board-b80s.sh MEMINFO=1): the board's memory budget for the virtual
# profile's data window (docs/plans/2026-10-16-virtual-profile-on-silicon.md). Takes no part in any arm's verdict.
# The last line is what lets the stage driver accept the arm: a host-binary arm that prints no `RESULT ... retval=`
# must print `BASELINE-PROBE <what> = <digits>` (run_sqlite_stages_fpga.py), or the boot HARD STOPs at it, as
# p1s16-c5 did (2026-10-16 22:21) with the first version of this file.
echo "MEMINFO BEGIN $*"
cat /proc/meminfo
echo "MEMINFO END $*"
while read -r k v u; do [ "$k" = "MemFree:" ] && echo "BASELINE-PROBE memfree_kb = $v"; done < /proc/meminfo
