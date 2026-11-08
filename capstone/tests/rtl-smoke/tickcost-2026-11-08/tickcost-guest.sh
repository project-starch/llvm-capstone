#!/bin/sh
# Runs on the board guest, as root, after boot. One boot: arms A-D are expected to return; the last arm (E1 or E2,
# one per boot) never returns by construction.
# Usage: tickcost-guest.sh /path/to/tickcost.ko E1|E2
M=${1:?path to tickcost.ko}
E=${2:?E1 or E2}
P=/sys/module/tickcost/parameters
insmod "$M" || { echo "@@TC INSMOD-FAIL"; exit 1; }
echo "@@TC A scanbench"
echo 1 > $P/scanbench
echo "@@TC A ALIVE"
echo "@@TC B spin 3000 us"
echo 0 > $P/mode; echo 3000 > $P/spin_us; echo 1 > $P/enable; sleep 30; echo 0 > $P/enable
echo "@@TC B ALIVE"
echo "@@TC C cvprof body over full tables"
echo 1 > $P/mode; echo 1 > $P/enable; sleep 30; echo 0 > $P/enable
echo "@@TC C ALIVE"
echo "@@TC D spin 15000 us"
echo 0 > $P/mode; echo 15000 > $P/spin_us; echo 1 > $P/enable; sleep 30; echo 0 > $P/enable
echo "@@TC D ALIVE"
case "$E" in
  E1) echo "@@TC E1 stuck in the tick, rdtime loop"; echo 2 > $P/mode ;;
  E2) echo "@@TC E2 stuck in the tick, register loop"; echo 3 > $P/mode ;;
  *) echo "@@TC BAD-ARM $E"; exit 1 ;;
esac
echo 1 > $P/enable; sleep 30
echo "@@TC $E ALIVE"
