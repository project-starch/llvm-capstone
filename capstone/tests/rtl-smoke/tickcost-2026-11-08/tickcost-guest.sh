#!/bin/sh
# Runs on the board guest, as root, after boot. One boot, arms in order; only arm D is expected not to return.
# Usage: tickcost-guest.sh /path/to/tickcost.ko
M=${1:?path to tickcost.ko}
P=/sys/module/tickcost/parameters
insmod "$M" || { echo "@@TC INSMOD-FAIL"; exit 1; }
echo "@@TC A scanbench"
echo 1 > $P/scanbench
echo "@@TC A ALIVE"
echo "@@TC B spin 3000 us"
echo 0 > $P/mode; echo 3000 > $P/spin_us; echo 1 > $P/enable; sleep 30; echo 0 > $P/enable
echo "@@TC B ALIVE"
dmesg -n 8
echo "@@TC C cvprof body over full tables"
echo 1 > $P/mode; echo 1 > $P/enable; sleep 30; echo 0 > $P/enable
echo "@@TC C ALIVE"
echo "@@TC D spin 15000 us"
echo 0 > $P/mode; echo 15000 > $P/spin_us; echo 1 > $P/enable; sleep 60; echo 0 > $P/enable
echo "@@TC D ALIVE"
dmesg -n 1
