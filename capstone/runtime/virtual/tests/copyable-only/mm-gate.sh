#!/bin/sh
set -eu
trap 'echo COPYABLE_MM_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cd /mnt/vm
swap=$(awk '$3 == "arch_prepare_to_swap" { print "0x" $1 }' /proc/kallsyms)
test -n "$swap"
insmod mm-probe.ko swap_prepare_address="$swap"
dmesg | grep RAW_MM
rmmod mm-probe
echo COPYABLE_MM_COMPLETE
