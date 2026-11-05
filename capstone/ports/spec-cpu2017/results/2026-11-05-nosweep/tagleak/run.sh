#!/bin/sh
# board-env-check: optional /proc/self/pagemap (tagleak falls back to the counting proof without it)
# 999.tagleak: does a frame a context returned without an exit sweep reach the next context still tagged?
# Per sweep mode (3 = the all2 kit's store sweep, 2 = no exit sweep): fill <MB> (a capability in every granule,
# its PFNs to /tmp/fill.pfn), then scan <MB> (tags in a fresh block, how many of its frames the fill held, and a
# self-made tag as the positive control). The mode is restored afterwards.
S=$(cd "$(dirname "$0")" && pwd)
V=${SPEC_VEXEC:?set SPEC_VEXEC to the batch launcher}
# 60 % of MemFree: fill + scan exceed what is free, so at least fill + scan - free of the scan's frames were the fill's
MB=${TAGLEAK_MB:-$(( $(grep MemFree /proc/meminfo | tr -s " " | cut -d" " -f2) * 6 / 10 / 1024 ))}
PM=${TAGLEAK_PM:-}
P=/sys/module/capstone_vm/parameters/retire_exit_sweep
old=$(cat $P)
echo "@@SPEC-STAGE 999.tagleak test arms=capstone-lifetimes commands=0 validations=0"
cd /tmp || exit 1
for mode in 3 2; do
  echo $mode > $P
  echo "TAGLEAK-MODE sweep=$(cat $P) mb=$MB free_kb=$(grep MemFree /proc/meminfo | tr -s ' ' | cut -d' ' -f2)"
  rm -f /tmp/fill.pfn
  "$V" "$S/bin/tagleak" fill $MB $PM > /tmp/tl.out 2> /tmp/tl.err &
  pid=$!
  # ETXTBSY probe (image_rootfs): while the image runs, an open for writing must fail; it writes nothing either way
  sleep ${TAGLEAK_ETX_S:-8}
  if kill -0 $pid 2>/dev/null; then
    if ( : >> "$S/bin/tagleak" ) 2> /tmp/etx.err; then echo "ETXPROBE running=WRITABLE"; else echo "ETXPROBE running=DENIED $(cat /tmp/etx.err)"; fi
  else echo "ETXPROBE running=ENDED-EARLY"; fi
  wait $pid; frc=$?
  echo "TAGLEAK-RC fill $frc $(cat /tmp/tl.out) $(grep -a -v CAPSTONE_VM_STATS /tmp/tl.err | head -3)"
  # its control: the same open on the image that no longer runs succeeds
  if ( : >> "$S/bin/tagleak" ) 2> /tmp/etx.err; then echo "ETXPROBE idle=WRITABLE"; else echo "ETXPROBE idle=DENIED $(cat /tmp/etx.err)"; fi
  echo "TAGLEAK-FREE before-scan free_kb=$(grep MemFree /proc/meminfo | tr -s ' ' | cut -d' ' -f2)"; "$V" "$S/bin/tagleak" scan $MB $PM > /tmp/tl.out 2> /tmp/tl.err; echo "TAGLEAK-RC scan $? $(cat /tmp/tl.out) $(grep -a -v CAPSTONE_VM_STATS /tmp/tl.err | head -3)"
done
echo $old > $P
echo "TAGLEAK-RESTORED sweep=$(cat $P)"
echo "@@SPEC-DONE 999.tagleak test"
