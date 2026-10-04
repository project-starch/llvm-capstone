#!/bin/sh
# B0.7 (docs/plans/b0-silicon-delegated-runtime.md): the board-side wrapper, used as the baked-rung driver's
# BAKED_CTL. Usage: b0run.sh <rung> <domain>. Prints "RESULT <rung> retval=<n>", which the driver scores against
# its oracle file.
#   k800     the CLASSIC control: the stock module and the ladder host, exactly as every board boot runs it.
#   b0-hello the process-ABI module replaces the stock one (rmmod/insmod -- insmod works on this board: the stages
#            driver loads /capstone.ko the same way every boot), then capstone-exec launches the gp-captable image.
# Distinct retvals say where a failure happened: 0 the hello line was printed; 901 the module swap failed;
# 902 capstone-exec --stats failed (the module's process ABI is not answered by the monitor); 903 no hello line
# (the launch itself; its stderr is printed above); 904 the line printed but the exit status was non-zero.
rung=$1 dom=$2
case "$rung" in
  k800|k800r)
    # k800r is the k800 RELINKED at 0x20000 (589ceee3): b0-hello.dom enters at 0x10000, and a second domain at a
    # reused entry VA can hang (R-3, preflight C15) -- the control is relinked, never the image under test.
    [ -c /dev/capstone ] || insmod /capstone.ko 2>/dev/null
    exec /test-domains/lpc "$rung" "$dom"
    ;;
  b0-hello)
    if [ -c /dev/capstone ]; then rmmod capstone || { echo "RESULT $rung retval=901"; exit 1; }; fi
    insmod /test-domains/capstone-proc.ko || { echo "RESULT $rung retval=901"; exit 1; }
    [ -c /dev/capstone ] || { echo "RESULT $rung retval=901"; exit 1; }
    echo "B0: module swapped"
    /usr/bin/capstone-exec --stats || { echo "RESULT $rung retval=902"; exit 1; }
    /usr/bin/capstone-exec "$dom" > /tmp/b0.out 2> /tmp/b0.err
    rc=$?
    cat /tmp/b0.out; cat /tmp/b0.err
    echo "B0: capstone-exec rc=$rc"
    if grep -q "B0: hello from a gp-captable delegated application" /tmp/b0.out; then
      if [ $rc -eq 0 ]; then echo "RESULT $rung retval=0"; else echo "RESULT $rung retval=904"; fi
    else
      echo "RESULT $rung retval=903"
    fi
    ;;
  *) echo "RESULT $rung retval=999" ;;
esac
