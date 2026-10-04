#!/bin/sh
# B0.7 (docs/plans/b0-silicon-delegated-runtime.md): the board-side wrapper, used as the baked-rung driver's
# BAKED_CTL. Usage: b0run.sh <rung> <domain>. Prints "RESULT <rung> retval=<n>", which the driver scores against
# its oracle file.
#   k800r    the CLASSIC control (lpc + the relinked k800) under the SAME process-ABI module the image uses.
#   b0-hello capstone-exec launches the gp-captable image under the process-ABI module.
# Distinct retvals say where a failure happened: 0 the hello line was printed; 901 the process-ABI module is not loaded;
# 902 capstone-exec --stats failed (the module's process ABI is not answered by the monitor); 903 no hello line
# (the launch itself; its stderr is printed above); 904 the line printed but the exit status was non-zero.
rung=$1 dom=$2
# The board's kernel has NO module unload (vermagic "6.4.14 SMP riscv", no mod_unload: rmmod answers "Function not
# implemented" -- B0.7's first boot, 2026-10-04 21:22). So the process-ABI module is the ONLY one loaded, first, and
# it serves the classic control too: a failing control then says "the module", a failing b0-hello "the process ABI".
load_proc_module() {
  if [ -c /dev/capstone ]; then
    [ -e /sys/module/capstone/parameters/process_cache_bytes ] && return 0
    echo "B0: a module without the process ABI is already loaded and cannot be unloaded"; return 1
  fi
  insmod /test-domains/capstone-proc.ko && [ -c /dev/capstone ]
}
case "$rung" in
  k800|k800r)
    # k800r is the k800 RELINKED at 0x20000 (589ceee3): b0-hello.dom enters at 0x10000, and a second domain at a
    # reused entry VA can hang (R-3, preflight C15) -- the control is relinked, never the image under test.
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    exec /test-domains/lpc "$rung" "$dom"
    ;;
  b0-hello)
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    echo "B0: process-ABI module loaded"
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
