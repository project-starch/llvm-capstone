#!/bin/sh
# B0.7 (docs/plans/b0-silicon-delegated-runtime.md): the board-side wrapper, used as the baked-rung driver's
# BAKED_CTL. Usage: b0run.sh <rung> <domain>. Prints "RESULT <rung> retval=<n>", which the driver scores against
# its oracle file.
#   k800r    the CLASSIC control (lpc + the relinked k800). It CANNOT share a boot with the process ABI: the module
#            serves one API per load (the first legacy ioctl sets legacy_api_selected, after which PROCESS_ENABLE is
#            EBUSY), the board cannot unload it, and lpc's DOM_CREATE struct predates the module's copy_len field, so
#            its ioctl number is unrecognised (B0.7 attempt 4, 2026-10-04).
#   b0-stats capstone-exec --stats alone: the monitor answers the process ABI's census (b0-stats2: the same, after
#            b0-hello -- it only runs if b0-hello returned).
#   b0-hello capstone-exec launches the gp-captable image under the process-ABI module.
# Distinct retvals say where a failure happened: 0 the hello line was printed; 901 the process-ABI module is not loaded;
# 902 capstone-exec --stats failed (the module's process ABI is not answered by the monitor); 903 no hello line
# (the launch itself; its stderr is printed above); 904 the line printed but the exit status was non-zero; 905 the
# watchdog ended a capstone-exec that was still running at 90 s, with Linux alive; 906 the line printed but the
# stream is not byte-exact.
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
  b0-stats|b0-stats2)
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    /usr/bin/capstone-exec --stats; rc=$?
    echo "B0: capstone-exec --stats rc=$rc"
    if [ $rc -eq 0 ]; then echo "RESULT $rung retval=0"; else echo "RESULT $rung retval=902"; fi
    ;;
  b0-hello)
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    echo "B0: process-ABI module loaded"
    /usr/bin/capstone-exec --stats || { echo "RESULT $rung retval=902"; exit 1; }
    # Attempt 6 (2026-10-04): attempt 5 went silent after the first STEP returned (SUPK 0), and a silent console
    # cannot tell a wedged core from a stuck capstone-exec. So: a bounded heartbeat (stops only if the core or the
    # kernel stops), capstone-exec's stderr live on the console with its diagnostics and counters, and a 90 s
    # watchdog that ends a capstone-exec Linux is still running (retval 905: Linux alive, capstone-exec did not end).
    rm -f /tmp/b0.wd
    ( i=0; while [ $i -lt 40 ]; do sleep 5; i=$((i+1)); echo "B0: hb $i"; done ) &
    hb=$!
    CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_DELEGATE_STATS=1 CAPSTONE_DELEGATE_TRACE=200 /usr/bin/capstone-exec "$dom" > /tmp/b0.out &
    cx=$!
    ( sleep 90
      if kill -0 $cx 2>/dev/null; then
        echo watchdog > /tmp/b0.wd
        echo "B0: watchdog: capstone-exec still running at 90 s; SIGTERM"; kill -TERM $cx; sleep 5; kill -KILL $cx 2>/dev/null
      fi ) &
    wd=$!
    wait $cx
    rc=$?
    kill $hb $wd 2>/dev/null
    cat /tmp/b0.out
    echo "B0: capstone-exec rc=$rc"
    # BYTE-EXACT, not a grep: attempt 11 scored 0 on a grep while the stream carried 65 extra bytes (R-29 in the
    # runtime's wire copies). 906: the line is there but the stream is not exactly it.
    printf 'B0: hello from a gp-captable delegated application\n' > /tmp/b0.want
    if [ -e /tmp/b0.wd ]; then
      echo "RESULT $rung retval=905"
    elif cmp -s /tmp/b0.out /tmp/b0.want; then
      if [ $rc -eq 0 ]; then echo "RESULT $rung retval=0"; else echo "RESULT $rung retval=904"; fi
    elif grep -q "B0: hello from a gp-captable delegated application" /tmp/b0.out; then
      echo "B0: stdout is $(wc -c < /tmp/b0.out) bytes, expected $(wc -c < /tmp/b0.want)"
      echo "RESULT $rung retval=906"
    else
      echo "RESULT $rung retval=903"
    fi
    ;;
  b0-memcpy|b0-printf|b1-thread|b0-strtod)
    # The application's own exit status is the result; capstone-exec's own failures keep their codes (125 and so on).
    # B0.8 b0-memcpy, the R-29 memcpy guard on silicon: 0 the unguarded control miscopied and the guarded copy never
    # did; 1 the guarded copy miscopied; 2 the control never miscopied (void: no hazard created).
    # B1.0 b0-printf, the narrowed vfprintf: 0 all 20 snprintf cases equal the host printf's strings; otherwise the
    # number of cases that differ (each printed).
    # B1 b1-thread, one minted context: 0 joined with the expected value; 3 pthread_create failed; 4 pthread_join
    # failed; 5 joined with a wrong value.
    # B1.0b b0-strtod, the narrowed float parser: 0 all 23 cases equal the host's; otherwise the number that differ.
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    /usr/bin/capstone-exec "$dom" > /tmp/b0.out 2> /tmp/b0.err
    rc=$?
    cat /tmp/b0.out; cat /tmp/b0.err
    echo "RESULT $rung retval=$rc"
    ;;
  b2-memcached)
    # B2: memcached as a gp-captable delegated application. The domain serves 127.0.0.1:21299 in the background;
    # the native client (/test-domains/mc-b2-client) runs version/set/get and exits 0 only on the exact replies;
    # the domain is then stopped with SIGTERM. retval = 10 * client code + (memcached's status != 0), so 0 is the
    # milestone, 1 means the replies were right but the shutdown status was not 0, and 20+ names the client's step.
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    # Unprivileged, as the SDK oracle runs it: started as root, memcached insists on -u and then drops supplementary
    # groups with setgroups, which the delegate runtime does not serve (ENOSYS, exit 71). capstone-job forwards
    # SIGTERM to the launcher.
    /usr/bin/capstone-job /tmp/b2-job.json --user 65534:65534 -- /usr/bin/capstone-exec "$dom" \
      -l 127.0.0.1 -p 21299 -U 0 -m 8 -t 1 \
      -o no_lru_crawler,no_lru_maintainer,no_slab_reassign,no_hashexpand > /tmp/b2.out 2> /tmp/b2.err &
    pid=$!
    /test-domains/mc-b2-client
    crc=$?
    kill -TERM "$pid" 2>/dev/null
    wait "$pid"
    mrc=$?
    echo "B2: client rc=$crc memcached rc=$mrc job $(cat /tmp/b2-job.json 2>/dev/null)"
    cat /tmp/b2.out; cat /tmp/b2.err
    echo "RESULT $rung retval=$((crc * 10 + (mrc != 0)))"
    ;;
  b3-oracle)
    # B3: memcached's oracle on silicon. The port's harness (ports/memcached/app/host/mc-harness) starts the domain
    # under capstone-job, runs its scripted 8-connection session and stops it with SIGTERM. Only the transcript's
    # hash and length are printed (it is ~1.9 MB); the verdict is that hash against the native reference's, taken
    # with the same flags and the same harness (docs/plans/b0-silicon-delegated-runtime.md, B3).
    # retval = harness status (0 = it ran to the end).
    load_proc_module || { echo "RESULT $rung retval=901"; exit 1; }
    rm -rf /tmp/mc; mkdir -p /tmp/mc
    /test-domains/mc-harness --out /tmp/mc --port 21299 --stop TERM -- \
      /usr/bin/capstone-job /tmp/mc/job.json --user 65534:65534 -- /usr/bin/capstone-exec "$dom" \
      -l 127.0.0.1 -p 21299 -U 0 -m 8 -t 1 -o no_lru_crawler,no_lru_maintainer,no_slab_reassign,no_hashexpand \
      > /tmp/b3.log 2>&1
    hrc=$?
    tail -5 /tmp/b3.log
    echo "B3: transcript $(sha256sum /tmp/mc/transcript.norm 2>/dev/null | cut -c1-16) bytes $(wc -c < /tmp/mc/transcript.norm 2>/dev/null)"
    echo "B3: identity $(cat /tmp/mc/identity.txt 2>/dev/null) job $(cat /tmp/mc/job.json 2>/dev/null) status $(cat /tmp/mc/status.txt 2>/dev/null)"
    # Diagnostic: with the native reference in the image, the first differing lines (the transcript stays here).
    if [ -f /test-domains/b3-native.norm ]; then
      echo "B3: diff native board (first 40 lines)"
      diff /test-domains/b3-native.norm /tmp/mc/transcript.norm | head -40 | sed 's/^/B3d /'
    fi
    echo "RESULT $rung retval=$hrc"
    ;;
  *) echo "RESULT $rung retval=999" ;;
esac
