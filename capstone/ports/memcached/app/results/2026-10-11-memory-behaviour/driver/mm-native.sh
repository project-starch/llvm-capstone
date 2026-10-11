#!/usr/bin/env bash
# mm-native.sh memcached|mruby: the native reference of the 2026-10-11 memory-behaviour run, three
# repetitions one after the other on the build host, the same workload and readings as the guest
# gate (mm-run.py) without the Capstone-only ones. Output under ~/sa-mem-mm/native/<program>/repN.
set -u
. $HOME/sa-mem-mm/env.sh
prog=${1:?program}
N=$S/native/$prog; rm -rf $N; mkdir -p $N
PORT=23917
procst() { grep -E '^(State|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|VmData)' /proc/$1/status 2>/dev/null | tr -s ' \t' ' ' | tr '\n' ';'; }
if [ $prog = memcached ]; then
  gcc -O1 -Wall $R/ports/memcached/app/host/mc-harness/mc-harness.c -lpthread -o $N/mc-harness
  gcc -O1 -Wall $S/mc-stats.c -o $N/mc-stats
  for rep in 1 2 3; do
    O=$N/rep$rep; mkdir -p $O/h
    $S/mc-build/native/bin/memcached -l 127.0.0.1 -p $PORT -U 0 -m 64 -t 4 > $O/server.out 2> $O/server.err &
    pid=$!
    $N/mc-harness --out $O/h --port $PORT --stop TERM -- sleep 100000; echo "harness=$?" > $O/markers.txt
    echo "after_workload $(procst $pid)" > $O/proc.txt
    $N/mc-stats $PORT > $O/stats.txt; echo "mcstats=$?" >> $O/markers.txt
    echo "after_stats $(procst $pid)" >> $O/proc.txt
    kill -TERM $pid; wait $pid; echo "server_exit=$?" >> $O/markers.txt
  done
else
  M=$S/mruby/stock/src/mruby
  for rep in 1 2 3; do
    O=$N/rep$rep; mkdir -p $O/gc; cp $R/ports/mruby/app/scripts/smoke.rb $O/gc/
    for label in ao smoke; do
      if [ $label = ao ]; then set -- $M/benchmark/bm_ao_render.rb; else set -- $O/gc/smoke.rb $O/gc; fi
      $M/build/native/bin/mruby $S/mstats.rb "$@" > $O/$label.out 2> $O/$label.err &
      pid=$!; last=""; done_read=""
      while :; do
        s=$(procst $pid)
        case "$s" in ""|*"State: Z"*) break ;; esac
        last=$s
        if [ -z "$done_read" ] && grep -q MM_DONE $O/$label.err 2>/dev/null; then done_read=$s; fi
        sleep 0.5
      done
      wait $pid; echo "$label rc=$?" >> $O/markers.txt
      echo "$label:last $last" >> $O/proc.txt; echo "$label:done $done_read" >> $O/proc.txt
    done
  done
fi
echo "NATIVE $prog done" > $S/native-$prog.status
