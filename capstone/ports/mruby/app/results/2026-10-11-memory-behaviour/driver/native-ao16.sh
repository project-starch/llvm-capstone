#!/usr/bin/env bash
# The native reference for the 16 x 16 AO render (run inline on the build host, 2026-10-11; kept
# here verbatim as a script): mstats.rb with the native interpreter, three runs, the launcher-less
# equivalent of mm-run2.py's sampler. Output under ~/sa-mem-mm/native/mruby-w16/repN.
. ~/sa-mem-mm/env.sh; M=$S/mruby/stock/src/mruby; N=$S/native/mruby-w16; rm -rf $N; mkdir -p $N
for rep in 1 2 3; do
  O=$N/rep$rep; mkdir -p $O
  $M/build/native/bin/mruby $S/mstats.rb $M/benchmark/bm_ao_render.rb 16 > $O/ao.out 2> $O/ao.err & pid=$!
  last=""; done_read=""
  while :; do
    s=$(grep -E "^(State|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|VmData)" /proc/$pid/status 2>/dev/null | tr -s " \t" " " | tr "\n" ";")
    case "$s" in ""|*"State: Z"*) break;; esac
    last=$s; if [ -z "$done_read" ] && grep -q MM_DONE $O/ao.err; then done_read=$s; fi; sleep 0.5
  done
  wait $pid; echo "ao rc=$?" > $O/markers.txt; echo "ao:last $last" > $O/proc.txt; echo "ao:done $done_read" >> $O/proc.txt
done
