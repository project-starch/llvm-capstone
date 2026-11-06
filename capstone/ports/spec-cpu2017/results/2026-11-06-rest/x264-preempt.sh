#!/bin/bash
# x264-preempt.sh: rest Addendum 3's swap on board 1. Waits (up to 6 h) for the x264 c3f QEMU gate (q-x264-c3f). If
# both commands end rc 0 and every output hash (except the small logs x264.err/.out, and except the 'control' hash which names the stage files) equals the -ac twin's (q-x264-ac),
# (logs log.dec and *.err are skipped: they differ between any two runs) it stops the running x264-ac board stage (run-stage, process group 2205025, checked by cmdline), so board 1's chain
# goes on to perf/rest/b1.list line 3 (x264 c3f). If the gate fails, it removes line 3 instead and x264-ac runs on.
main() {
  set -u
  local H=${CAPSTONE_WORK:?}; local B=$H/perf/best L=$H/perf/best/x264-preempt.log G=$H/perf/best/q-x264-c3f t=0 v
  log() { echo "XP $(date '+%F %T') $*" >> "$L"; }; log "START pid $$"
  until grep -qE '^READ|Traceback' $G.log 2>/dev/null || ! pgrep -f 'qgate-c3f.sh' > /dev/null; do
    [ $t -ge 21600 ] && { log "STOP: no gate result in 6 h"; return 2; }; sleep 30; t=$((t+30)); done
  sleep 60
  v=$(python3 - $G/result.json $H/perf/rest/q-x264-ac/result.json <<'PY'
import json, sys
try:
    a = json.load(open(sys.argv[1]))['arms']['capstone-lifetimes']; b = json.load(open(sys.argv[2]))['arms']['capstone-lifetimes']
except Exception as e:
    print(f"FAIL no result: {e}"); sys.exit()
rc = [c['rc'] for c in a['commands']]
skip = lambda k: k == 'control' or k == 'log.dec' or k.endswith('.err')
ha = {k: v for k, v in a['hashes'].items() if not skip(k)}; hb = {k: v for k, v in b['hashes'].items() if not skip(k)}
diff = sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k))
ok = len(rc) == 2 and all(x == 0 for x in rc) and not diff and len(ha) >= 3
print(("PASS" if ok else "FAIL") + f" rc={rc} hashes={len(ha)} differ={diff}")
PY
)
  log "GATE $v"
  if [ "${v%% *}" = PASS ]; then
    local cmd; cmd=$(tr '\0' ' ' < /proc/2205026/cmdline 2>/dev/null)
    case "$cmd" in *"--stage $H/perf/rest/stages/525.x264_r-test-capstone-lifetimes-devmk-ac "*)
      log "stopping x264-ac run-stage (pgid 2205025)"; kill -TERM -2205025; sleep 30; log "after: $(ps -p 2205026 > /dev/null && echo still || echo gone)";;
      *) log "x264-ac run-stage not found by cmdline (pid 2205026: '${cmd:0:80}'); nothing stopped";; esac
  else
    python3 - $H/perf/rest/b1.list <<'PY'
import sys
p = sys.argv[1]; L = open(p).read().split('\n')
assert 'devmk-c3f' in L[2]
with open(p, 'r+') as f: f.seek(0); f.write(L[0] + '\n' + L[1] + '\n'); f.truncate()
PY
    log "gate failed: line 3 removed, x264-ac runs on"
  fi
  log "DONE"
}
main "$@"; exit $?
