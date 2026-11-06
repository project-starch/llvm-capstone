#!/bin/bash
# b2-seq.sh: board 2's queue after gcc train (perf/ga2), as one task so no step waits on another task's pid:
#  1. wait for the ga2 driver (SEQ_WAITPID) to exit with DONE in its log;
#  2. the R-32 SHRINKTO readback bare boot on C.5a (perf/r32/r32-c5a.sh);
#  3. the THP mcf runs (perf/thp/gt-run.sh, pre-registration 122cd9ffd4e1 + addendum), unchanged;
#  4. perlbench test in the best configuration (perf/best/gt-run.sh, list perf/best/b2.list).
# Steps 3 and 4 run whatever the step before returned, unless SEQ_STOP_ON_FAIL=1; each driver does its own checks.
# Stubs: SEQ_R32 SEQ_THP SEQ_BEST (commands), and the GT_* / R32_* stubs pass through to the drivers.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}; local O=${SEQ_O:-$H/perf/best} L rc t=0
  L=$O/b2-seq.log; log() { echo "SEQ $(date '+%F %T') $*" >> "$L"; }; log "START pid $$"
  [ -n "${SEQ_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$SEQ_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${SEQ_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  while [ -n "${SEQ_WAITPID:-}" ] && kill -0 "$SEQ_WAITPID" 2>/dev/null; do
    [ $t -ge 172800 ] && { log "STOP: pid $SEQ_WAITPID still runs"; return 2; }; sleep "${SEQ_POLL_S:-60}"; t=$((t + ${SEQ_POLL_S:-60})); done
  [ -z "${SEQ_WAITPID:-}" ] || grep -q "^GT .* DONE" "${SEQ_WAITLOG:-/dev/null}" \
    || { log "STOP: pid ${SEQ_WAITPID} did not end with DONE in ${SEQ_WAITLOG:-?}"; return 2; }
  log "R32 BEGIN"; ${SEQ_R32:-bash $H/perf/r32/r32-c5a.sh} 2 "${SEQ_R32_OUT:-$H/perf/r32/run-b2}"; rc=$?; log "R32 END rc=$rc"
  [ "${SEQ_STOP_ON_FAIL:-0}" = 1 ] && [ $rc = 3 ] && { log "STOP after R32"; return 2; }
  log "THP BEGIN"; env GT_PREREG=${SEQ_THP_PREREG:-122cd9ffd4e1} GT_OUT=${SEQ_THP_OUT:-$H/spec/board/out-thp-b2} GT_O=${SEQ_THP_O:-$H/perf/thp} \
    ${SEQ_THP:-bash $H/perf/thp/gt-run.sh} 2; rc=$?; log "THP END rc=$rc"
  log "BEST BEGIN"; env GT_PREREG=$SEQ_PREREG GT_OUT=${SEQ_BEST_OUT:-$H/spec/board/out-best-b2} GT_O=${SEQ_BEST_O:-$H/perf/best} \
    ${SEQ_BEST:-bash $H/perf/best/gt-run.sh} 2; rc=$?; log "BEST END rc=$rc"
  log "DONE"
}
main "$@"; exit $?
