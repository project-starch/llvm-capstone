#!/usr/bin/env bash
# calib.sh (perf/plainhwm): the Plain arm's calibration runs on the twin, before any H16 run is pre-registered: ctl
# (the board image, byte-identical) and hl0 (the same libc with PLAIN_REUSE_TEST_LOG: the cache's request stream,
# policy unchanged) for mcf test, gcc train01, mcf train. hmcheck and replay --check ctl over each hl0 log. Output perf/plainhwm/out, RESULTS.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}"
main() {
  local R=${CAPSTONE_WORK}/perf/plainhwm S=${CAPSTONE_WORK}/perf/plainhwm/stages M=${CAPSTONE_WORK}/perf/mrhwm
  export TWIN_CMDLINE_MAX=1024 OUTROOT=$R
  export EXTRA=" capstone_vm.node_table_pool=4 capstone_vm.dom_grant_root=/tmp/nt/sr capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1"
  echo "$(date '+%F %T') CALIB BEGIN dry ${DRY:-0}" >> $R/RESULTS
  for n in pmcftest-ctl pmcftest-hl0 pgcctrain01-ctl pgcctrain01-hl0 pmcftrain-ctl pmcftrain-hl0; do
    [ -e $R/out/$n ] && { echo "$(date '+%F %T') SKIP $n: out exists" >> $R/RESULTS; continue; }
    bash $M/run.sh 29 ${CAPSTONE_WORK}/perf/tree/images-tree $n:$S/$n:262144
    case $n in *-hl0)
      [ "${DRY:-0}" = 1 ] && continue
      local e; e=$(ls $R/out/$n/results/*.err 2>/dev/null | head -1)
      python3 $R/hmcheck.py "${e:-/nonexistent}" > $R/out/$n.hmcheck 2>&1; local hc=$?
      python3 $R/replay.py "${e:-/nonexistent}" --check ctl ctl hwm+16 > $R/out/$n.replay 2>&1; local rc=$?
      echo "$(date '+%F %T') HMCHECK $n rc=$hc: $(tr '\n' ' ' < $R/out/$n.hmcheck)" >> $R/RESULTS
      echo "$(date '+%F %T') REPLAY $n rc=$rc: $(tr '\n' ' ' < $R/out/$n.replay)" >> $R/RESULTS;;
    esac
  done
  echo "$(date '+%F %T') CALIB END" >> $R/RESULTS
}
main "$@"; exit
