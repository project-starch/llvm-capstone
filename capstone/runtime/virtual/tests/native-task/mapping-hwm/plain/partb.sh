#!/usr/bin/env bash
# partb.sh (perf/plainhwm): pre-registration 3 part B, the Plain H16 rule on the twin: H16 and HL16 of mcf test, gcc
# train01 and mcf train, and two noise controls (ctl2). hmcheck --slack 16 MiB and replay --check hwm+16 over each HL16
# log. Output perf/plainhwm/out, RESULTS.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}"
main() {
  local R=${CAPSTONE_WORK}/perf/plainhwm S=${CAPSTONE_WORK}/perf/plainhwm/stages M=${CAPSTONE_WORK}/perf/mrhwm
  export TWIN_CMDLINE_MAX=1024 OUTROOT=$R
  export EXTRA=" capstone_vm.node_table_pool=4 capstone_vm.dom_grant_root=/tmp/nt/sr capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1"
  [ -n "${PREREG:-}" ] || { echo "no PREREG commit given"; exit 1; }
  git -C ${CAPSTONE_WORK}/wt-mrhwm fetch -q origin mapping-cache-hwm || exit 1
  git -C ${CAPSTONE_WORK}/wt-mrhwm merge-base --is-ancestor "$PREREG" origin/mapping-cache-hwm || { echo "PREREG $PREREG not on origin"; exit 1; }
  echo "$(date '+%F %T') PARTB BEGIN prereg $PREREG dry ${DRY:-0}" >> $R/RESULTS
  for n in pgcctrain01-h16 pmcftest-h16 pmcftrain-h16 pgcctrain01-hl16 pmcftest-hl16 pmcftrain-hl16 pgcctrain01-ctl2 pmcftrain-ctl2; do
    [ -e $R/out/$n ] && { echo "$(date '+%F %T') SKIP $n: out exists" >> $R/RESULTS; continue; }
    bash $M/run.sh 29 ${CAPSTONE_WORK}/perf/tree/images-tree $n:$S/$n:262144
    case $n in *-hl16)
      [ "${DRY:-0}" = 1 ] && continue
      local e; e=$(ls $R/out/$n/results/*.err 2>/dev/null | head -1)
      python3 $R/hmcheck.py "${e:-/nonexistent}" --hwm --slack 16777216 > $R/out/$n.hmcheck 2>&1; local hc=$?
      python3 $R/replay.py "${e:-/nonexistent}" --check hwm+16 hwm+16 > $R/out/$n.replay 2>&1; local rc=$?
      echo "$(date '+%F %T') HMCHECK $n --slack 16777216 rc=$hc: $(tr '\n' ' ' < $R/out/$n.hmcheck)" >> $R/RESULTS
      echo "$(date '+%F %T') REPLAY $n rc=$rc: $(tr '\n' ' ' < $R/out/$n.replay)" >> $R/RESULTS;;
    esac
  done
  echo "$(date '+%F %T') PARTB END" >> $R/RESULTS
}
main "$@"; exit
