#!/usr/bin/env bash
# chain3.sh: pre-registration 2 (HWM with 16 MiB slack) on the twin, one run at a time under the shared qemu.lock;
# hmcheck and the replay check over each logged run. Output perf/mrhwm/out, perf/mrhwm/RESULTS.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}"
main() {
  shopt -s nullglob
  local R=${CAPSTONE_WORK}/perf/mrhwm S=${CAPSTONE_WORK}/perf/mrhwm/stages
  export TWIN_CMDLINE_MAX=1024
  export OUTROOT=$R EXTRA=" capstone_vm.node_table_pool=4 capstone_vm.dom_grant_root=/tmp/nt/sr capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1"
  [ -n "${PREREG:-}" ] || { echo "no PREREG commit given"; exit 1; }
  git -C ${CAPSTONE_WORK}/wt-mrhwm fetch -q origin mapping-cache-hwm || exit 1
  git -C ${CAPSTONE_WORK}/wt-mrhwm merge-base --is-ancestor "$PREREG" origin/mapping-cache-hwm || { echo "PREREG $PREREG not on origin"; exit 1; }
  echo "$(date '+%F %T') CHAIN3 BEGIN prereg $PREREG dry ${DRY:-0}" >> $R/RESULTS
  for n in gcctrain01-h16 mcftest-h16 mcftrain-h16 gcctrain01-hl16 mcftrain-hl16 \
           lbm-ctl lbm-h16 nab-ctl nab-h16 imagick-ctl imagick-h16 xz-ctl xz-h16 gcctest-ctl gcctest-h16 \
           mcftest-ctl2 gcctrain01-ctl2; do
    [ -e $R/out/$n ] && { echo "$(date '+%F %T') SKIP $n: out exists" >> $R/RESULTS; continue; }
    bash $R/run.sh 29 ${CAPSTONE_WORK}/perf/tree/images-tree $n:$S/$n:262144
    case $n in *-hl16)
      [ "${DRY:-0}" = 1 ] && continue
      local e; e=$(ls $R/out/$n/results/*.err 2>/dev/null | head -1)
      python3 $R/hmcheck.py "${e:-/nonexistent}" --hwm --slack 16777216 > $R/out/$n.hmcheck 2>&1; local hc=$?
      python3 $R/replay.py "${e:-/nonexistent}" --check hwm+16 hwm+16 > $R/out/$n.replay 2>&1; local rc=$?
      echo "$(date '+%F %T') HMCHECK $n --slack 16777216 rc=$hc: $(tr '\n' ' ' < $R/out/$n.hmcheck)" >> $R/RESULTS
      echo "$(date '+%F %T') REPLAY $n rc=$rc: $(tr '\n' ' ' < $R/out/$n.replay)" >> $R/RESULTS;;
    esac
  done
  echo "$(date '+%F %T') CHAIN3 END" >> $R/RESULTS
}
main "$@"; exit
