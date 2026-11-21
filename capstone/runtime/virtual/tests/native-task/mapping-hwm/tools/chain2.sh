#!/usr/bin/env bash
# chain2.sh: Amendment 1's reruns of the logged arms with the fixed TEST_LOG images, after chain.sh, under the shared
# qemu.lock; then hmcheck over each. Output perf/mrhwm/out, perf/mrhwm/RESULTS.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}"
main() {
  shopt -s nullglob
  local R=${CAPSTONE_WORK}/perf/mrhwm S=${CAPSTONE_WORK}/perf/mrhwm/stages
  export TWIN_CMDLINE_MAX=1024
  export OUTROOT=$R EXTRA=" capstone_vm.node_table_pool=4 capstone_vm.dom_grant_root=/tmp/nt/sr capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1"
  [ -n "${PREREG:-}" ] || { echo "no PREREG commit given"; exit 1; }
  git -C ${CAPSTONE_WORK}/wt-mrhwm fetch -q origin mapping-cache-hwm || exit 1
  git -C ${CAPSTONE_WORK}/wt-mrhwm merge-base --is-ancestor "$PREREG" origin/mapping-cache-hwm || { echo "PREREG $PREREG not on origin"; exit 1; }
  echo "$(date '+%F %T') CHAIN2 BEGIN prereg $PREREG" >> $R/RESULTS
  for n in mcftest-hl gcctrain01-hl mcftrain-fl; do
    [ -e $R/out/$n ] && { mkdir -p $R/attic-a1 && mv $R/out/$n $R/out/$n.* $R/attic-a1/ || exit 1; }
    bash $R/run.sh 29 ${CAPSTONE_WORK}/perf/tree/images-tree $n:$S/$n:262144
    local e; e=$(ls $R/out/$n/results/*.err 2>/dev/null | head -1)
    local f=""; [ "${n%-hl}" != "$n" ] && f=--hwm
    python3 $R/hmcheck.py "${e:-/nonexistent}" $f > $R/out/$n.hmcheck 2>&1; local hc=$?
    echo "$(date '+%F %T') HMCHECK $n $f rc=$hc: $(tr '\n' ' ' < $R/out/$n.hmcheck)" >> $R/RESULTS
  done
  echo "$(date '+%F %T') CHAIN2 END" >> $R/RESULTS
}
main "$@"; exit
