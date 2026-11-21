#!/usr/bin/env bash
# chain.sh: the pre-registered twin runs of CAPSTONE_HEAP_MAPPING_REUSE_HWM, in order (short ones first), one at a time
# under the shared qemu.lock; then hmcheck over each logged run. Output perf/mrhwm/out, perf/mrhwm/RESULTS.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}" "${CAPSTONE_LLVM:=$HOME/dev/llvm-capstone}"
main() {
  local R=${CAPSTONE_WORK}/perf/mrhwm S=${CAPSTONE_WORK}/perf/mrhwm/stages
  # The tree kernel's COMMAND_LINE_SIZE is 1024 (run-stage-twin-x3.py:69); the recorded flags are 587 bytes.
  export TWIN_CMDLINE_MAX=1024
  export OUTROOT=$R EXTRA=" capstone_vm.node_table_pool=4 capstone_vm.dom_grant_root=/tmp/nt/sr capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1"
  [ -n "${PREREG:-}" ] || { echo "no PREREG commit given"; exit 1; }
  git -C ${CAPSTONE_WORK}/wt-mrhwm fetch -q origin mapping-cache-hwm || exit 1
  git -C ${CAPSTONE_WORK}/wt-mrhwm merge-base --is-ancestor "$PREREG" origin/mapping-cache-hwm || { echo "PREREG $PREREG not on origin"; exit 1; }
  echo "$(date '+%F %T') CHAIN BEGIN prereg $PREREG dry ${DRY:-0}" >> $R/RESULTS
  for n in mcftest-ctl mcftest-h mcftest-hl gcctrain01-ctl gcctrain01-h gcctrain01-hl mcftrain-fl mcftrain-h mcftrain-hl; do
    bash $R/run.sh 29 ${CAPSTONE_WORK}/perf/tree/images-tree $n:$S/$n:262144
    case $n in *-hl|*-fl)
      [ "${DRY:-0}" = 1 ] && continue
      local e; e=$(ls $R/out/$n/results/*.err 2>/dev/null | head -1)
      local f=""; [ "${n%-hl}" != "$n" ] && f=--hwm
      python3 $R/hmcheck.py "${e:-/nonexistent}" $f > $R/out/$n.hmcheck 2>&1; local hc=$?
      echo "$(date '+%F %T') HMCHECK $n $f rc=$hc: $(tr '\n' ' ' < $R/out/$n.hmcheck)" >> $R/RESULTS;;
    esac
  done
  echo "$(date '+%F %T') CHAIN END" >> $R/RESULTS
}
main "$@"; exit
