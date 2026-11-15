#!/usr/bin/env bash
# run-reuse.sh ARM REP IMAGE WORKLOAD WORK [SETTING...]: one boot of run-pgbench-reuse.py with run-5's
# options (kit-run-3, prebuilt pg-share, 3600 s step timeout); SETTINGs only for diagnostic runs.
set -euo pipefail
F=$HOME/asr-work/fix-postgres TREE=$HOME/asr-work/fix-postgres/tree-38b1e0a
ARM=$1 REP=$2 IMAGE=$3 WL=$4 WORK=$5; shift 5
S=(); for x in "$@"; do S+=(--setting "$x"); done
mkdir -p $F/locks
export CAPSTONE_QEMU_LOCK=$HOME/asr-work/qemu-gate.lock CAPSTONE_QEMU_DOMAIN_LOCK=$F/locks/$(basename $WORK).lock
t0=$(date +%s)
set +e
python3 $TREE/capstone/experiments/study/reuse-hooks/postgres/run-pgbench-reuse.py --arm $ARM --rep $REP \
  --image $IMAGE --share $HOME/asr-work/eval/prebuilt/reuse/pg-share --workload $WL \
  --kit $HOME/asr-work/eval/kit-run-3 --work $WORK --step-timeout 3600 "${S[@]}"
rc=$?
set -e
echo "RUN $(basename $WORK) rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a $F/logs/runs.txt
exit $rc
