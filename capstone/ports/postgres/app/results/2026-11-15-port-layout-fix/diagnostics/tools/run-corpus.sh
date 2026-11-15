#!/usr/bin/env bash
# run-corpus.sh NAME ONLY [STEP_TIMEOUT]: one sql-repros virtual-pg-pools boot of the new image, with
# run-4's kit, fixture, share and environment (eval/run4.sh), into ~/asr-work/eval/run-6/corpora/NAME.
set -euo pipefail
E=$HOME/asr-work/eval F=$HOME/asr-work/fix-postgres TREE=$HOME/asr-work/fix-postgres/tree-38b1e0a
NAME=$1 ONLY=$2
cd $E && source $TREE/capstone/runtime/virtual/tests/trap-platform/eval-p13.env
export CAPSTONE_QEMU_LOCK=$HOME/asr-work/qemu-gate.lock CAPSTONE_QEMU_DOMAIN_LOCK=$F/locks/$NAME.lock
export CAPSTONE_TMP_ROOT=$F/tmp PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
[[ -z ${3:-} ]] || export CAPSTONE_STEP_TIMEOUT=$3
mkdir -p $F/locks $F/raw
t0=$(date +%s)
set +e
( cd $TREE && python3 capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py --arm virtual-pg-pools \
    --virtual-kit $E/kit-run-4 --image $F/build/corpus-protected/image/postgres.dom \
    --fixture $E/prebuilt/reuse/pg-fixture/cluster --pg-share $E/prebuilt/reuse/pg-share \
    --llvm-bin $HOME/lanes/gatefix/llvm/bin --out $E/run-6/corpora/$NAME --raw $F/raw/$NAME --only $ONLY )
rc=$?
set -e
echo "CORPUS $NAME rc=$rc wall=$(( $(date +%s) - t0 ))s step_timeout=${CAPSTONE_STEP_TIMEOUT:-default}" | tee -a $F/logs/runs.txt
exit $rc
