#!/usr/bin/env bash
# build-reuse.sh TREE ARM OUT: one PostgreSQL reuse arm (reuse-hooks/postgres/build.sh), as run-5 built it.
set -euo pipefail
F=$HOME/asr-work/fix-postgres
TREE=$1 ARM=$2 OUT=$3
export CAPSTONE_REPO_ROOT=$TREE CAPSTONE_TMP_ROOT=$F/tmp
export CAPSTONE_LLVM_BUILD_DIR=$HOME/lanes/gatefix/llvm CAPSTONE_LLVM_BIN=$HOME/lanes/gatefix/llvm/bin
export JOBS=12
cd $TREE
source capstone/tests/capstone-test-env.sh
bash capstone/experiments/study/reuse-hooks/postgres/build.sh $ARM $OUT
