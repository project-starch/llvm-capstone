#!/usr/bin/env bash
# build-corpus.sh TREE ARM OUT: one sql-repros image (build-trap.sh), with run-4 build-whole.sh settings.
set -euo pipefail
F=$HOME/asr-work/fix-postgres
TREE=$1 ARM=$2 OUT=$3
export CAPSTONE_REPO_ROOT=$TREE CAPSTONE_TMP_ROOT=$F/tmp
export CAPSTONE_LLVM_BUILD_DIR=$HOME/lanes/gatefix/llvm CAPSTONE_LLVM_BIN=$HOME/lanes/gatefix/llvm/bin
export JOBS=12 FFAPP_JOBS=12 CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin
export PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
cd $TREE
bash capstone/bug-corpora/postgres/sql-repros/shared/build-trap.sh $ARM $OUT
