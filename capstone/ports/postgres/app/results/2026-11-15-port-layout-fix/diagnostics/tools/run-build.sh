#!/usr/bin/env bash
# run-build.sh TREE ARM PATCH OUT: one census variant of the reuse build (run-5 build-variant.sh, this tree).
set -euo pipefail
F=$HOME/asr-work/fix-postgres
export TREE=$1 EXTRA_PATCH=$3
export CAPSTONE_REPO_ROOT=$TREE CAPSTONE_TMP_ROOT=$F/tmp
export CAPSTONE_LLVM_BUILD_DIR=$HOME/lanes/gatefix/llvm CAPSTONE_LLVM_BIN=$HOME/lanes/gatefix/llvm/bin
export JOBS=12
cd "$TREE"
source capstone/tests/capstone-test-env.sh
bash "$F/census/build-variant.sh" "$2" "$4"
