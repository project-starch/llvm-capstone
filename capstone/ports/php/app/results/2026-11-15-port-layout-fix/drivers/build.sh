#!/usr/bin/env bash
# build.sh WHAT: one build of the PHP port layout fix into ~/asr-work/fix-php/prebuilt/.
#   reuse-stock | reuse-protected   reuse-hooks/php/build.sh (as run-5 build-arm.sh)
#   trace-stock | trace-protected   reuse-hooks/php/build-layout-trace.sh
#   whole                           the corpus image (as run-4 build-whole.sh php protected) plus probe-nested.dom
F=$HOME/asr-work/fix-php
cd "$F"
export CAPSTONE_REPO_ROOT=$F/tree CAPSTONE_LLVM_BUILD_DIR=$HOME/lanes/gatefix/llvm CAPSTONE_LLVM_BIN=$HOME/lanes/gatefix/llvm/bin
export CAPSTONE_TMP_ROOT=$F/tmp JOBS=8
export PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p "$F/prebuilt" "$F/tmp" "$F/logs"
W=$1 H=tree/capstone/experiments/study/reuse-hooks/php
t0=$(date +%s)
case $W in
  reuse-stock|reuse-protected) bash $H/build.sh ${W#reuse-} $F/prebuilt/php-${W#reuse-} ;;
  trace-stock|trace-protected) bash $H/build-layout-trace.sh ${W#trace-} $F/prebuilt/trace-${W#trace-} ;;
  whole)
    export MUSL_CACHE_ROOT=$F/musl-whole CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin
    O=$F/prebuilt/whole-php-protected; rm -rf "$O"
    CAPSTONE_VIRTUAL_LIFETIMES=1 bash tree/capstone/ports/common/application/build-virtual.sh php "$O" \
      && PHP_VIRTUAL_OUT=$O CAPSTONE_APPLICATION_PROFILE=virtual bash tree/capstone/ports/php/app/build-virtual.sh nested-probe ;;
  *) echo "what?" >&2; exit 2 ;;
esac > "$F/logs/build-$W.log" 2>&1
echo "BUILD $W rc=$? $(( $(date +%s) - t0 ))s $(date -u +%T)"
