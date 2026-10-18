#!/bin/bash
# Tier 3 on the board: build the four prebuilt arms' cases and virtual controls from the frozen SDK (t3-freeze), with
# the recorded cache options (as tier3/tier3-batch2.sh did on v6-fix), into <out>/b<n>/{cases,controls}. Build only:
# the board boot reads them through run-virtual-cases.py --prebuilt, and the QEMU prediction runs on the same trees.
# Usage: tier3-prebuilt-build.sh <out>. Ends by itself.
main() {
  set -u
  local O=${1:?out} F=/mnt/hdd/$USER/capstone/t3-freeze G=/tmp/capstone/wt-gate; local SDKROOT=${T3_SDKROOT:-$F}
  local SDK=$SDKROOT/sdk INC=$F/llvm/lib/clang/22/include C=$G/capstone/bug-corpora
  export CAPSTONE_SDK=$SDK
  local LB=-DCAPSTONE_LLVM_BUILD_DIR=$F/llvm FFA=-DFFPOOL_ARCHIVE=/tmp/capstone/ffmpeg-buffer-pool/download/ffmpeg-9.0.1.tar.xz
  local spec n corpus arm opts B r1 r2
  for spec in "9|ffmpeg/pool-repros|virtual-malloc|-DFFPOOL_SUBLET=OFF $FFA" "10|ffmpeg/pool-repros|virtual-nested-pools|-DFFPOOL_SUBLET=ON $FFA" \
              "12|memcached/allocator-repros|virtual-malloc|-DMCP_STOCK_MALLOC=ON -DMCP_SUBLET=OFF" \
              "13|memcached/allocator-repros|virtual-nested-pools|-DMCP_STOCK_MALLOC=OFF -DMCP_SUBLET=ON"; do
    IFS='|' read -r n corpus arm opts <<<"$spec"
    B=$O/b$n
    [ -e "$B" ] && { echo "PB b$n SKIPPED: $B exists"; continue; }
    mkdir -p "$B"
    echo "PB b$n BUILD $(date +%T) $corpus $arm"
    # shellcheck disable=SC2086
    nice -n 10 bash $C/$corpus/shared/build-cases.sh capstone-application $B/cases -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-isystem $INC" $LB $opts > $B/cases.log 2>&1; r1=$?
    # shellcheck disable=SC2086
    nice -n 10 bash $C/$corpus/controls/virtual/shared/build-cases.sh capstone-application $B/controls -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-isystem $INC" $LB $opts > $B/controls.log 2>&1; r2=$?
    echo "PB b$n BUILT $(date +%T) cases rc=$r1 ($(ls $B/cases/bin 2>/dev/null | wc -l) bins) controls rc=$r2 ($(ls $B/controls/bin 2>/dev/null | wc -l) bins)"
  done
  echo "PB DONE $(date +%T)"
}
main "$@"; exit $?
