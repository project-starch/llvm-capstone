#!/bin/bash
# Tier 3 batch 2: the four prebuilt bundles on the fixed v6 kit (pre-registered at lane f00b112e536a). Build each
# corpus's capstone-application cases and virtual controls against the v6 SDK with the recorded cache options, then
# run the runner in prebuilt mode. Serial; a failed build skips its run and says so.
main() {
  local SRC=/tmp/capstone/vp2-prep/v6-fix/src O=/tmp/capstone/vp2-prep/v6-fix T=/tmp/capstone/vp2-prep/tier3-fix
  local SDK=$O/out/sdk INC=/tmp/capstone/llvm-dev-current-20261017/lib/clang/22/include
  export CAPSTONE_SDK=$SDK CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock; local LB=-DCAPSTONE_LLVM_BUILD_DIR=$O/llvm-build
  cd $SRC || return 1
  local C=$SRC/capstone/bug-corpora
  for spec in "9|ffmpeg/pool-repros|virtual-malloc|-DFFPOOL_SUBLET=OFF -DFFPOOL_ARCHIVE=/tmp/capstone/ffmpeg-buffer-pool/download/ffmpeg-9.0.1.tar.xz" \
              "10|ffmpeg/pool-repros|virtual-nested-pools|-DFFPOOL_SUBLET=ON -DFFPOOL_ARCHIVE=/tmp/capstone/ffmpeg-buffer-pool/download/ffmpeg-9.0.1.tar.xz" \
              "12|memcached/allocator-repros|virtual-malloc|-DMCP_STOCK_MALLOC=ON -DMCP_SUBLET=OFF" \
              "13|memcached/allocator-repros|virtual-nested-pools|-DMCP_STOCK_MALLOC=OFF -DMCP_SUBLET=ON"; do
    IFS='|' read -r n corpus arm opts <<<"$spec"
    local tag=b$n-$(echo $corpus | tr '/' '-')-$arm B=$T/build-b$n
    rm -rf $B; mkdir -p $B
    echo "T3 $tag BUILD $(date +%T)"
    nice -n 10 bash $C/$corpus/shared/build-cases.sh capstone-application $B/cases -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-isystem $INC" $LB $opts > $B/cases.log 2>&1; local r1=$?
    nice -n 10 bash $C/$corpus/controls/virtual/shared/build-cases.sh capstone-application $B/controls -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-isystem $INC" $LB $opts > $B/controls.log 2>&1; local r2=$?
    echo "T3 $tag BUILT cases rc=$r1 ($(ls $B/cases/bin 2>/dev/null | wc -l) bins) controls rc=$r2 ($(ls $B/controls/bin 2>/dev/null | wc -l) bins) $(date +%T)"
    if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then echo "T3 $tag SKIPPED: build failed"; tail -n 3 $B/cases/work/port.log $B/controls/work/port.log; continue; fi
    local pb=$B/cases/work/port pc=$B/controls/work/port
    timeout 3600 python3 capstone/bug-corpora/tools/run-virtual-cases.py --corpus $C/$corpus --arm $arm --virtual-kit $O/out/kit --sdk $SDK \
      --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --prebuilt $B/cases --prebuilt-controls $B/controls \
      --out $T/$tag/$arm --raw $T/raw-$tag > $T/$tag.log 2>&1
    echo "T3 $tag END $(date +%T) rc=$? $(tail -2 $T/$tag.log | head -1 | cut -c1-140)"
  done
  echo T3_BATCH2_DONE
}
main "$@"; exit
