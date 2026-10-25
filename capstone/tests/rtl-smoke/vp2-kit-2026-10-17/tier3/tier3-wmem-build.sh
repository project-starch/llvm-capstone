#!/bin/bash
# Tier 3: wireshark/wmem-repros hosted builds against the v6-fix SDK (cases and controls, WM_SUBLET OFF and ON).
main() {
  local SRC=/tmp/capstone/vp2-prep/v6-fix/src O=/tmp/capstone/vp2-prep/v6-fix T=/tmp/capstone/vp2-prep/tier3-fix/wmem
  local P=$SRC/capstone/ports/wireshark/wmem C=$SRC/capstone/bug-corpora/wireshark/wmem-repros
  local INC=/tmp/capstone/llvm-dev-current-20261017/lib/clang/22/include
  mkdir -p $T
  for s in OFF ON; do for kind in cases controls; do
    local dir=$C; [ $kind = controls ] && dir=$C/controls
    local B=$T/$kind-$s; rm -rf $B
    echo "WMEM $kind $s CONFIGURE $(date +%T)"
    CAPSTONE_SDK=$O/out/sdk nice -n 10 cmake --preset capstone-application -S $P -B $B -DCAPSTONE_SDK=$O/out/sdk \
      -DCAPSTONE_LLVM_BUILD_DIR=$O/llvm-build -DWM_CORPUS_DIR=$dir -DWM_SUBLET=$s \
      -DWM_ARCHIVE=/tmp/capstone/wireshark-wmem/download/wireshark-4.6.8.tar.xz -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-isystem $INC" > $T/$kind-$s.log 2>&1 \
      && CAPSTONE_SDK=$O/out/sdk nice -n 10 cmake --build $B -j 8 >> $T/$kind-$s.log 2>&1
    echo "WMEM $kind $s rc=$? bins=$(ls $B/bin 2>/dev/null | wc -l) $(date +%T)"
  done; done
  echo WMEM_BUILD_DONE
}
main "$@"; exit
