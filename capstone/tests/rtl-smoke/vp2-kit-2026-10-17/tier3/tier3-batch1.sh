#!/bin/bash
# Tier 3 batch 1: the five plain case.c bundles on the fixed v6 kit (pre-registered at lane 8d61a2c302ce), serial.
main() {
  cd /tmp/capstone/vp2-prep/v6-fix/src || return 1
  local O=/tmp/capstone/vp2-prep/v6-fix T=/tmp/capstone/vp2-prep/tier3-fix
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  mkdir -p $T
  for spec in "5|ffmpeg/carved-repros" "7|ffmpeg/plain-heap-repros" "8|ffmpeg/plain-temporal-repros" "23|wireshark/plain-heap-repros" "24|wireshark/plain-temporal-repros"; do
    local n=${spec%%|*} c=${spec#*|} tag
    tag=b$n-$(echo $c | tr '/' '-')
    echo "T3 $tag START $(date +%T)"
    timeout 3600 python3 capstone/bug-corpora/tools/run-virtual-cases.py --corpus capstone/bug-corpora/$c --arm virtual-malloc \
      --virtual-kit $O/out/kit --sdk $O/out/sdk --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 \
      --out $T/$tag/virtual-malloc --raw $T/raw-$tag > $T/$tag.log 2>&1
    echo "T3 $tag END $(date +%T) rc=$? $(tail -2 $T/$tag.log | head -1 | cut -c1-120)"
  done
  echo T3_BATCH1_DONE
}
main "$@"; exit
