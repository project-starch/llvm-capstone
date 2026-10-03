#!/bin/bash
# The pre-registered bare session: call-retpc (control), intloop-q64 (matched control), capstl-q64, -q47, -q16.
# One power cycle per image, so a wedge in one costs nothing in the next.
main() {
  set -u
  local F=$(cd "$(dirname "$0")" && pwd) OUT=${CAPSTL_OUT:-/tmp/capstone/capstl-run}
  local L=$F/../sup-bare-2026-10-03-ladder V1=$F/../sup-bare-2026-10-02/images
  export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"
  export FPGA_BITSTREAM=caplifive_supcall_36a641e0b.bit
  mkdir -p $OUT
  local jobs=("control-call-retpc|$V1/call-retpc.bin|80003800")
  for n in intloop-q64 capstl-q64 capstl-q47 capstl-q16; do jobs+=("$n|$F/images/$n.bin|80003c40"); done
  for j in "${jobs[@]}"; do
    IFS='|' read -r tag img rec <<< "$j"
    echo "$tag START $(date +%H:%M:%S) $(sha256sum $img | cut -c1-16)" >> $OUT/SEQ
    SUP_IMG=$img SUP_OUT=$OUT/runs/$tag SUP_REC_ADDR=$rec SUP_TIMEOUT=${CAPSTL_TIMEOUT:-90} \
      taskset -c 0-7,32-39 nice -n 10 python3 $L/run_sup_bare.py > $OUT/runs-$tag.log 2>&1
    echo "$tag END rc=$? $(date +%H:%M:%S)" >> $OUT/SEQ
    sleep 8
  done
}
main "$@"; exit $?
