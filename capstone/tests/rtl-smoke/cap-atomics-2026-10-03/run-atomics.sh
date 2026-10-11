#!/bin/bash
# The pre-registered bare session: call-retpc (control), then cap-atomics. One power cycle per image.
main() {
  set -u
  local F=$(cd "$(dirname "$0")" && pwd) OUT=${ATOMICS_OUT:-/tmp/capstone/atomics-run}
  local L=$F/../sup-bare-2026-10-03-ladder V1=$F/../sup-bare-2026-10-02/images
  export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"
  export FPGA_BITSTREAM=caplifive_supcall_36a641e0b.bit
  mkdir -p $OUT
  # board_rec addresses: call-retpc's (a05ca464) from the ELF the 2026-10-02 runs used, whose objcopy is
  # byte-identical to the committed image; cap-atomics' from build.sh.
  local jobs=("control-call-retpc|$V1/call-retpc.bin|80003800"
              "cap-atomics|$F/images/cap-atomics.bin|80004040")
  for j in "${jobs[@]}"; do
    IFS='|' read -r tag img rec <<< "$j"
    echo "$tag START $(date +%H:%M:%S) $(sha256sum $img | cut -c1-16)" >> $OUT/SEQ
    SUP_IMG=$img SUP_OUT=$OUT/runs/$tag SUP_REC_ADDR=$rec SUP_TIMEOUT=90 \
      taskset -c 0-7,32-39 nice -n 10 python3 $L/run_sup_bare.py > $OUT/runs-$tag.log 2>&1
    echo "$tag END rc=$? $(date +%H:%M:%S)" >> $OUT/SEQ
    sleep 8
  done
}
main "$@"; exit $?
