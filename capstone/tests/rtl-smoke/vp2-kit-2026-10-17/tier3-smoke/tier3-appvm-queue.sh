#!/bin/bash
# The appvm arms' board queue after a given sequence ends: the probe arms named in PROBES, then sq0's six --only subsets (each scored at once against the
# prediction cut to its cases), then wm1 last (the known wedge risk). Waits for <after>'s end marker; checks only
# patterns its own command line cannot contain. Usage: tier3-appvm-queue.sh <after seq out> <out>. Ends by itself.
main() {
  set -u
  local W=${1:?after} O=${2:?out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7
  local LB=/tmp/capstone/vp2-prep/llvm-bin-v6 C=/tmp/capstone/wt-gate/capstone/bug-corpora end=$(( $(date +%s) + 14400 )) g t
  until grep -a -q "T3PG DONE\|T3PG STOP\|^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 15; done
  sleep 20
  grep -a -q "T3PG DONE\|T3PG STOP\|^STOP" "$W" || { echo "STOP: $W has not ended" >> "$O"; return 1; }
  # the pymalloc probe arms first (PROBES, e.g. "ppA ppB ppC"): <mnt-hdd>/capstone/pym-probe-hosted-<A|B|C>, scored at once
  for t in ${PROBES:-}; do
    pgrep -f "fpga_driver[.]run_baked" > /dev/null && { echo "T3APPVM STOP: a driver is running" >> "$O"; return 1; }
    echo "T3APPVM $t BEGIN $(date +%T)" >> "$O"
    EXPECT_IMAGES=$I/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$S/tier3-appvm-arm.sh" --board-kit $K "$H/t3-run/$t" \
      $C/cpython/pymalloc-repros/runners/virtual/run-virtual.py --arm virtual-malloc --build "$H/pym-probe-hosted-${t#pp}" \
      --controls-build $I/pym/controls-OFF --llvm-bin $LB >> "$O" 2>&1
    echo "T3APPVM $t END rc=$? $(date +%T)" >> "$O"
    bash "$S/score-arm.sh" "$t" "$H/t3-pred-$t/out" "$H/t3-run/$t" "$O" >> "$O" 2>&1
  done
  for g in a b c d e f; do
    t=sq0$g
    pgrep -f "fpga_driver[.]run_baked" > /dev/null && { echo "T3APPVM STOP: a driver is running" >> "$O"; return 1; }
    local only; only=$(cut -c67-68 "$I/$t.images.SHA256SUMS" | grep -v '^co' | sort -u | paste -sd,)
    echo "T3APPVM $t BEGIN $(date +%T) only $only" >> "$O"
    EXPECT_IMAGES=$I/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$S/tier3-appvm-arm.sh" --board-kit $K "$H/t3-run/$t" \
      $C/sqlite/engine-repros/shared/run-virtual.py --arm virtual-malloc --build $I/sqlite/build-virtual-malloc --only "$only" \
      --llvm-bin $LB >> "$O" 2>&1
    echo "T3APPVM $t END rc=$? $(date +%T)" >> "$O"
    python3 "$S/subset-bundle.py" "$H/t3-pred-sq0/out" "$H/t3-run/$t/out" "$H/t3-subsets/$t" >> "$O" 2>&1 \
      && bash "$S/score-arm.sh" "$t" "$H/t3-subsets/$t" "$H/t3-run/$t" "$O" >> "$O" 2>&1
  done
  t=wm1
  pgrep -f "fpga_driver[.]run_baked" > /dev/null && { echo "T3APPVM STOP: a driver is running" >> "$O"; return 1; }
  echo "T3APPVM $t BEGIN $(date +%T)" >> "$O"
  EXPECT_IMAGES=$I/wm1.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$S/tier3-appvm-arm.sh" --board-kit $K "$H/t3-run/$t" \
    $C/wireshark/wmem-repros/shared/run-defects.py --arm virtual-nested-pools --hosted-build $I/wmem/cases-ON \
    --controls-hosted-build $I/wmem/controls-ON --llvm-bin $LB >> "$O" 2>&1
  echo "T3APPVM $t END rc=$? $(date +%T)" >> "$O"
  bash "$S/score-arm.sh" wm1 "$H/t3-pred-wm1/out" "$H/t3-run/wm1" "$O" >> "$O" 2>&1
  echo "T3APPVM QUEUE2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
