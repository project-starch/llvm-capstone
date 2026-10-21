#!/bin/bash
# The caplifive-sbi #4 (SBI RFENCE) board pair (2026-10-21; pre-registered in README.md before its boots): pb10 twice on
# BOARD 1 (tree A), cold boots, the CONTROL first: kit bkit7-rev2xDN2 (monitor capstone-sbi f13dbc5), then the VARIANT:
# kit bkit7-rev2xDN2rf, identical but for its monitor worktree /tmp/capstone/csbi-rfence = f13dbc5 + `git cherry-pick fab38bb`
# (PR #4's head, refs/pull/4/head), commit e1db99e (local; tree 8bc1c53d62d7, re-derived from the public commits). Per boot:
# pb10's cells against ref-tB-mimic/pb10 (compare-pilot), the kernel's SBI readings of THIS boot (console-readings.py),
# b0-stats' ticks, and the S-23/pc=0/INJS counts. Booked on the lab hub; refused on a 409. A boot that does not complete
# stops the pair. Usage: tier3-rfence-pair.sh <out>. RFP_RUN / RFP_BUSY / RFP_ROOT / RFP_BOOK stub the dry run.
main() {
  set -u
  local O=${1:?out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone N=/mnt/hdd/$USER/capstone/t3-images-cw G=/tmp/capstone/wt-gate LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb-nomovc-cw/sdk C=/tmp/capstone/wt-gate/capstone/bug-corpora P=$N/prebuilt
  local RUN=$G/capstone/bug-corpora/tools/run-virtual-cases.py REF=$N/ref-tB-mimic/pb10 RR=${RFP_ROOT:-$H/t3-run-treeA/rfp}
  local BUSY="python3 $G/capstone/tests/rtl-smoke/board-busy.py 1"
  [ "$(sha256sum $RUN | cut -c1-16)" = 9b5e409c92e1fe66 ] && grep -q "runner capstone/bug-corpora/tools/run-virtual-cases.py 9b5e409c92e1fe66 " $REF.stamp \
    || { echo "RFP STOP: the runner is not pb10's reference's" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-rfence rev-parse --short=7 HEAD)" = e1db99e ] && [ "$(git -C /tmp/capstone/csbi-rfence rev-parse HEAD^{tree} | cut -c1-12)" = 8bc1c53d62d7 ] \
    && [ -z "$(git -C /tmp/capstone/csbi-rfence status --porcelain)" ] || { echo "RFP STOP: the variant monitor is not a clean e1db99e / tree 8bc1c53d62d7" >> "$O"; return 1; }
  local arm kit mon d rc bid B
  for arm in ctl:bkit7-rev2xDN2:f13dbc5 var:bkit7-rev2xDN2rf:e1db99e; do
    IFS=: read -r arm kit mon <<<"$arm"; local K=$H/t3-smoke/$kit
    ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env && grep -qx "BOARD_MONITOR_COMMIT=$mon" $K/board.env \
      && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env \
      || { echo "RFP STOP: kit $kit is not the board-1 $mon kit" >> "$O"; return 1; }
    d=$RR-$arm; [ -e "$d" ] && { echo "RFP STOP: $d exists" >> "$O"; return 1; }
    ${RFP_BUSY:-$BUSY} > /dev/null || { echo "RFP STOP: board 1 is busy" >> "$O"; return 1; }
    bid=$(${RFP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book 1 40 "vp2-tier12 RFENCE pair $arm (caplifive-sbi #4), board 1"); rc=$?
    [ $rc = 3 ] && { echo "RFP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "RFP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    mkdir -p $d; echo "RFP $arm BEGIN $(date +%T) kit $kit monitor $mon booking ${bid:-none}" >> "$O"
    if [ -n "${RFP_RUN:-}" ]; then $RFP_RUN $arm $REF $d > $d/run.log 2>&1; rc=$?
    else ( cd $G && taskset -c 16-31 timeout 10800 python3 $RUN --corpus $C/ffmpeg/pool-repros --arm virtual-nested-pools --board-kit $K --sdk $SDK \
             --llvm-bin $LB --prebuilt $P/b10/cases --prebuilt-controls $P/b10/controls --out $d/out --raw $d/raw ) > $d/run.log 2>&1; rc=$?; fi
    [ -n "$bid" ] && ${RFP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
    echo "RFP $arm END $(date +%T) rc=$rc :: $(grep -a '^--- ' $d/run.log | tail -1)" >> "$O"
    python3 $S/compare-pilot.py $REF/out $d/out > $d.cmp.txt 2>&1; echo "RFP $arm COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
    python3 $S/console-readings.py $d/raw/guest/driver.log 2>&1 | sed "s/^/RFP $arm console: /" >> "$O"
    if [ -n "$B" ] && [ -s "$B" ]; then
      echo "RFP $arm readings: b0-stats $(grep -a -o 'ticks=[0-9]*' $B | head -1) pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
      echo "RFP $arm wall: $(grep -a 'BOARD_RUN_START\|DRIVER rc' $d/raw/guest/SEQ 2>/dev/null | awk '{print $1, $2, $3}' | tr '\n' ' ')" >> "$O"
    else echo "RFP $arm readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "RFP STOP after $arm: the boot did not complete (no vbatch END rc=0)" >> "$O"; return 1; }
  done
  echo "RFP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
