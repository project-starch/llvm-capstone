#!/bin/bash
# The corpus pass's REPAIRED arms on one board (2026-10-20; pre-registered in README.md before its first boot): pb13q15
# (memcached allocator, virtual-nested-pools, the Q-15 placement repair), pgm1q15 (postgres mmgr, virtual-pg-pools) and
# wm1q15 (wireshark wmem, virtual-nested-pools, the repaired 0001 patch), one cold boot each, in that order.
# Each runs its OWN reference's tree, by hash: pb13q15 and pgm1q15 from /tmp/capstone/wt-q15cw-pass (detached at
# d893ec627057, the tree the compiler lane's q15 references ran: run-virtual-cases.py 9b5e409c92e1fe66, mmgr run-defects.py
# 543fc02a119c4459); wm1q15 from /tmp/capstone/wt-cpy-pass (dev d0f0f4b41ae6: wmem run-defects.py d05e9077541ef3f1,
# tier3-appvm-arm.sh 666d920eedef445b). Scored by compare-pilot.py against the silicon-mimic references: q15/tB-pb13/bundle,
# q15/tB-pgm1/bundle, ref-tB-mimic/wm1q15/out. Completion is the driver's vbatch END line (tier3-pass-seq2.sh's test).
# Booked on the lab hub; refused on a 409. A boot that does not complete stops the run.
# Usage: tier3-pass-rep.sh <board 1|2> <after|-> <out>: starts when <after> reads "CPY nm DONE" or "PASS seq2-1 DONE"
# (holds on a STOP/HOLD); "-" at once. REP_RUN / REP_BUSY / REP_ROOT / REP_BOOK stub the dry run.
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} O=${3:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone N=/mnt/hdd/$USER/capstone/t3-images-cw LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb-nomovc-cw/sdk Q=/mnt/hdd/$USER/capstone/t3-images-cw/q15
  local WQ=/tmp/capstone/wt-q15cw-pass WD=/tmp/capstone/wt-cpy-pass K bs RR BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; RR=${REP_ROOT:-$H/t3-run-treeA/pass}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; RR=${REP_ROOT:-$H/t3-run-treeB/pass}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "REP STOP: board $BRD" >> "$O"; return 2;;
  esac
  if [ "$W" != - ]; then
    until grep -a -q "^CPY nm DONE\|^PASS seq2-1 DONE\|^CPY STOP\|^CPY HOLD\|^PASS STOP\|^PASS HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
    grep -a -q "^CPY nm DONE\|^PASS seq2-1 DONE" "$W" || { echo "REP HOLD: the previous arm did not end DONE ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
  fi
  [ "$(git -C $WQ rev-parse --short=12 HEAD)" = d893ec627057 ] && [ -z "$(git -C $WQ status --porcelain)" ] || { echo "REP STOP: $WQ is not a clean d893ec627057" >> "$O"; return 1; }
  [ "$(git -C $WD rev-parse --short=12 HEAD)" = d0f0f4b41ae6 ] && [ -z "$(git -C $WD status --porcelain)" ] || { echo "REP STOP: $WD is not a clean d0f0f4b41ae6" >> "$O"; return 1; }
  local RC=$WQ/capstone/bug-corpora/tools/run-virtual-cases.py RM=$WQ/capstone/bug-corpora/postgres/mmgr-repros/shared/run-defects.py
  local RW=$WD/capstone/bug-corpora/wireshark/wmem-repros/shared/run-defects.py AP=$WD/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  [ "$(sha256sum $RC | cut -c1-16)" = 9b5e409c92e1fe66 ] && [ "$(sha256sum $RM | cut -c1-16)" = 543fc02a119c4459 ] \
    && [ "$(sha256sum $RW | cut -c1-16)" = d05e9077541ef3f1 ] && [ "$(sha256sum $AP | cut -c1-16)" = 666d920eedef445b ] \
    || { echo "REP STOP: a runner is not its reference's" >> "$O"; return 1; }
  grep -q "runner .*run-defects.py d05e9077541ef3f1 .*appvm-arm 666d920eedef445b" $N/ref-tB-mimic/wm1q15.stamp && grep -q "SWITCHES movc_null_scalar=1 compact_rounding=ON" $N/ref-tB-mimic/wm1q15.stamp \
    || { echo "REP STOP: wm1q15's stamp does not name its runner and the mimic switches" >> "$O"; return 1; }
  [ -s $Q/tB-pb13/bundle/verdicts.jsonl ] && [ -s $Q/tB-pgm1/bundle/verdicts.jsonl ] && [ -s $N/ref-tB-mimic/wm1q15/out/verdicts.jsonl ] || { echo "REP STOP: a reference bundle is missing" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "REP STOP: kit $K" >> "$O"; return 1; }
  echo "REP START $(date +%T) board $BRD" >> "$O"
  local tag ref d rc bid B
  for tag in pb13q15 pgm1q15 wm1q15; do
    case $tag in pb13q15) ref=$Q/tB-pb13/bundle;; pgm1q15) ref=$Q/tB-pgm1/bundle;; wm1q15) ref=$N/ref-tB-mimic/wm1q15/out;; esac
    d=$RR-$tag; [ -e "$d" ] && { echo "REP STOP: $d exists" >> "$O"; return 1; }
    ${REP_BUSY:-$BUSY} > /dev/null || { echo "REP STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${REP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2-tier12 pass $tag (repaired arm, board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "REP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "REP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "PASS $tag BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
    if [ -n "${REP_RUN:-}" ]; then mkdir -p $d; $REP_RUN $tag $ref $d > $d/run.log 2>&1; rc=$?
    else case $tag in
      pb13q15) mkdir -p $d; ( cd $WQ && taskset -c 16-31 timeout 10800 python3 $RC --corpus $WQ/capstone/bug-corpora/memcached/allocator-repros --arm virtual-nested-pools \
                 --board-kit $K --sdk $SDK --llvm-bin $LB --prebuilt $Q/prebuilt/b13/cases --prebuilt-controls $Q/prebuilt/b13/controls --out $d/out --raw $d/raw ) > $d/run.log 2>&1; rc=$?;;
      pgm1q15) mkdir -p $d; ( cd $WQ && taskset -c 16-31 timeout 10800 python3 $RM $d/out --board-kit $K --arm virtual-pg-pools --hosted-build $Q/pg/mmgr-cases-ON \
                 --controls-hosted-build $Q/pg/mmgr-controls-ON --llvm-bin $LB --raw $d/raw ) > $d/run.log 2>&1; rc=$?;;
      wm1q15) ( cd $N/ref-tB-mimic/wm1q15/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
              EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash $AP --board-kit $K $d $RW --arm virtual-nested-pools \
                 --hosted-build $N/wmem-q15/cases-ON --controls-hosted-build $N/wmem-q15/controls-ON --llvm-bin $LB > $d.run.log 2>&1; rc=$?
              mkdir -p $d; cp $d.run.log $d/run.log;;
    esac; fi
    [ -n "$bid" ] && ${REP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
    echo "PASS $tag END $(date +%T) rc=$rc :: $(grep -a '^--- ' $d/run.log | tail -1)" >> "$O"
    python3 $S/compare-pilot.py $ref $d/out > $d.cmp.txt 2>&1; echo "PASS $tag COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
    if [ -n "$B" ] && [ -s "$B" ]; then
      echo "PASS $tag readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) cause=29 $(grep -a -c 'cause=29' $B) banner node-floor-2 x$(grep -a -c node-floor-2 $B)" >> "$O"
    else echo "PASS $tag readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "PASS STOP after $tag: the boot did not complete (no vbatch END rc=0 in its transcript)" >> "$O"; return 1; }
  done
  echo "PASS rep$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
