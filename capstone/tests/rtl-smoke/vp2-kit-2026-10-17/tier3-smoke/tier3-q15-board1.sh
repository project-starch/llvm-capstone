#!/bin/bash
# sqlite/engine-repros vp2-nested-pools with the memsys5 Q-15 repair (dev f1d522d7fff1) on BOARD 1, tree A, in six --only
# boots (sq0's split: 00-05, 06-11, 12-17, 18-23, 24-29, 30-32, each with both memsys5 controls; the payload cap).
# Pre-registered in README.md before the first boot. The compiler lane's invocation: run-virtual.py --arm
# virtual-nested-pools on build-virtual-nested-pools-q15, through dev's tier3-appvm-arm.sh (the C-81 gate) from the
# dev-pinned worktree /tmp/capstone/wt-q15, kit bkit7-rev2xD (tree A, module 39d8492d, monitor f13dbc5). Each subset's
# EXPECT_IMAGES is m5q.images.SHA256SUMS (eab23e459a1d0972) cut to its cases plus the controls; each boot is scored at once
# by compare-pilot.py against run C (results/2026-10-20-q15-repair/run-C) cut to its cases (subset-bundle.py).
# Waits for <after> to read CPP DONE, STOP or HOLD. Usage: tier3-q15-board1.sh <after> <out>. Ends by itself.
# Q15_ARM / Q15_BUSY / Q15_ROOT / Q15_SUBSETS stub for the dry run.
main() {
  set -u
  local W=${1:?after} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone
  local K=$H/t3-smoke/bkit7-rev2xD R=${Q15_ROOT:-$H/t3-run/q15} I=$H/t3-images-nb L=$H/t3-q15sq
  local Q=/tmp/capstone/wt-q15/capstone
  local RUN=$Q/bug-corpora/sqlite/engine-repros/shared/run-virtual.py ARM=$Q/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  local C=$Q/bug-corpora/sqlite/engine-repros/results/2026-10-20-q15-repair/run-C
  local B=$I/sqlite/build-virtual-nested-pools-q15 g lo hi only t rc
  until grep -a -q "^CPP DONE\|^CPP STOP\|^CPP HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CPP DONE\|^CPP STOP\|^CPP HOLD" "$W" || { echo "Q15 HOLD: board 1's cpython pair did not end" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-q15 rev-parse --short=12 HEAD)" = f1d522d7fff1 ] && [ -z "$(git -C /tmp/capstone/wt-q15 status --porcelain)" ] \
    || { echo "Q15 STOP: wt-q15 is not a clean f1d522d7fff1" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = b3f1aa0ea60442f8 ] && [ "$(sha256sum $ARM | cut -c1-16)" = 666d920eedef445b ] \
    || { echo "Q15 STOP: runner or arm script" >> "$O"; return 1; }
  [ "$(sha256sum $I/m5q.images.SHA256SUMS | cut -c1-16)" = eab23e459a1d0972 ] || { echo "Q15 STOP: image list" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env \
    && ! grep -q "^FPGA_BOARD=2" $K/board.env || { echo "Q15 STOP: kit is not bkit7-rev2xD as recorded" >> "$O"; return 1; }
  mkdir -p $L
  for g in ${Q15_SUBSETS:-a b c d e f}; do
    t=q15$g
    case $g in a) lo=0;; b) lo=6;; c) lo=12;; d) lo=18;; e) lo=24;; f) lo=30;; esac; hi=$(( lo + 5 )); [ $hi -gt 32 ] && hi=32
    only=$(seq -f '%02g' $lo $hi | paste -sd,)
    awk -v lo=$lo -v hi=$hi '{ n=substr($0,67,2); if (n=="co" || (n+0>=lo && n+0<=hi && n ~ /^[0-9][0-9]$/)) print }' \
      $I/m5q.images.SHA256SUMS > $L/$t.images.SHA256SUMS
    [ "$(cut -c67-68 $L/$t.images.SHA256SUMS | grep -c '^co')" = 2 ] && [ "$(wc -l < $L/$t.images.SHA256SUMS)" = $(( hi - lo + 3 )) ] \
      || { echo "Q15 STOP: $t's image list is not its cases plus two controls" >> "$O"; return 1; }
    [ -e "$R$g" ] && { echo "Q15 STOP: $R$g exists" >> "$O"; return 1; }
    ${Q15_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "Q15 STOP: board 1 is busy" >> "$O"; return 1; }
    echo "Q15 $t BEGIN $(date +%T) only $only" >> "$O"
    EXPECT_IMAGES=$L/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 ${Q15_ARM:-bash $ARM} --board-kit $K $R$g $RUN \
      --arm virtual-nested-pools --build $B --only "$only" --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 >> "$O" 2>&1; rc=$?
    echo "Q15 $t ARM END rc=$rc $(date +%T)" >> "$O"
    # the arm script's own refusals (78 the C-81 gate, 2 usage or an existing root) stop the queue; a boot's result does not
    case $rc in 78|2) echo "Q15 STOP: the arm script refused $t (rc=$rc)" >> "$O"; return 1;; esac
    python3 $S/subset-bundle.py $C $R$g/out $L/$t.pred > $L/$t.subset.txt 2>&1 \
      && { python3 $S/compare-pilot.py $L/$t.pred $R$g/out > $R$g.cmp.txt 2>&1; echo "Q15 $t COMPARE vs run C rc=$? :: $(tail -1 $R$g.cmp.txt)" >> "$O"; } \
      || echo "Q15 $t COMPARE NO DATA :: $(tail -1 $L/$t.subset.txt)" >> "$O"
    local G=$R$g/session/guest/board.txt
    [ -s "$G" ] || { echo "Q15 $t readings: NO DATA, no guest board.txt" >> "$O"; continue; }
    echo "Q15 $t readings: Q15_PAD $(grep -a -c 'Q15_PAD' $G 2>/dev/null); cause=29 $(grep -a -c 'cause=29' $G 2>/dev/null); DRT1 $(grep -a -c '^DRT1:' $G 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $G 2>/dev/null) FRTY $(grep -a -c 'FRTY:' $G 2>/dev/null) Oops $(grep -a -c Oops $G 2>/dev/null)" >> "$O"
  done
  echo "Q15 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
