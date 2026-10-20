#!/bin/bash
# The corpus pass on the rebuilt images (2026-10-20; pre-registered in README.md before its first boot): one appvm arm
# (sqlite engine-repros) on ONE board in --only boots of six cases each, every boot with both controls (the payload cap,
# sq0's split), scored at once by compare-pilot.py against the compiler lane's silicon-mimic reference cut to its cases.
# The runner, the arm script and the collected images are pinned to the reference's own stamp and session: the runner and
# tier3-appvm-arm.sh by sha256/16, the images by the reference's session/collect/images cut to the boot's cases.
# Usage: tier3-pass-appvm.sh <board 1|2> <tag> <ref dir> <runner sha16> <arm sha16> <groups> <out> -- <runner args...>
#   <ref dir> is <mnt-hdd>/capstone/t3-images-cw/ref-tB-mimic/<set> (its out/ and session/collect/images);
#   <groups> is a space-separated list of a..f (cases 00-05, 06-11, ..., 30-32).
# Each boot books the board on the lab hub first and refuses to start on a 409 (someone else holds the window); the
# booking is cancelled when the boot ends. Ends by itself. PASS_ARM / PASS_BUSY / PASS_ROOT / PASS_BOOK stub the dry run.
main() {
  set -u
  local BRD=${1:?board} TAG=${2:?tag} REF=${3:?ref dir} RSHA=${4:?runner sha} ASHA=${5:?arm sha} GRPS=${6:?groups} O=${7:?out}
  shift 7; [ "${1:-}" = -- ] && shift
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone G=/tmp/capstone/wt-gate K L R BUSY bs
  local ARM=$G/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  local RUN=$G/capstone/bug-corpora/sqlite/engine-repros/shared/run-virtual.py
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${PASS_ROOT:-$H/t3-run-treeA/pass-$TAG}; BUSY="python3 $G/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${PASS_ROOT:-$H/t3-run-treeB/pass-$TAG}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PASS STOP: board $BRD" >> "$O"; return 2;;
  esac
  L=$R-lists
  [ "$(sha256sum $RUN | cut -c1-16)" = "$RSHA" ] && [ "$(sha256sum $ARM | cut -c1-16)" = "$ASHA" ] \
    || { echo "PASS STOP: runner or arm script is not the reference's ($RSHA / $ASHA)" >> "$O"; return 1; }
  grep -q "runner .* $RSHA .*appvm-arm $ASHA" $REF.stamp 2>/dev/null || { echo "PASS STOP: $REF.stamp does not name $RSHA and $ASHA" >> "$O"; return 1; }
  grep -q "SWITCHES movc_null_scalar=1 compact_rounding=ON" $REF.stamp || { echo "PASS STOP: $REF is not a silicon-mimic reference" >> "$O"; return 1; }
  [ -s $REF/out/verdicts.jsonl ] && [ -d $REF/session/collect/images ] || { echo "PASS STOP: no reference bundle or collected images" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    || { echo "PASS STOP: kit $K is not the board-$BRD 54d8a3af kit" >> "$O"; return 1; }
  mkdir -p $L
  ( cd $REF/session/collect/images && sha256sum $(ls | sort) ) > $L/all.images.SHA256SUMS
  local g lo hi only t rc bid
  for g in $GRPS; do
    t=$TAG$g
    case $g in a) lo=0;; b) lo=6;; c) lo=12;; d) lo=18;; e) lo=24;; f) lo=30;; *) echo "PASS STOP: group $g" >> "$O"; return 2;; esac
    hi=$(( lo + 5 )); [ $hi -gt 32 ] && hi=32
    only=$(seq -f '%02g' $lo $hi | paste -sd,)
    awk -v lo=$lo -v hi=$hi '{ n=substr($0,67,2); if (n=="co" || (n+0>=lo && n+0<=hi && n ~ /^[0-9][0-9]$/)) print }' \
      $L/all.images.SHA256SUMS > $L/$t.images.SHA256SUMS
    [ "$(cut -c67-68 $L/$t.images.SHA256SUMS | grep -c '^co')" = 2 ] && [ "$(wc -l < $L/$t.images.SHA256SUMS)" = $(( hi - lo + 3 )) ] \
      || { echo "PASS STOP: $t's image list is not its cases plus two controls" >> "$O"; return 1; }
    [ -e "$R$g" ] && { echo "PASS STOP: $R$g exists" >> "$O"; return 1; }
    ${PASS_BUSY:-$BUSY} > /dev/null || { echo "PASS STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${PASS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2-tier12 pass $t (rebuilt images, board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "PASS STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "PASS NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "PASS $t BEGIN $(date +%T) board $BRD only $only booking ${bid:-none}" >> "$O"
    EXPECT_IMAGES=$L/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 ${PASS_ARM:-bash $ARM} --board-kit $K $R$g $RUN \
      "$@" --only "$only" --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 >> "$O" 2>&1; rc=$?
    [ -n "$bid" ] && ${PASS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "PASS $t ARM END rc=$rc $(date +%T)" >> "$O"
    case $rc in 78|2) echo "PASS STOP: the arm script refused $t (rc=$rc)" >> "$O"; return 1;; esac
    python3 $S/subset-bundle.py $REF/out $R$g/out $L/$t.pred > $L/$t.subset.txt 2>&1 \
      && { python3 $S/compare-pilot.py $L/$t.pred $R$g/out > $R$g.cmp.txt 2>&1; echo "PASS $t COMPARE vs reference rc=$? :: $(tail -1 $R$g.cmp.txt)" >> "$O"; } \
      || echo "PASS $t COMPARE NO DATA :: $(tail -1 $L/$t.subset.txt)" >> "$O"
    local GB=$R$g/session/guest/board.txt
    [ -s "$GB" ] || { echo "PASS $t readings: NO DATA, no guest board.txt" >> "$O"; continue; }
    echo "PASS $t readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $GB) DRT1 $(grep -a -c '^DRT1:' $GB) DRTX $(grep -a -c '^DRTX:' $GB) FRTY $(grep -a -c 'FRTY:' $GB) Oops $(grep -a -c Oops $GB) banner node-floor-2 x$(grep -a -c node-floor-2 $GB)" >> "$O"
  done
  echo "PASS $TAG DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
