#!/bin/bash
# The mruby and perl release-differential arms on one board (2026-10-21; each pre-registered in README.md before its
# boot): one cold boot per arm, through tier3-appvm-arm.sh, pinned to the arm's QEMU reference: the runner and the arm
# script by the sha256/16 its stamp names, the tree by the stamp's wt-ref commit (worktree /tmp/capstone/wt-mp at that
# commit), the images by the reference's session/collect/images (EXPECT_IMAGES). Scored by matrix-compare.py against
# the reference's out/matrix.tsv. Completion is the driver's vbatch END line. Booked on the lab hub; refused on a 409.
# Usage: tier3-mp-arm.sh <board 1|2> <out log> <tag>... with tags mruby-stock mruby-sublet (perl's once its references
# land). A boot that does not complete stops the run. MP_ARM / MP_BUSY / MP_BOOK / MP_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}; shift 2
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone C=/mnt/hdd/$USER/capstone/corpus-mp W=/tmp/capstone/wt-mp LB=/tmp/capstone/vp2-prep/llvm-bin-v6 K bs RR BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; RR=${MP_ROOT:-$H/t3-run-treeA/mp}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; RR=${MP_ROOT:-$H/t3-run-treeB/mp}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MP STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "MP STOP: kit $K" >> "$O"; return 1; }
  local AP=$W/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh tag ref st run args wref rsha asha d rc bid B
  for tag in "$@"; do
    ref=$C/refs/tB-mimic/$tag; st=$ref.stamp
    case $tag in
      mruby-stock|mruby-sublet)
        run=$W/capstone/bug-corpora/mruby/release-differential/probe/run-virtual.py
        [ $tag = mruby-stock ] && args="--arm virtual-malloc" || args="--arm virtual-nested-pools"
        args="$args --image $C/cw/$tag/src/mruby/build/capstone/bin/mruby --capi-dir $C/cw/$tag/capi";;
      *) echo "MP STOP: tag $tag has no recipe" >> "$O"; return 2;;
    esac
    [ -s $ref/out/matrix.tsv ] && [ -d $ref/session/collect/images ] || { echo "MP STOP: $tag has no reference bundle or collected images" >> "$O"; return 1; }
    grep -q "SWITCHES movc_null_scalar=1 compact_rounding=ON" $st || { echo "MP STOP: $tag's reference is not silicon-mimic" >> "$O"; return 1; }
    wref=$(grep -o "wt-ref [0-9a-f]\{12\} dirty 0" $st | head -1 | cut -d' ' -f2); rsha=$(grep -o "runner [0-9a-f]\{16\}" $st | head -1 | cut -d' ' -f2); asha=$(grep -o "appvm-arm [0-9a-f]\{16\}" $st | head -1 | cut -d' ' -f2)
    [ -n "$wref" ] && [ "$(git -C $W rev-parse --short=12 HEAD)" = "$wref" ] && [ -z "$(git -C $W status --porcelain)" ] || { echo "MP STOP: $W is not the clean wt-ref commit $tag's stamp names (${wref:-none})" >> "$O"; return 1; }
    [ "$(sha256sum $run | cut -c1-16)" = "$rsha" ] && [ "$(sha256sum $AP | cut -c1-16)" = "$asha" ] || { echo "MP STOP: the runner or arm script is not $tag's reference's ($rsha / $asha)" >> "$O"; return 1; }
    d=$RR-$tag; [ -e "$d" ] && { echo "MP STOP: $d exists" >> "$O"; return 1; }
    ${MP_BUSY:-$BUSY} > /dev/null || { echo "MP STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${MP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2-tier12 $tag (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "MP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "MP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    ( cd $ref/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
    echo "MP $tag BEGIN $(date +%T) board $BRD runner $rsha appvm-arm $asha wt-ref $wref booking ${bid:-none}" >> "$O"
    # shellcheck disable=SC2086
    EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 18000 ${MP_ARM:-bash $AP} --board-kit $K $d $run $args --llvm-bin $LB > $d.run.log 2>&1; rc=$?
    [ -n "$bid" ] && ${MP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "MP $tag END $(date +%T) rc=$rc :: $(tail -1 $d.run.log | cut -c1-160)" >> "$O"
    case $rc in 78|2) echo "MP STOP: the arm script refused $tag (rc=$rc)" >> "$O"; return 1;; esac
    python3 $S/matrix-compare.py $ref/out $d/out > $d.cmp.txt 2>&1; echo "MP $tag COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
    B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
    if [ -n "$B" ] && [ -s "$B" ]; then
      echo "MP $tag readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) banner node-floor-2 x$(grep -a -c node-floor-2 $B)" >> "$O"
    else echo "MP $tag readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "MP STOP after $tag: the boot did not complete (no vbatch END rc=0)" >> "$O"; return 1; }
  done
  echo "MP board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
