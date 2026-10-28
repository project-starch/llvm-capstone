#!/bin/bash
# BITSTREAM 3 version (2026-10-28; pre-registered in README.md before its first boot): tier3-mp-stock-b2.sh with only the board side
# changed: the bitstream-3 kit bkit11-b3-b<board> (make-b3-kit.sh copies of the bitstream-2 kits, one board.env line,
# BOARD_BITSTREAM=caplifive_vp2c_e7af48457, module 215a09be unchanged), output <mnt-hdd>/capstone/t3-run-b3-e7af48457-
# b<board>/, and "bitstream 3" in the booking text. The references (the bitstream-2 twin's), images, runners, trees, SDK
# and grade are bitstream 2's, unchanged. The bitstream-2 version's header follows.
# BITSTREAM 2 version (2026-10-25; pre-registered in README.md before its boot): tier3-mp-stock-batch.sh on the board's bitstream-2 kit
# bkit9-b2-b<board> (caplifive_vp2c_dcc6daf6e, module 215a09be), scored against the compiler lane's bitstream-2 reference
# batch-refs-b2/rest/mruby-stock (qkit7-b2twin3-q8). The batch reference's stamp still pins the runner, arm script, tree and
# image (the bitstream-2 run used the same inputs); batch-refs-b2/REST-A must name the twin (BEGIN ... qemu 708bbff5
# module 33254735), carry this arm's STAMP line with the same runner and image, a clean read and "differ 0", and the
# reference's collected images must equal the batch reference's. Output <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/.
# The batch version's header follows.
# mruby STOCK (virtual-malloc) on the BATCH bitstream c721c1a2a (2026-10-23; pre-registered in README.md before its boot):
# tier3-mp-batch.sh (mruby Sublet with smoke-fit) with the stock arm's inputs, the compiler lane's tick reference
# batch-refs-r60/tick/mruby-stock (qkit7-batch2; MRB_TICK_UNIT=100 images, corpus-mp-r60t, the S-34 workaround; stamp in
# batch-refs-r60/MPT: image 1ab24a92, runner 9db9f5f7, appvm-arm 11967b37, tree 6764d96229f6; read "control smoke ok",
# fault 20 completed 3; MOVE 0 against the r60 4 ms reference), the stock smoke (no --smoke), --arm virtual-malloc.
# Usage: tier3-mp-stock-batch.sh <board 1|2> <out log>. MB_ARM / MB_BUSY / MB_BOOK / MB_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone LB=/tmp/capstone/vp2-prep/llvm-bin-v6 K BUSY
  local BREF=$H/batch-refs-r60/tick/mruby-stock REF=$H/batch-refs-b2/rest/mruby-stock ST=$H/batch-refs-r60/MPT RL=$H/batch-refs-r60/tick/mruby-stock.log
  local W=$H/wt-ref I=$H/corpus-mp-r60t/cw/mruby-stock ok="control smoke ok"
  local RUN=$W/capstone/bug-corpora/mruby/release-differential/probe/run-virtual.py
  local AP=$W/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MB STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit11-b3-b$BRD
  local d=${MB_ROOT:-$H/t3-run-b3-e7af48457-b$BRD/mp-mruby-stock-tick}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_e7af48457" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] || { echo "MB STOP: $K is not the bitstream-3 kit" >> "$O"; return 1; }
  # Bitstream-2 reference gate (see the header).
  local RA=$H/batch-refs-b2/REST-A
  grep -q "^BEGIN A .* qemu 708bbff5663f0e86 module 3325473537072a3a " $RA && grep -q "^STAMP mruby-stock .*runner 9db9f5f7cf60fcda .*image 1ab24a9215078a53" $RA \
    && grep -q "^mruby-stock rc=0 .*APPVM read rc=0 .*control smoke ok" $RA && grep -q "mruby-stock batch(tick) -> b2: cells 23/23.* differ 0$" $RA \
    && diff -q <(cd $BREF/session/collect/images && sha256sum *) <(cd $REF/session/collect/images && sha256sum *) > /dev/null \
    || { echo "MB STOP: the bitstream-2 reference mruby-stock is not the twin's clean run of the batch inputs" >> "$O"; return 1; }
  grep -q "^STAMP mruby-stock tick image 1ab24a9215078a53 .*runner 9db9f5f7cf60fcda appvm-arm 11967b378bda73c8 .*tree 6764d96229f6 dirty 0 qemu ecaf50bbd5ec44aa" $ST \
    || { echo "MB STOP: the tick reference's stamp" >> "$O"; return 1; }
  grep -qF "$ok" <(grep "^APPVM read rc=0 " $RL) || { echo "MB STOP: the reference did not read clean" >> "$O"; return 1; }
  [ -s $REF/out/matrix.tsv ] && [ -d $REF/session/collect/images ] || { echo "MB STOP: no reference bundle" >> "$O"; return 1; }
  [ "$(git -C $W rev-parse --short=12 HEAD)" = 6764d96229f6 ] && [ -z "$(git -C $W status --porcelain)" ] || { echo "MB STOP: $W is not a clean 6764d96229f6" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = 9db9f5f7cf60fcda ] && [ "$(sha256sum $AP | cut -c1-16)" = 11967b378bda73c8 ] \
    && [ "$(sha256sum $I/src/mruby/build/capstone/bin/mruby | cut -c1-16)" = 1ab24a9215078a53 ] \
    && [ "$(sha256sum $I/capi/capi-11.dom | cut -c1-16)" = dee93c7d71172676 ] \
    || { echo "MB STOP: the runner, arm script, image or capi-11 is not the reference's" >> "$O"; return 1; }
  [ -e "$d" ] && { echo "MB STOP: $d exists" >> "$O"; return 1; }
  mkdir -p "$(dirname $d)"
  ${MB_BUSY:-$BUSY} > /dev/null || { echo "MB STOP: board $BRD is busy" >> "$O"; return 1; }
  local bid rc
  bid=$(${MB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2 bitstream 3 mruby-stock tick (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "MB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "MB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  ( cd $REF/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
  echo "MB BEGIN $(date +%T) board $BRD kit ${K##*/} booking ${bid:-none}" >> "$O"
  EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 18000 ${MB_ARM:-bash $AP} --board-kit $K $d $RUN --arm virtual-malloc \
    --image $I/src/mruby/build/capstone/bin/mruby --capi-dir $I/capi --llvm-bin $LB > $d.run.log 2>&1; rc=$?
  [ -n "$bid" ] && ${MB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "MB END $(date +%T) rc=$rc :: $(tail -1 $d.run.log | cut -c1-160)" >> "$O"
  case $rc in 78|2) echo "MB STOP: the arm script refused (rc=$rc)" >> "$O"; return 1;; esac
  if grep -qF "$ok" <(grep "^APPVM read rc=0 " $d.run.log); then
    python3 $S/matrix-compare.py $REF/out $d/out > $d.cmp.txt 2>&1; echo "MB COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
  else
    echo "MB VOID: the board run's own controls did not pass :: $(grep '^APPVM read' $d.run.log | tail -1 | cut -c1-140)" >> "$O"
  fi
  local B; B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
  if [ -n "$B" ] && [ -s "$B" ]; then
    echo "MB readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) collect-skipped $(grep -a -c 'collect skipped' $B) banner gen-push x$(grep -a -c 'diag gen-push' $B)" >> "$O"
  else echo "MB readings: NO DATA, no guest board.txt" >> "$O"; fi
  [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "MB STOP: the boot did not complete (no vbatch END rc=0)" >> "$O"; return 1; }
  echo "MB board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
