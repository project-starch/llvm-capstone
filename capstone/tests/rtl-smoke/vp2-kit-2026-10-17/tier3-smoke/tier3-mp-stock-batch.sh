#!/bin/bash
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
  local REF=$H/batch-refs-r60/tick/mruby-stock ST=$H/batch-refs-r60/MPT RL=$H/batch-refs-r60/tick/mruby-stock.log
  local W=$H/wt-ref I=$H/corpus-mp-r60t/cw/mruby-stock ok="control smoke ok"
  local RUN=$W/capstone/bug-corpora/mruby/release-differential/probe/run-virtual.py
  local AP=$W/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MB STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit8-batch-b$BRD
  local d=${MB_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/mp-mruby-stock-tick}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] || { echo "MB STOP: $K is not the batch kit" >> "$O"; return 1; }
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
  bid=$(${MB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2 batch mruby-stock tick (board $BRD)"); rc=$?
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
