#!/bin/bash
# mruby Sublet (nested pools) with smoke-fit.rb on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md
# before its boot). tier3-mp-arm-t100.sh's mruby-sublet leg with what the batch changes:
#   - the kit is the flashed board's bkit8-batch-b<board> (bitstream c721c1a2a, batch module 21fbfc39, monitor d9da397);
#   - the reference is the compiler lane's smoke-fit/sublet run on qkit7-batch2 (module a05c198b, the same source as
#     21fbfc39 built for QEMU), whose stamp reads movc_null_scalar=OFF: on c721c1a2a MOVC keeps an untagged source (batch
#     item 5), so OFF is the silicon mimic there and qkit7-batch2 refuses ON;
#   - --smoke smoke-fit.rb (5b588c1c), the reference's one added variable;
#   - the runner and arm script from <mnt-hdd>/capstone/wt-ref (clean 6764d96229f6), pinned by the stamp's hashes.
# Output to a NEW directory named for the bitstream; scored by matrix-compare.py against the reference's out/matrix.tsv.
# Usage: tier3-mp-batch.sh <board 1|2> <out log>. MB_ARM / MB_BUSY / MB_BOOK / MB_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone LB=/tmp/capstone/vp2-prep/llvm-bin-v6 K BUSY
  local REF=$H/smoke-fit/sublet ST=$H/smoke-fit/sublet.stamp RL=$H/smoke-fit/sublet.log SM=$H/smoke-fit/smoke-fit.rb
  local W=$H/wt-ref I=$H/corpus-mp-r60t/cw/mruby-sublet ok="control smoke ok"
  local RUN=$W/capstone/bug-corpora/mruby/release-differential/probe/run-virtual.py
  local AP=$W/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MB STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit8-batch-b$BRD
  local d=${MB_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/mp-mruby-sublet-smokefit}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] || { echo "MB STOP: $K is not the batch kit" >> "$O"; return 1; }
  grep -q "SWITCHES movc_null_scalar=OFF compact_rounding=ON" $ST && grep -q "generations=ON" $ST || { echo "MB STOP: the reference stamp is not the batch mimic" >> "$O"; return 1; }
  grep -qF "$ok" <(grep "^APPVM read rc=0 " $RL) || { echo "MB STOP: the reference did not read clean" >> "$O"; return 1; }
  [ -s $REF/out/matrix.tsv ] && [ -d $REF/session/collect/images ] || { echo "MB STOP: no reference bundle" >> "$O"; return 1; }
  [ "$(git -C $W rev-parse --short=12 HEAD)" = 6764d96229f6 ] && [ -z "$(git -C $W status --porcelain)" ] || { echo "MB STOP: $W is not a clean 6764d96229f6" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = 9db9f5f7cf60fcda ] && grep -q "runner 9db9f5f7cf60fcda" $ST \
    && [ "$(sha256sum $AP | cut -c1-16)" = 11967b378bda73c8 ] && grep -q "appvm-arm 11967b378bda73c8" $ST \
    && [ "$(sha256sum $SM | cut -c1-16)" = 5b588c1c35261904 ] && grep -q "smoke 5b588c1c35261904" $ST \
    && [ "$(sha256sum $I/src/mruby/build/capstone/bin/mruby | cut -c1-16)" = 0f9bca66063f083f ] && grep -q "image 0f9bca66063f083f" $ST \
    || { echo "MB STOP: the runner, arm script, smoke or image is not the reference's" >> "$O"; return 1; }
  [ -e "$d" ] && { echo "MB STOP: $d exists" >> "$O"; return 1; }
  mkdir -p "$(dirname $d)"
  ${MB_BUSY:-$BUSY} > /dev/null || { echo "MB STOP: board $BRD is busy" >> "$O"; return 1; }
  local bid rc
  bid=$(${MB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2 batch mruby-sublet smoke-fit (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "MB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "MB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  ( cd $REF/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
  echo "MB BEGIN $(date +%T) board $BRD kit ${K##*/} booking ${bid:-none}" >> "$O"
  EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 18000 ${MB_ARM:-bash $AP} --board-kit $K $d $RUN --arm virtual-nested-pools \
    --image $I/src/mruby/build/capstone/bin/mruby --capi-dir $I/capi --smoke $SM --llvm-bin $LB > $d.run.log 2>&1; rc=$?
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
