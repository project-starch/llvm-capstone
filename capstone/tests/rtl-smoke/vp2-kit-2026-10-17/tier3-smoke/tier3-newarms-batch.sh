#!/bin/bash
# Two new vp2 corpus arms on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md before the first boot),
# both landed on dev with the lead's approval as relayed: xlang (repro + lua-cdp, dev da4cb338b37f) and sqlite/capi-repros
# (dev 75e077151800). Runners from the clean dev worktree /tmp/capstone/wt-newarms, each pinned to the sha256/16 its QEMU
# reference records as platform.runner; --virtual-kit replaced by the flashed board's batch kit bkit8-batch-b<board>; SDK
# t3-freeze-nb-nomovc-cw-r60 (sdk.json f5145cbb, the references'). In order:
#   1. xlang: ONE boot (3.6 MB of images), two bundles, each scored against its reference
#      xlang/{repro,lua-cdp}/results/2026-10-22-virtual/virtual-malloc;
#   2. capi: SIX --only boots (each row image is 5.7 MB, -g, unstripped so the image hashes stay the reference's): rows
#      1-3, 4-6, 7-9, 10-12, 13-16 (15 has no program), 17-19, the two generic controls in every boot. Each boot is preceded
#      by a measure-only bake on bkit8-batch-mb<board> and runs only if the firmware is under 26,000,000 B; each is scored
#      against the reference results/2026-10-22-virtual-run2/virtual-malloc cut to its rows (subset-bundle.py).
# Scored by compare-pilot.py (image, verdict, reason, reached, completed, cause, symbol+offset, attribution, controls) and
# verdicts-compare.py. Output to NEW directories <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/na-<tag>. A boot that
# does not complete stops the sequence. Starts when <after> reads <after ERE> ("-" at once).
# Usage: tier3-newarms-batch.sh <board 1|2> <after|-> <after ERE> <out>.
# NA_RUN / NA_BUSY / NA_BOOK / NA_ROOT stub the dry run (NA_RUN gets <kit> <out dir(s)...> after the real argv's tag).
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} WP=${3:?after ERE} O=${4:?out} S end=$(( $(date +%s) + 72000 ))
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone T=/tmp/capstone/wt-newarms LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=$H/t3-freeze-nb-nomovc-cw-r60/sdk K=$H/t3-smoke/bkit8-batch-b$BRD KM=$H/t3-smoke/bkit8-batch-mb$BRD
  local RR=${NA_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/na} BUSY
  local XR=$T/xlang/virtual/run-virtual.py CR=$T/capstone/bug-corpora/sqlite/capi-repros/shared/run-virtual.py
  local XREF=$T/xlang/repro/results/2026-10-22-virtual/virtual-malloc XLUA=$T/xlang/lua-cdp/results/2026-10-22-virtual/virtual-malloc
  local CREF=$T/capstone/bug-corpora/sqlite/capi-repros/results/2026-10-22-virtual-run2/virtual-malloc
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "NA STOP: board $BRD" >> "$O"; return 2;;
  esac
  if [ "$W" != - ]; then
    until grep -a -q -E "$WP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
    grep -a -q -E "$WP" "$W" || { echo "NA HOLD: $W did not read /$WP/ within 20 h" >> "$O"; return 1; }
    sleep 5
  fi
  git -C $T merge-base --is-ancestor 75e077151800 HEAD && git -C $T merge-base --is-ancestor da4cb338b37f HEAD \
    && [ -z "$(git -C $T status --short --untracked-files=no)" ] || { echo "NA STOP: $T is not a clean dev checkout carrying both arms" >> "$O"; return 1; }
  local k; for k in $K $KM; do
    ( cd $k && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $k/board.env \
      && [ "$(sha256sum $k/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] && ! grep -q norandmaps $k/board.env \
      && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $k/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $k/board.env; }; } \
      || { echo "NA STOP: $k is not board $BRD's batch kit" >> "$O"; return 1; }
  done
  grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "NA STOP: $KM is not the measure-only kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] || { echo "NA STOP: the SDK is not f5145cbb" >> "$O"; return 1; }
  local rr run ref; for rr in "$XR $XREF" "$XR $XLUA" "$CR $CREF"; do read -r run ref <<<"$rr"
    [ -s $ref/verdicts.jsonl ] && [ "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['platform']['runner'][:16], d['build']['sdk_json_sha256'][:16], d['platform']['module'][:16])" $ref/inputs.json)" \
      = "$(sha256sum $run | cut -c1-16) f5145cbb88dde6b5 a05c198bfcc5f55c" ] || { echo "NA STOP: $ref does not record runner $(sha256sum $run | cut -c1-16), SDK f5145cbb and module a05c198b" >> "$O"; return 1; }
  done
  echo "NA START $(date +%T) board $BRD tree $(git -C $T rev-parse --short=12 HEAD)" >> "$O"
  local t d rc bid rows sz B; local -a argv
  # (tag rows): xlang first, then capi's six groups
  for t in xlang capi-a:1,2,3 capi-b:4,5,6 capi-c:7,8,9 capi-d:10,11,12 capi-e:13,14,15,16 capi-f:17,18,19; do
    rows=${t#*:}; t=${t%%:*}; d=$RR-$t
    [ -e "$d" ] && { echo "NA STOP: $d exists" >> "$O"; return 1; }
    if [ $t != xlang ]; then
      mkdir -p $d-measure
      ( cd $T && taskset -c 16-31 timeout 3600 ${NA_RUN:-python3} $CR --arm virtual-malloc --board-kit $KM --sdk $SDK --llvm-bin $LB --only $rows \
          --out $d-measure/out --raw $d-measure/raw ) > $d-measure.run.log 2>&1
      sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" $d-measure/raw/guest/SEQ 2>/dev/null | grep -o "[0-9]\+" | head -1)
      echo "NA $t MEASURE rows $rows: firmware ${sz:-NONE} B against 26000000" >> "$O"
      [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "NA STOP: $t's measured firmware is not under the cap; no boot" >> "$O"; return 1; }
    fi
    ${NA_BUSY:-$BUSY} > /dev/null || { echo "NA STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${NA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 90 "vp2 batch new arm $t (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "NA STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "NA NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    mkdir -p $d; echo "NA $t BEGIN $(date +%T) board $BRD ${rows:+rows $rows }booking ${bid:-none}" >> "$O"
    if [ $t = xlang ]; then argv=($XR --arm virtual-malloc --board-kit $K --sdk $SDK --llvm-bin $LB --out-repro $d/out-repro --out-lua-cdp $d/out-lua-cdp --raw $d/raw)
    else argv=($CR --arm virtual-malloc --board-kit $K --sdk $SDK --llvm-bin $LB --only $rows --out $d/out --raw $d/raw); fi
    ( cd $T && taskset -c 16-31 timeout 10800 ${NA_RUN:-python3} "${argv[@]}" ) > $d.run.log 2>&1; rc=$?
    [ -n "$bid" ] && ${NA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "NA $t END $(date +%T) rc=$rc :: $(grep -a '^--- \|^tally\|judged' $d.run.log | tail -1 | cut -c1-200)" >> "$O"
    [ $rc = 78 ] && { echo "NA STOP: the C-81 gate refused the SDK" >> "$O"; return 1; }
    local pairs; if [ $t = xlang ]; then pairs="repro:$XREF:$d/out-repro lua-cdp:$XLUA:$d/out-lua-cdp"
    else python3 $S/subset-bundle.py $CREF $d/out $d.pred > $d.subset.txt 2>&1 && pairs="capi:$d.pred:$d/out" \
      || { echo "NA $t COMPARE NO DATA :: $(tail -1 $d.subset.txt)" >> "$O"; pairs=; }; fi
    local p n a b; for p in $pairs; do IFS=: read -r n a b <<<"$p"
      python3 $S/compare-pilot.py $a $b > $d.$n.cmp.txt 2>&1; echo "NA $t $n COMPARE vs reference rc=$? :: $(tail -1 $d.$n.cmp.txt)" >> "$O"
      python3 $S/verdicts-compare.py $a $b > $d.$n.vcmp.txt 2>&1; echo "NA $t $n VCOMPARE rc=$? :: $(tail -1 $d.$n.vcmp.txt)" >> "$O"
    done
    B=$d/raw/guest/board.txt
    if [ -s "$B" ]; then
      echo "NA $t readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) collect-skipped $(grep -a -c 'collect skipped' $B) cause=29 $(grep -a -c 'domain fault cause=29' $B)" >> "$O"
    else echo "NA $t readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -s "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "NA STOP after $t: the boot did not complete (no vbatch END rc=0 in its transcript)" >> "$O"; return 1; }
  done
  echo "NA seq-$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
