#!/bin/bash
# New vp2 corpus arms on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md before the first boot),
# landed on dev with the lead's approval as relayed: xlang (repro + lua-cdp), sqlite/capi-repros and ffmpeg/plane-repros
# (both virtual arms). REVISION 2 (18:5x): the first pins (dev da4cb338b37f) built path-dependent images and were held
# before any boot; the compiler lane's path-independent runners and re-run references are on dev 76054751cd4c, plane on
# fd8c28583642. Runners from the clean dev worktree /tmp/capstone/wt-newarms at bcf5f1724110 (fd8c28583642 plus the
# runtime lane's boardvm step-tail fix: a stalled step's last 240 B ride every @@HB, and its bound is clamped inside the
# boot; the three runners are byte-identical at both commits), each pinned to the
# sha256/16 its QEMU reference records as platform.runner (capi 6981dee6, xlang 6fcf3b75, plane 1de34fdf);
# --virtual-kit replaced by the flashed board's batch kit bkit8-batch-b<board>; SDK t3-freeze-nb-nomovc-cw-r60 (f5145cbb).
# EVERY boot is preceded by a measure-only bake of the same invocation on bkit8-batch-mb<board> (no board), and boots only
# if (1) the firmware is under 26,000,000 B and (2) images-gate.py finds every reference image_sha256 of the boot's cases
# among the images the bake built (a path-dependent build is refused here, not voided after the boot). In order:
#   xlang (one boot, two bundles); plane virtual-malloc, plane virtual-nested-pools (one boot each); capi in --only
#   groups of rows (CAPI_GROUPS below, sized by the bakes measured before the pre-registration), the two generic
#   controls in every boot, each scored against the reference cut to its rows (subset-bundle.py).
# Scored by compare-pilot.py and verdicts-compare.py. Output to NEW directories
# <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/na-<tag>. A boot that does not complete stops the sequence.
# Usage: tier3-newarms-batch.sh <board 1|2> <after|-> <after ERE> <out>.
# NA_RUN / NA_BUSY / NA_BOOK / NA_ROOT / NA_GATE stub the dry run (NA_RUN replaces python3 before the runner path).
CAPI_GROUPS="1,2,3 4,5,6 7,8,9 10,11,12 13,14,15,16 17,18,19"
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} WP=${3:?after ERE} O=${4:?out} S end=$(( $(date +%s) + 72000 ))
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone T=/tmp/capstone/wt-newarms LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=$H/t3-freeze-nb-nomovc-cw-r60/sdk K=$H/t3-smoke/bkit8-batch-b$BRD KM=$H/t3-smoke/bkit8-batch-mb$BRD
  local RR=${NA_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/na} BUSY
  local XR=$T/xlang/virtual/run-virtual.py CR=$T/capstone/bug-corpora/sqlite/capi-repros/shared/run-virtual.py
  local PR=$T/capstone/bug-corpora/tools/run-virtual-cases.py PC=$T/capstone/bug-corpora/ffmpeg/plane-repros
  local XREF=$T/xlang/repro/results/2026-10-22-virtual-run2/virtual-malloc XLUA=$T/xlang/lua-cdp/results/2026-10-22-virtual-run2/virtual-malloc
  local CREF=$T/capstone/bug-corpora/sqlite/capi-repros/results/2026-10-22-virtual-run3/virtual-malloc
  local PM=$PC/results/2026-10-22-batch-kit/virtual-malloc PN=$PC/results/2026-10-22-batch-kit/virtual-nested-pools
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
  [ "$(git -C $T rev-parse --short=12 HEAD)" = bcf5f1724110 ] && [ -z "$(git -C $T status --short --untracked-files=no)" ] \
    || { echo "NA STOP: $T is not a clean bcf5f1724110" >> "$O"; return 1; }
  local k; for k in $K $KM; do
    ( cd $k && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $k/board.env \
      && [ "$(sha256sum $k/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] && ! grep -q norandmaps $k/board.env \
      && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $k/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $k/board.env; }; } \
      || { echo "NA STOP: $k is not board $BRD's batch kit" >> "$O"; return 1; }
  done
  grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "NA STOP: $KM is not the measure-only kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] || { echo "NA STOP: the SDK is not f5145cbb" >> "$O"; return 1; }
  local rr run ref; for rr in "$XR $XREF" "$XR $XLUA" "$CR $CREF" "$PR $PM" "$PR $PN"; do read -r run ref <<<"$rr"
    [ -s $ref/verdicts.jsonl ] && [ "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['platform']['runner'][:16], d['platform']['module'][:16])" $ref/inputs.json)" \
      = "$(sha256sum $run | cut -c1-16) a05c198bfcc5f55c" ] || { echo "NA STOP: $ref does not record runner $(sha256sum $run | cut -c1-16) and module a05c198b" >> "$O"; return 1; }
  done
  echo "NA START $(date +%T) board $BRD tree $(git -C $T rev-parse --short=12 HEAD) capi groups $CAPI_GROUPS" >> "$O"
  local u t d rc bid rows sz B g; local -a argv margv cc
  for u in xlang plane-malloc plane-nested $(for g in $CAPI_GROUPS; do echo capi-${g//,/-}:$g; done); do
    rows=; case $u in capi-*) rows=${u#*:}; u=${u%%:*};; esac; t=$u; d=$RR-$t
    [ -e "$d" ] && { echo "NA STOP: $d exists" >> "$O"; return 1; }
    case $t in
      xlang) argv=($XR --arm virtual-malloc --sdk $SDK --llvm-bin $LB); margv=(--out-repro $d-measure/out-repro --out-lua-cdp $d-measure/out-lua-cdp --raw $d-measure/raw)
             gref="$XREF $XLUA";;
      plane-*) ref=$PM; a=virtual-malloc; [ $t = plane-nested ] && { ref=$PN; a=virtual-nested-pools; }
             mapfile -t cc < <(python3 -c "import json,sys; [print('--cc-arg='+x.replace('<user>', sys.argv[2])) for x in json.load(open(sys.argv[1]))['build']['cc_args']]" $ref/inputs.json "$USER")
             argv=($PR --sdk $SDK --llvm-bin $LB --corpus $PC --arm $a "${cc[@]}"); margv=(--out $d-measure/out --raw $d-measure/raw); gref=$ref;;
      capi-*) argv=($CR --arm virtual-malloc --sdk $SDK --llvm-bin $LB --only $rows); margv=(--out $d-measure/out --raw $d-measure/raw); gref=$CREF;;
    esac
    mkdir -p $d-measure
    ( cd $T && taskset -c 16-31 timeout 3600 ${NA_RUN:-python3} "${argv[@]}" --board-kit $KM "${margv[@]}" ) > $d-measure.run.log 2>&1
    sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" $d-measure/raw/guest/SEQ 2>/dev/null | grep -o "[0-9]\+" | head -1)
    echo "NA $t MEASURE ${rows:+rows $rows }firmware ${sz:-NONE} B against 26000000" >> "$O"
    [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "NA STOP: $t's measured firmware is not under the cap; no boot" >> "$O"; return 1; }
    ${NA_GATE:-python3 $S/images-gate.py} $d-measure/raw/bin $gref ${rows:+--rows $rows} > $d.gate.txt 2>&1; rc=$?
    echo "NA $t IMAGE GATE rc=$rc :: $(tail -1 $d.gate.txt)" >> "$O"
    [ $rc = 0 ] || { echo "NA STOP: $t's built images are not the reference's; no boot" >> "$O"; return 1; }
    ${NA_BUSY:-$BUSY} > /dev/null || { echo "NA STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${NA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 90 "vp2 batch new arm $t (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "NA STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "NA NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    mkdir -p $d; echo "NA $t BEGIN $(date +%T) board $BRD ${rows:+rows $rows }booking ${bid:-none}" >> "$O"
    case $t in xlang) margv=(--out-repro $d/out-repro --out-lua-cdp $d/out-lua-cdp --raw $d/raw);; *) margv=(--out $d/out --raw $d/raw);; esac
    ( cd $T && taskset -c 16-31 timeout 10800 ${NA_RUN:-python3} "${argv[@]}" --board-kit $K "${margv[@]}" ) > $d.run.log 2>&1; rc=$?
    [ -n "$bid" ] && ${NA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "NA $t END $(date +%T) rc=$rc :: $(grep -a '^--- \|^tally\|judged' $d.run.log | tail -1 | cut -c1-200)" >> "$O"
    [ $rc = 78 ] && { echo "NA STOP: the C-81 gate refused the SDK" >> "$O"; return 1; }
    local pairs; case $t in
      xlang) pairs="repro:$XREF:$d/out-repro lua-cdp:$XLUA:$d/out-lua-cdp";;
      plane-*) pairs="$t:$gref:$d/out";;
      capi-*) python3 $S/subset-bundle.py $CREF $d/out $d.pred > $d.subset.txt 2>&1 && pairs="capi:$d.pred:$d/out" \
                || { echo "NA $t COMPARE NO DATA :: $(tail -1 $d.subset.txt)" >> "$O"; pairs=; };;
    esac
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
