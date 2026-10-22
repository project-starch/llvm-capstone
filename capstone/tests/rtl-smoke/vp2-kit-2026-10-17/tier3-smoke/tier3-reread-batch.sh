#!/bin/bash
# The corpus pass RE-READ on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md before its first boot):
# a SEQUENCE of arms on one board, one cold boot per arm (one per group of six for the sqlite arms), each run with the
# compiler lane's batch-reference invocation (<mnt-hdd>/capstone/batch-refs/run-pass-r60.sh ebd2406b3351c381) with
# --virtual-kit replaced by --board-kit. tier3-pass-seq2.sh's shape, with what the batch changes:
#   - the kit is the flashed board's bkit8-batch-b<board> (c721c1a2a, batch module 21fbfc39, monitor d9da397);
#   - the images are the r60 rebuilds (t3-images-r60, SDK t3-freeze-nb-nomovc-cw-r60 sdk.json f5145cbb), the reference's;
#   - the reference is batch-refs-r60/ref-batch/<tag> (out/ or bundle/), whose stamp must name the runner's sha256/16 (and
#     the arm script's, for appvm kinds) and read SWITCHES movc_null_scalar=OFF ... generations=ON(compiled);
#   - scored by compare-pilot.py (image, verdict, reason, reached, completed, cause, symbol+offset, attribution, controls),
#     the standing pass's tool, and by verdicts-compare.py (the batch reference's own five-field compare);
#   - output to a NEW directory named for the bitstream, <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/rr-<tag>.
# Each spec line is  tag|kind|tree|tmproot|runner args  (tree "-" = /tmp/capstone/wt-gate; tmproot "-" = unset), kind:
#   cases  tools/run-virtual-cases.py --board-kit K --sdk SDK --llvm-bin LB <args> --out --raw
#   pgc    postgres/c-repros/shared/run-arm.py --board-kit K --llvm-bin LB <args> --out --raw
#   pgm    postgres/mmgr-repros/shared/run-defects.py <out> --board-kit K --llvm-bin LB <args> --raw
#   appvm:<runner rel>   tier3-appvm-arm.sh (from the same tree) --board-kit K <root> <runner> <args> --llvm-bin LB,
#          EXPECT_IMAGES = the reference's session/collect/images
#   appvm6:<runner rel>  as appvm, in --only boots of six cases (groups a-f: 00-05 ... 30-32) plus both controls, each
#          scored against the reference cut to its cases (subset-bundle.py), tier3-pass-appvm.sh's split
# A boot that does not complete stops the sequence (the first failure is the reading). Starts when <after> reads one of
# <after pattern> (an ERE; "-" starts at once) and holds on anything else after a 12 h bound.
# Usage: tier3-reread-batch.sh <board 1|2> <after|-> <after ERE> <spec file> <out>.
# RR_RUN / RR_BUSY / RR_ROOT / RR_BOOK stub the dry run (RR_RUN replaces the runner call and is given kind, ref, out dir).
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} WP=${3:?after ERE} SPEC=${4:?spec} O=${5:?out} S end=$(( $(date +%s) + 43200 ))
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone G=/tmp/capstone/wt-gate LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb-nomovc-cw-r60/sdk RB=/mnt/hdd/$USER/capstone/batch-refs-r60/ref-batch
  local K=$H/t3-smoke/bkit8-batch-b$BRD RR=${RR_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/rr} BUSY
  case $BRD in
    1) BUSY="python3 $G/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "RR STOP: board $BRD" >> "$O"; return 2;;
  esac
  if [ "$W" != - ]; then
    until grep -a -q -E "$WP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
    grep -a -q -E "$WP" "$W" || { echo "RR HOLD: $W did not read /$WP/ within 12 h ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
    sleep 5
  fi
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    && ! grep -q norandmaps $K/board.env || { echo "RR STOP: kit $K is not the board-$BRD batch kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] || { echo "RR STOP: the SDK is not f5145cbb" >> "$O"; return 1; }
  echo "RR START $(date +%T) board $BRD spec $(sha256sum $SPEC | cut -c1-16) ($(grep -c "|" $SPEC) arms)" >> "$O"
  local line tag kind tree tmp args rel run ap ref rb rc bid d g lo hi only t B; local -a xa
  while IFS= read -r line; do
    [ -z "$line" ] && continue; case $line in \#*) continue;; esac
    IFS='|' read -r tag kind tree tmp args <<<"$line"; read -r -a xa <<<"${args:-}"
    [ "$tree" = - ] && tree=$G; [ "$tmp" = - ] && tmp=
    ref=$RB/$tag; d=$RR-$tag
    if [ -s $ref/out/verdicts.jsonl ]; then rb=$ref/out; elif [ -s $ref/bundle/verdicts.jsonl ]; then rb=$ref/bundle
    else echo "RR STOP: $tag has no batch reference verdicts" >> "$O"; return 1; fi
    case $kind in
      cases) rel=capstone/bug-corpora/tools/run-virtual-cases.py;; pgc) rel=capstone/bug-corpora/postgres/c-repros/shared/run-arm.py;;
      pgm) rel=capstone/bug-corpora/postgres/mmgr-repros/shared/run-defects.py;; appvm:*|appvm6:*) rel=${kind#*:};;
      *) echo "RR STOP: $tag kind $kind" >> "$O"; return 2;;
    esac
    run=$tree/$rel; ap=$tree/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
    [ -z "$(git -C $tree status --short --untracked-files=no 2>/dev/null)" ] && git -C $tree rev-parse -q --verify HEAD > /dev/null \
      || { echo "RR STOP: $tag tree $tree is not a clean checkout" >> "$O"; return 1; }
    grep -q "^STAMP $tag runner $(sha256sum $run | cut -c1-16) " $ref.stamp 2>/dev/null || { echo "RR STOP: $tag runner $rel is not its stamp's" >> "$O"; return 1; }
    grep -q "SWITCHES movc_null_scalar=OFF compact_rounding=ON.* generations=ON" $ref.stamp || { echo "RR STOP: $tag's reference is not the batch mimic" >> "$O"; return 1; }
    case $kind in appvm*) grep -q "appvm-arm $(sha256sum $ap | cut -c1-16)" $ref.stamp && [ -d $ref/session/collect/images ] \
      || { echo "RR STOP: $tag arm script is not its stamp's, or no collected images" >> "$O"; return 1; };; esac
    case $kind in cases) grep -q "sdk.json" $ref.stamp && ! grep -q "sdk.json f5145cbb88dde6b5" $ref.stamp \
      && { echo "RR STOP: $tag's stamp names another SDK" >> "$O"; return 1; };; esac
    for g in $([ "${kind%%:*}" = appvm6 ] && echo a b c d e f || echo -); do
      t=$tag; [ $g = - ] || t=$tag$g; d=$RR-$t
      [ -e "$d" ] && { echo "RR STOP: $d exists" >> "$O"; return 1; }
      ${RR_BUSY:-$BUSY} > /dev/null || { echo "RR STOP: board $BRD is busy" >> "$O"; return 1; }
      bid=$(${RR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 90 "vp2 batch re-read $t (board $BRD)"); rc=$?
      [ $rc = 3 ] && { echo "RR STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
      [ $rc = 0 ] || { echo "RR NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
      only=; local -a oa=()
      if [ $g != - ]; then
        case $g in a) lo=0;; b) lo=6;; c) lo=12;; d) lo=18;; e) lo=24;; f) lo=30;; esac
        hi=$(( lo + 5 )); [ $hi -gt 32 ] && hi=32; only=$(seq -f '%02g' $lo $hi | paste -sd,); oa=(--only "$only")
      fi
      case $kind in appvm*) mkdir -p "$(dirname $d)"
        if [ $g = - ]; then ( cd $ref/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
        else ( cd $ref/session/collect/images && sha256sum $(ls | sort) ) \
               | awk -v lo=$lo -v hi=$hi '{ n=substr($0,67,2); if (n=="co" || (n+0>=lo && n+0<=hi && n ~ /^[0-9][0-9]$/)) print }' > $d.images.SHA256SUMS
             [ "$(cut -c67-68 $d.images.SHA256SUMS | grep -c '^co')" = 2 ] && [ "$(wc -l < $d.images.SHA256SUMS)" = $(( hi - lo + 3 )) ] \
               || { echo "RR STOP: $t's image list is not its cases plus two controls" >> "$O"; [ -n "$bid" ] && ${RR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1; return 1; }
        fi;;
      esac
      echo "RR $t BEGIN $(date +%T) board $BRD kind $kind tree $(git -C $tree rev-parse --short=12 HEAD) ${only:+only $only }booking ${bid:-none}" >> "$O"
      if [ -n "${RR_RUN:-}" ]; then mkdir -p $d; $RR_RUN $kind $rb $d > $d.run.log 2>&1; rc=$?
      else case $kind in
        cases) mkdir -p $d; ( cd $tree && env ${tmp:+CAPSTONE_TMP_ROOT=$tmp} taskset -c 16-31 timeout 10800 python3 $run --board-kit $K --sdk $SDK --llvm-bin $LB "${xa[@]}" --out $d/out --raw $d/raw ) > $d.run.log 2>&1; rc=$?;;
        pgc)   mkdir -p $d; ( cd $tree && env ${tmp:+CAPSTONE_TMP_ROOT=$tmp} taskset -c 16-31 timeout 10800 python3 $run --board-kit $K --llvm-bin $LB "${xa[@]}" --out $d/out --raw $d/raw ) > $d.run.log 2>&1; rc=$?;;
        pgm)   mkdir -p $d; ( cd $tree && env ${tmp:+CAPSTONE_TMP_ROOT=$tmp} taskset -c 16-31 timeout 10800 python3 $run $d/out --board-kit $K --llvm-bin $LB "${xa[@]}" --raw $d/raw ) > $d.run.log 2>&1; rc=$?;;
        appvm*) ( cd $tree && env ${tmp:+CAPSTONE_TMP_ROOT=$tmp} EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash $ap --board-kit $K $d $run "${xa[@]}" "${oa[@]}" --llvm-bin $LB ) > $d.run.log 2>&1; rc=$?;;
      esac; fi
      [ -n "$bid" ] && ${RR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
      echo "RR $t END $(date +%T) rc=$rc :: $(grep -a '^--- \|^APPVM read' $d.run.log | tail -1 | cut -c1-200)" >> "$O"
      case $kind in appvm*) case $rc in 78|2) echo "RR STOP: the arm script refused $t (rc=$rc)" >> "$O"; return 1;; esac;; esac
      local cref=$rb
      if [ $g != - ]; then
        python3 $S/subset-bundle.py $rb $d/out $d.pred > $d.subset.txt 2>&1 || { echo "RR $t COMPARE NO DATA :: $(tail -1 $d.subset.txt)" >> "$O"; cref=; }
        [ -n "$cref" ] && cref=$d.pred
      fi
      if [ -n "$cref" ]; then
        python3 $S/compare-pilot.py $cref $d/out > $d.cmp.txt 2>&1; echo "RR $t COMPARE vs batch reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
        python3 $S/verdicts-compare.py $cref $d/out > $d.vcmp.txt 2>&1; echo "RR $t VCOMPARE rc=$? :: $(tail -1 $d.vcmp.txt)" >> "$O"
      fi
      B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
      if [ -n "$B" ] && [ -s "$B" ]; then
        echo "RR $t readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) collect-skipped $(grep -a -c 'collect skipped' $B) cause=29 $(grep -a -c 'domain fault cause=29' $B)" >> "$O"
      else echo "RR $t readings: NO DATA, no guest board.txt" >> "$O"; fi
      [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "RR STOP after $t: the boot did not complete (no vbatch END rc=0 in its transcript)" >> "$O"; return 1; }
    done
  done < "$SPEC"
  echo "RR seq-$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
