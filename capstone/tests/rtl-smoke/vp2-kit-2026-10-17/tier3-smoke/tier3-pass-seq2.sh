#!/bin/bash
# tier3-pass-seq.sh with ONE change (2026-10-20 19:1x): a boot's completion is read from the driver's own transcript, the
# line "### RUNG <n> vbatch END rc=0 ###" in the guest board.txt, instead of the runner's "virtual boot: completed", which
# run-defects.py (pgm) never prints, so tier3-pass-seq.sh stopped falsely after pgm0, whose boot had completed. Controls: the
# line is present in every completed pass boot of each runner kind and absent from a panicked (sql0-treeA) and a wedged
# (wm0, 2026-10-18) boot.
# The corpus pass on the rebuilt images: a SEQUENCE of arms on one board, one cold boot per arm (2026-10-20; pre-registered
# in README.md before its first boot). Each spec line of <spec file> is  tag|kind|runner args (single spaces)  with kind
#   cases  tools/run-virtual-cases.py --board-kit K --sdk SDK --llvm-bin LB <args> --out --raw
#   pgc    postgres/c-repros/shared/run-arm.py --board-kit K --llvm-bin LB <args> --out --raw
#   pgm    postgres/mmgr-repros/shared/run-defects.py <out> --board-kit K --llvm-bin LB <args> --raw
#   appvm:<runner rel path>  tier3-appvm-arm.sh --board-kit K <root> <runner> <args> --llvm-bin LB (EXPECT_IMAGES = the
#          reference's collected images)
# -- the compiler lane's reference invocations (t3-images-cw/run-refs-cw.sh) with --virtual-kit replaced by --board-kit.
# Per arm: the runner (and for appvm the arm script) must equal the hash its reference's stamp names; the board is booked
# on the lab hub (refused on a 409); the boot is scored by compare-pilot.py against ref-tB-mimic/<tag>/out. A boot that does
# not complete stops the sequence (the first failure is the reading, tier3-seq.sh's rule). Starts when <after> reads
# "PASS <after tag> DONE" (holds on a STOP/HOLD); <after> "-" starts at once. Usage:
# tier3-pass-seq.sh <board 1|2> <after|-> <after tag> <spec file> <out>. SEQ_RUN / SEQ_BUSY / SEQ_ROOT / SEQ_BOOK stub.
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} WT=${3:?after tag} SPEC=${4:?spec} O=${5:?out} S end=$(( $(date +%s) + 43200 ))
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone G=/tmp/capstone/wt-gate N=/mnt/hdd/$USER/capstone/t3-images-cw LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb-nomovc-cw/sdk C=/tmp/capstone/wt-gate/capstone/bug-corpora
  local AP=$G/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh K bs RR BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; RR=${SEQ_ROOT:-$H/t3-run-treeA/pass}; BUSY="python3 $G/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; RR=${SEQ_ROOT:-$H/t3-run-treeB/pass}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "SEQ STOP: board $BRD" >> "$O"; return 2;;
  esac
  if [ "$W" != - ]; then
    until grep -a -q "^PASS $WT DONE\|^CPY nm DONE\|^PASS STOP\|^PASS HOLD\|^CPY STOP\|^CPY HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
    grep -a -q "^PASS $WT DONE\|^CPY nm DONE" "$W" || { echo "SEQ HOLD: $WT did not end DONE ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
  fi
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    || { echo "SEQ STOP: kit $K is not the board-$BRD 54d8a3af kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = 64bc9086452fc30a ] || { echo "SEQ STOP: the SDK is not 64bc9086" >> "$O"; return 1; }
  echo "SEQ START $(date +%T) board $BRD spec $(sha256sum $SPEC | cut -c1-16) ($(grep -c "|" $SPEC) arms)" >> "$O"
  local line tag kind args rel run ref rc bid d; local -a xa
  while IFS= read -r line; do
    [ -z "$line" ] && continue; case $line in \#*) continue;; esac
    IFS='|' read -r tag kind args <<<"$line"; read -r -a xa <<<"${args:-}"
    ref=$N/ref-tB-mimic/$tag; d=$RR-$tag
    case $kind in
      cases) rel=capstone/bug-corpora/tools/run-virtual-cases.py;; pgc) rel=capstone/bug-corpora/postgres/c-repros/shared/run-arm.py;;
      pgm) rel=capstone/bug-corpora/postgres/mmgr-repros/shared/run-defects.py;; appvm:*) rel=${kind#appvm:};;
      *) echo "SEQ STOP: $tag kind $kind" >> "$O"; return 2;;
    esac
    run=$G/$rel
    grep -q "runner $rel $(sha256sum $run | cut -c1-16) " $ref.stamp 2>/dev/null || { echo "SEQ STOP: $tag runner $rel is not its stamp's" >> "$O"; return 1; }
    grep -q "SWITCHES movc_null_scalar=1 compact_rounding=ON" $ref.stamp && [ -s $ref/out/verdicts.jsonl ] || { echo "SEQ STOP: $tag has no silicon-mimic reference" >> "$O"; return 1; }
    case $kind in appvm:*) grep -q "appvm-arm $(sha256sum $AP | cut -c1-16)" $ref.stamp || { echo "SEQ STOP: $tag arm script is not its stamp's" >> "$O"; return 1; };; esac
    [ -e "$d" ] && { echo "SEQ STOP: $d exists" >> "$O"; return 1; }
    ${SEQ_BUSY:-$BUSY} > /dev/null || { echo "SEQ STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${SEQ_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2-tier12 pass $tag (rebuilt images, board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "SEQ STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "SEQ NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    mkdir -p $d; echo "PASS $tag BEGIN $(date +%T) board $BRD kind $kind booking ${bid:-none}" >> "$O"
    if [ -n "${SEQ_RUN:-}" ]; then $SEQ_RUN $kind $ref $d > $d/run.log 2>&1; rc=$?
    else case $kind in
      cases) ( cd $G && taskset -c 16-31 timeout 10800 python3 $run --board-kit $K --sdk $SDK --llvm-bin $LB "${xa[@]}" --out $d/out --raw $d/raw ) > $d/run.log 2>&1; rc=$?;;
      pgc)   ( cd $G && taskset -c 16-31 timeout 10800 python3 $run --board-kit $K --llvm-bin $LB "${xa[@]}" --out $d/out --raw $d/raw ) > $d/run.log 2>&1; rc=$?;;
      pgm)   ( cd $G && taskset -c 16-31 timeout 10800 python3 $run $d/out --board-kit $K --llvm-bin $LB "${xa[@]}" --raw $d/raw ) > $d/run.log 2>&1; rc=$?;;
      appvm:*) rmdir $d; ( cd $ref/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
             EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash $AP --board-kit $K $d $run "${xa[@]}" --llvm-bin $LB > $d.run.log 2>&1; rc=$?
             mkdir -p $d; cp $d.run.log $d/run.log;;
    esac; fi
    [ -n "$bid" ] && ${SEQ_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    local B; B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
    echo "PASS $tag END $(date +%T) rc=$rc :: $(grep -a -o 'virtual boot: [A-Z a-z]*\|boot completed\|boot DID NOT COMPLETE' $d/run.log | head -1) :: $(grep -a '^--- ' $d/run.log | tail -1)" >> "$O"
    python3 $S/compare-pilot.py $ref/out $d/out > $d.cmp.txt 2>&1; echo "PASS $tag COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
    if [ -n "$B" ] && [ -s "$B" ]; then
      echo "PASS $tag readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) DRTX $(grep -a -c '^DRTX:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) banner node-floor-2 x$(grep -a -c node-floor-2 $B)" >> "$O"
    else echo "PASS $tag readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "PASS STOP after $tag: the boot did not complete (no vbatch END rc=0 in its transcript)" >> "$O"; return 1; }
  done < "$SPEC"
  echo "PASS seq2-$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
