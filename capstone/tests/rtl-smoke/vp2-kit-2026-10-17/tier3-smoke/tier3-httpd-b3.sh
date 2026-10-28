#!/bin/bash
# BITSTREAM 3 version (2026-10-28; pre-registered in README.md before its first boot): tier3-httpd-b2.sh with only the
# board side changed: the bitstream-3 kit bkit11-b3-b<board> (BOARD_BITSTREAM=caplifive_vp2c_e7af48457, module 215a09be
# unchanged), output <mnt-hdd>/capstone/t3-run-b3-e7af48457-b<board>/, and "bitstream 3" in the booking text. The
# worktree, runner, prebuilt images, SDK, llvm-bin, references and grade are bitstream 2's, unchanged. The bitstream-2
# version's header follows.
# httpd (APR pools 1 case, apr-util buckets 8 cases; dev 6d9e397df3ca) on bitstream 2, both virtual arms, one board boot per
# corpus x arm (2026-10-26; pre-registered in README.md before the boots). run-virtual-cases.py --prebuilt from a clean dev
# worktree at 6d9e397df3ca (runner e464b106, the runner the QEMU references record) on the board's bitstream-2 kit
# bkit9-b2-b<board>, with the compiler lane's prebuilt images <mnt-hdd>/capstone/httpd-v/build/<corpus>/<arm>[-controls]
# (each image's sha256/16 must equal the STAMP line its QEMU reference was run from, httpd-v/REFS.txt), the SDK and llvm-bin
# of that run. Graded by verdicts-compare.py against the committed QEMU reference
# capstone/bug-corpora/httpd/<corpus>/results/2026-10-26-virtual/<arm> (qkit7-b2twin3-q8). Output
# <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/<HT_SUB:-httpd>/<corpus>-<arm>. A boot that does not complete stops the run.
# Usage: tier3-httpd-b2.sh <board 1|2> <out log>. HT_RUN / HT_BUSY / HT_BOOK / HT_ROOT stub the dry run; HT_SUB names the
# output subdirectory (an N=2 run uses its own).
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local W=/tmp/capstone/wt-httpd-b SDK=$H/t3-freeze-nb-nomovc-cw-r60/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6 D=$H/httpd-v K R BUSY
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "HT STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit11-b3-b$BRD; R=${HT_ROOT:-$H/t3-run-b3-e7af48457-b$BRD/${HT_SUB:-httpd}}
  [ "$(git -C $W rev-parse --short=12 HEAD)" = 6d9e397df3ca ] && [ -z "$(git -C $W status --porcelain)" ] \
    && [ "$(sha256sum $W/capstone/bug-corpora/tools/run-virtual-cases.py | cut -c1-16)" = e464b106e1eb5c8e ] \
    || { echo "HT STOP: $W is not a clean 6d9e397df3ca with runner e464b106" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_e7af48457" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] && [ ! -e $K/adapter/module/params ] \
    || { echo "HT STOP: $K is not board $BRD's bitstream-3 kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] || { echo "HT STOP: the SDK" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "HT STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  local c arm B o ref st got rc bid
  for c in apr-pool-repros bucket-repros; do
    for arm in virtual-malloc virtual-nested-pools; do
      B=$D/build/$c/$arm; o=$R/$c-$arm; ref=$W/capstone/bug-corpora/httpd/$c/results/2026-10-26-virtual/$arm
      st=$(grep "^STAMP $c $arm " $D/REFS.txt | head -1 | cut -d' ' -f4-)
      got="$(for f in $B/bin/*; do echo -n "$(basename $f)=$(sha256sum $f | cut -c1-16) "; done)ctl $(sha256sum $B-controls/bin/defect-90 | cut -c1-16)"
      [ -n "$st" ] && [ "$st" = "$got" ] && [ -s $ref/verdicts.jsonl ] \
        || { echo "HT STOP: $c $arm: the images are not the reference's (REFS '${st:-none}' vs '$got') or no reference" >> "$O"; return 1; }
      ${HT_BUSY:-$BUSY} > /dev/null || { echo "HT STOP: board $BRD is busy before $c $arm" >> "$O"; return 1; }
      bid=$(${HT_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 45 "vp2 bitstream 3 httpd $c $arm (board $BRD)"); rc=$?
      [ $rc = 3 ] && { echo "HT STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
      [ $rc = 0 ] || { echo "HT NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
      echo "HT $c $arm BEGIN $(date +%T) board $BRD images $got booking ${bid:-none}" >> "$O"
      ( cd $W/capstone && taskset -c 16-31 timeout 5400 ${HT_RUN:-python3 $W/capstone/bug-corpora/tools/run-virtual-cases.py} --corpus $W/capstone/bug-corpora/httpd/$c \
          --arm $arm --board-kit $K --sdk $SDK --llvm-bin $LB --prebuilt $B --prebuilt-controls $B-controls --out $o/out --raw $o/raw ) > $o.log 2>&1; rc=$?
      [ -n "$bid" ] && ${HT_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
      echo "HT $c $arm END rc=$rc $(date +%T) :: $(grep -a '^--- ' $o.log | tail -1 | cut -c1-160)" >> "$O"
      grep -a -E "control |board boot|virtual boot" $o.log | sed 's/^/  /' | cut -c1-200 >> "$O"
      local v; v=$(find $o/out -name verdicts.jsonl | head -1)
      python3 $S/verdicts-compare.py $ref $(dirname ${v:-$o/out/none}) > $o.cmp.txt 2>&1; echo "HT $c $arm COMPARE vs reference rc=$? :: $(tail -1 $o.cmp.txt)" >> "$O"
      grep "^DIFF" $o.cmp.txt | cut -c1-220 >> "$O"
      local tb; tb=$(find $o/raw -name board.txt -path '*guest*' 2>/dev/null | head -1)
      if [ -n "$tb" ] && [ -s "$tb" ]; then
        echo "HT $c $arm readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb) DRT1 $(grep -a -c '^DRT1:' $tb) FRTY $(grep -a -c 'FRTY:' $tb) Oops $(grep -a -c Oops $tb) INJS $(grep -a -c INJS $tb) INJL $(grep -a -c INJL $tb)" >> "$O"
      fi
      [ -n "$tb" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$tb" || { echo "HT STOP after $c $arm: the boot did not complete" >> "$O"; return 1; }
    done
  done
  echo "HT board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
