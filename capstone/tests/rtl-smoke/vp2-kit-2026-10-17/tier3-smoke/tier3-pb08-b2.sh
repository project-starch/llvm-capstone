#!/bin/bash
# memcached/allocator-repros case 08 on the q15 arm (pb13q15 08), the compiler lane's usable-extent FILL variant and its
# UNFLAGGED control, on bitstream 2 (2026-10-26; pre-registered in README.md before the boots): the lane's pair recipe
# <mnt-hdd>/capstone/pb08/run-pair.sh (pre-registered at its lane 8cd19dfb30ee; result dev 66e305edaa66) with --board-kit:
# run-virtual-cases.py --only 08 --arm virtual-nested-pools from a clean worktree at 8cd19dfb30ee (runner 9b5e409c), the
# prebuilt images pb08/b13-<v>/{cases,controls} (sha256/16 equal to PAIR.txt's STAMP lines), unflagged then fill, one boot
# each, on bkit9-b2-b<board>. Graded by verdicts-compare.py against the lane's QEMU bundle pb08/run-<v>/bundle
# (qkit7-b2twin3-q8). Output <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/pb08/<v>.
# Usage: tier3-pb08-b2.sh <board 1|2> <out log>. PB_RUN / PB_BUSY / PB_BOOK / PB_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local G=/tmp/capstone/wt-pb08-b D=$H/pb08 SDK=$H/t3-freeze-nb-nomovc-cw-r60/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6 K R BUSY
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PB STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit9-b2-b$BRD; R=${PB_ROOT:-$H/t3-run-b2-dcc6daf6e-b$BRD/pb08}
  local C=$G/capstone/bug-corpora
  [ "$(git -C $G rev-parse --short=12 HEAD)" = 8cd19dfb30ee ] && [ -z "$(git -C $G status --porcelain)" ] \
    && [ "$(sha256sum $C/tools/run-virtual-cases.py | cut -c1-16)" = 9b5e409c92e1fe66 ] \
    || { echo "PB STOP: $G is not a clean 8cd19dfb30ee with runner 9b5e409c" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_dcc6daf6e" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] && [ ! -e $K/adapter/module/params ] \
    || { echo "PB STOP: $K is not board $BRD's bitstream-2 kit" >> "$O"; return 1; }
  [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] || { echo "PB STOP: the SDK" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "PB STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  local v B o st got rc bid tb
  for v in unflagged fill; do
    B=$D/b13-$v; o=$R/$v
    st=$(grep "^STAMP $v " $D/PAIR.txt | head -1 | cut -d' ' -f3-)
    got="defect-08 $(sha256sum $B/cases/bin/defect-08 | cut -c1-16) ctl $(sha256sum $B/controls/bin/defect-90 | cut -c1-16)"
    [ -n "$st" ] && [ "$st" = "$got" ] && [ -s $D/run-$v/bundle/verdicts.jsonl ] \
      || { echo "PB STOP: $v: the images are not the reference's (PAIR '${st:-none}' vs '$got') or no reference" >> "$O"; return 1; }
    ${PB_BUSY:-$BUSY} > /dev/null || { echo "PB STOP: board $BRD is busy before $v" >> "$O"; return 1; }
    bid=$(${PB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 bitstream 2 pb13q15 08 $v (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "PB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "PB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "PB $v BEGIN $(date +%T) board $BRD images $got booking ${bid:-none}" >> "$O"
    ( cd $G && taskset -c 16-31 timeout 3600 ${PB_RUN:-python3 $C/tools/run-virtual-cases.py} --corpus $C/memcached/allocator-repros --arm virtual-nested-pools \
        --only 08 --board-kit $K --sdk $SDK --llvm-bin $LB --prebuilt $B/cases --prebuilt-controls $B/controls --out $o/bundle --raw $o/raw ) > $o.log 2>&1; rc=$?
    [ -n "$bid" ] && ${PB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "PB $v END rc=$rc $(date +%T) :: $(grep -a '^--- ' $o.log | tail -1 | cut -c1-160)" >> "$O"
    grep -a -E "control |board boot|virtual boot" $o.log | sed 's/^/  /' | cut -c1-200 >> "$O"
    python3 $S/verdicts-compare.py $D/run-$v/bundle $o/bundle > $o.cmp.txt 2>&1; echo "PB $v COMPARE vs reference rc=$? :: $(tail -1 $o.cmp.txt)" >> "$O"
    grep "^DIFF" $o.cmp.txt | cut -c1-220 >> "$O"
    python3 - $o/bundle/verdicts.jsonl >> "$O" 2>&1 <<'PY'
import json, sys
for l in open(sys.argv[1]):
    if l.strip():
        r = json.loads(l); o = r["observation"]; f = o.get("fault") or {}
        print(f"PB cell {o['case'][:44]} {r['verdict']} {r.get('reason') or ''} cause={f.get('cause')} sym={f.get('symbol')} attr={o.get('attribution')} notes={(o.get('notes') or '')[:60]}")
PY
    tb=$(find $o/raw -name board.txt -path '*guest*' 2>/dev/null | head -1)
    [ -n "$tb" ] && [ -s "$tb" ] && echo "PB $v readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb) DRT1 $(grep -a -c '^DRT1:' $tb) FRTY $(grep -a -c 'FRTY:' $tb) Oops $(grep -a -c Oops $tb) INJS $(grep -a -c INJS $tb) INJL $(grep -a -c INJL $tb)" >> "$O"
    [ -n "$tb" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$tb" || { echo "PB STOP after $v: the boot did not complete" >> "$O"; return 1; }
  done
  echo "PB board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
