#!/bin/bash
# cpython lt0 18 / lt1 18 with the compiler lane's APPLICATION-MREMAP relinks on the batch bitstream (2026-10-24;
# pre-registered in README.md before the boots): one call per cold boot, the session b2-pred/sessions/<tag>-18-board
# (its collect/images holds only the row's image: lt0 7b871db7, lt1 8bcc40cf), replayed by tier3-replay-board.py (the
# lane's replay-stats2.py on a board kit: --stats, the call, --stats) on the board's NORANDMAPS batch kit
# bkit8-batch-nb<board>, against the lane's batch-kit QEMU reading b2-pred/out/batch2-nr-<tag>-18-board/readings.json
# (qkit7-batch2-nr; dev 60cf79496ee3, pre-registered at lane 40739a9d9389), compared on exit, fault cause and function,
# nodes_high_water, nodes_allocated_total and node_capacity.
# Usage: tier3-c18-batch.sh <board 1|2> <lt0|lt1> <out log>. C18_RUN / C18_BUSY / C18_BOOK / C18_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} TAG=${2:?lt0|lt1} O=${3:?out log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K SS REF R BUSY rc bid want
  case $TAG in lt0) want=7b871db70b7ec373;; lt1) want=8bcc40cf13731e12;; *) echo "C18 STOP: tag $TAG" >> "$O"; return 2;; esac
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "C18 STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit8-batch-nb$BRD; SS=$H/b2-pred/sessions/$TAG-18-board; REF=$H/b2-pred/out/batch2-nr-$TAG-18-board/readings.json
  R=${C18_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/c18-$TAG-mremap}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $K/board.env && grep -q norandmaps $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] || { echo "C18 STOP: $K is not the norandmaps batch kit" >> "$O"; return 1; }
  [ "$(ls $SS/collect/images | wc -l)" = 1 ] && [ "$(sha256sum $SS/collect/images/* | cut -c1-16)" = $want ] && [ "$(wc -l < $SS/plan.jsonl)" = 1 ] \
    || { echo "C18 STOP: $SS is not one call on image $want" >> "$O"; return 1; }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['completed'] and len(d['calls'])==1" $REF 2>/dev/null || { echo "C18 STOP: no completed reference at $REF" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-newarms rev-parse --short=12 HEAD)" = bcf5f1724110 ] && [ -z "$(git -C /tmp/capstone/wt-newarms status --short --untracked-files=no)" ] \
    || { echo "C18 STOP: wt-newarms is not a clean bcf5f1724110" >> "$O"; return 1; }
  [ -e $R ] && { echo "C18 STOP: $R exists" >> "$O"; return 1; }
  mkdir -p "$(dirname $R)"
  ${C18_BUSY:-$BUSY} > /dev/null || { echo "C18 STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${C18_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2 batch cpython $TAG 18 mremap (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "C18 STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "C18 NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "C18 $TAG BEGIN $(date +%T) board $BRD kit ${K##*/} image $want booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 5400 ${C18_RUN:-python3 $S/tier3-replay-board.py} $SS $K $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${C18_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "C18 $TAG END rc=$rc $(date +%T) :: $(tail -1 $R.log | cut -c1-160)" >> "$O"
  python3 - $REF $R/readings.json >> "$O" 2>&1 <<'PY'
import json, sys
def key(p):
    d = json.load(open(p)); c = d["calls"][0]; st = c.get("stats") or {}; f = c.get("fault") or {}
    return d["completed"], {"exit": (c["result"].get("kind"), c["result"].get("value")), "cause": f.get("cause"), "function": f.get("function"),
            "high_water": st.get("nodes_high_water"), "allocated_total": st.get("nodes_allocated_total"), "capacity": st.get("node_capacity"),
            "tail": c.get("tail")}
try: (ra, a), (rb, b) = key(sys.argv[1]), key(sys.argv[2])
except (FileNotFoundError, KeyError, IndexError) as e: print(f"C18 COMPARE NO DATA: {e}"); sys.exit(2)
print("C18 reference:", a); print("C18 board:    ", b)
d = [k for k in a if k != "tail" and a[k] != b[k]]
print(f"C18 COMPARE: board completed {rb}; fields differing {d if d else 'none'}")
sys.exit(1 if d or not rb else 0)
PY
  echo "C18 $TAG COMPARE rc=$?" >> "$O"
  local B=$R/guest/board.txt
  [ -s "$B" ] && echo "C18 readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
  echo "C18 $TAG board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
