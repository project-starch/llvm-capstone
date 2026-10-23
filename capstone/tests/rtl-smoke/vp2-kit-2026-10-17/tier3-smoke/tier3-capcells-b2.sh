#!/bin/bash
# Bitstream 2's capacity cells on board 2 (2026-10-24; pre-registered in README.md before the first boot): the compiler
# lane's 20 calls (b2-pred/queue.tsv: the ten lt1 capacity cells with every case/control/probe call, plus lt0 18), one
# call per cold boot, on the mremap relinks (lt1 8bcc40cf, lt0 7b871db7), replayed by tier3-replay-board.py on
# bitstream 2's plain kit bkit9-b2-b2 (the twin qkit7-b2twin3-q8 ran without norandmaps) and compared with the 18-bit
# twin's reading b2-pred/out/b2twin3-q8-<tag>/readings.json (module 33254735 = 215a09be's QEMU twin, launcher fad465ce)
# on exit, fault cause and function (the verdict) and nodes_high_water / allocated_total / node_capacity (reported).
# Each session dir is the batch session hardlinked, holding BOTH 916e14f9 and the row's relink: the call runs from a
# hardlinked copy holding ONLY the row's image. The first call is measured first on the measure-only kit bkit9-b2-mb2
# (no board) and the run stops unless that firmware is under 26,000,000 B. A boot that does not complete stops the run.
# Usage: tier3-capcells-b2.sh <out log> [tag ...] (default: the queue's order). CC_RUN / CC_BUSY / CC_BOOK / CC_ROOT stub.
main() {
  set -u
  local O=${1:?out log}; shift; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone Q=/mnt/hdd/$USER/capstone/b2-pred K KM R BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"
  K=$H/t3-smoke/bkit9-b2-b2; KM=$H/t3-smoke/bkit9-b2-mb2; R=${CC_ROOT:-$H/t3-run-b2-dcc6daf6e-b2/capcells}
  local k; for k in $K $KM; do
    ( cd $k && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_dcc6daf6e" $k/board.env && grep -qx "FPGA_BOARD=2" $k/board.env \
      && [ "$(sha256sum $k/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] && ! grep -q norandmaps $k/board.env \
      || { echo "CC STOP: $k is not board 2's bitstream-2 kit" >> "$O"; return 1; }
  done
  grep -q MEASURE-ONLY $KM/board-run.sh || { echo "CC STOP: $KM is not measure-only" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-newarms rev-parse --short=12 HEAD)" = bcf5f1724110 ] && [ -z "$(git -C /tmp/capstone/wt-newarms status --short --untracked-files=no)" ] \
    || { echo "CC STOP: wt-newarms is not a clean bcf5f1724110" >> "$O"; return 1; }
  [ -e $R ] && { echo "CC STOP: $R exists" >> "$O"; return 1; }
  local tags=("$@"); [ ${#tags[@]} = 0 ] && mapfile -t tags < <(cut -f1 $Q/queue.tsv)
  mkdir -p $R
  local tag name img ss dst ref other n=0 rc bid sz
  for tag in "${tags[@]}"; do
    read -r name img < <(awk -F'\t' -v t="$tag" '$1==t{print $2"\t"$3}' $Q/queue.tsv)
    ss=$Q/sessions/$tag; ref=$Q/out/b2twin3-q8-$tag/readings.json; dst=$R/sess-$tag
    [ -n "${img:-}" ] && [ -d $ss ] && [ "$(wc -l < $ss/plan.jsonl)" = 1 ] && grep -q "\"$name\"" $ss/plan.jsonl \
      && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['completed'] and len(d['calls'])==1" $ref 2>/dev/null \
      || { echo "CC STOP: $tag: no queue row, session or completed twin reading" >> "$O"; return 1; }
    cp -al $ss $dst || { echo "CC STOP: $tag: copy" >> "$O"; return 1; }
    for other in $(ls $dst/collect/images); do case $other in *-$img.dom) ;; *) rm -f -- "${dst:?}/collect/images/${other:?}";; esac; done
    [ "$(ls $dst/collect/images)" = "cpython-$img.dom" ] && [ "$(sha256sum $dst/collect/images/cpython-$img.dom | cut -c1-16)" = $img ] \
      || { echo "CC STOP: $tag: the staged images are not exactly cpython-$img.dom" >> "$O"; return 1; }
    n=$((n+1))
    if [ $n = 1 ]; then
      taskset -c 16-31 timeout 3600 ${CC_RUN:-python3 $S/tier3-replay-board.py} $dst $KM $R/measure-$tag > $R/measure-$tag.log 2>&1
      sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" $R/measure-$tag/guest/SEQ 2>/dev/null | grep -o "[0-9]\+" | head -1)
      echo "CC MEASURE $tag firmware ${sz:-NONE} B against 26000000" >> "$O"
      [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "CC STOP: the measured firmware is not under the cap; no boot" >> "$O"; return 1; }
    fi
    ${CC_BUSY:-$BUSY} > /dev/null || { echo "CC STOP: board 2 is busy before $tag" >> "$O"; return 1; }
    bid=$(${CC_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book 2 60 "vp2 bitstream 2 capacity cell $tag (board 2)"); rc=$?
    [ $rc = 3 ] && { echo "CC STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "CC NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "CC $tag BEGIN $(date +%T) image $img booking ${bid:-none}" >> "$O"
    taskset -c 16-31 timeout 5400 ${CC_RUN:-python3 $S/tier3-replay-board.py} $dst $K $R/$tag > $R/$tag.log 2>&1; rc=$?
    [ -n "$bid" ] && ${CC_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "CC $tag END rc=$rc $(date +%T) :: $(tail -1 $R/$tag.log | cut -c1-120)" >> "$O"
    python3 - $ref $R/$tag/readings.json $tag >> "$O" 2>&1 <<'PY'
import json, sys
def key(p):
    d = json.load(open(p)); c = d["calls"][0]; st = c.get("stats") or {}; f = c.get("fault") or {}
    return d["completed"], {"exit": (c["result"].get("kind"), c["result"].get("value")), "cause": f.get("cause"), "function": f.get("function"),
            "high_water": st.get("nodes_high_water"), "allocated_total": st.get("nodes_allocated_total"), "capacity": st.get("node_capacity")}
t = sys.argv[3]
try: (ra, a), (rb, b) = key(sys.argv[1]), key(sys.argv[2])
except (FileNotFoundError, KeyError, IndexError, TypeError) as e: print(f"CC {t} COMPARE NO DATA: {e}"); sys.exit(2)
v = [k for k in ("exit", "cause", "function", "capacity") if a[k] != b[k]]
c = {k: (a[k], b[k]) for k in ("high_water", "allocated_total") if a[k] != b[k]}
print(f"CC {t} VERDICT {'SAME' if not v else 'DIFF ' + str(v)} :: board {b['exit']} cause {b['cause']} {b['function']} high_water {b['high_water']} "
      f"allocated {b['allocated_total']} capacity {b['capacity']} :: counts vs twin {c if c else 'equal'}")
sys.exit(0 if (rb and not v) else 1)
PY
    local B=$R/$tag/guest/board.txt
    [ -s "$B" ] && echo "CC $tag readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
    [ -s "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "CC STOP after $tag: the boot did not complete" >> "$O"; return 1; }
  done
  echo "CC board2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
