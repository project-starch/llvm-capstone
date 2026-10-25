#!/bin/bash
# The Juliet SPATIAL stratified sample on one board (2026-10-22; pre-registered in README.md before its first boot): the
# compiler lane's census build juliet-sp/build-census (755 cases, all 261 functional variants x flow roles 01/54/67;
# manifest-spatial-sample.json 03f048a1, build.json 806cc24a, image-set digest 83fddeb7deb79e77, cw r60 SDK) through
# juliet/run.py --board-kit and probe.py --board-kit from wt-jb (juliet-board a772f0112ae3, the temporal set's), then
# judge.py from wt-corpus (93aa5ad9, the 10-19 record's) exactly as the compiler lane's tB-mimic judging (juliet-sp/run.sh).
# No seeded build (no rand() flow). Every boot is a cold board boot. The record's 10 s per-program limit.
# Usage: tier3-juliet-spatial.sh <board 1|2> <out log>. Booked on the lab hub per phase (refused on a 409).
# JS_RUN / JS_PROBE / JS_JUDGE / JS_BUSY / JS_BOOK / JS_ROOT stub the dry run.
# A DRY RUN'S VERDICTS ARE NEVER DATA: judge.py's batch validity compares step NAMES (s000.., ok-*, ctl-*), not
# contents, so a stub of another set's transcript can pass it (2026-10-22: a temporal stub made 23 of 24 spatial
# batches count). A live board is safe: each batch's transcript is that batch's own boot.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local H=/mnt/hdd/$USER/capstone J=/mnt/hdd/$USER/capstone/juliet-sp JV=/mnt/hdd/$USER/capstone/juliet-vp2 JB=/tmp/capstone/wt-jb/capstone/bug-corpora/juliet JC=/mnt/hdd/$USER/capstone/wt-corpus/capstone/bug-corpora/juliet K bs R BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${JS_ROOT:-$H/t3-run-treeA/juliet-spatial}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${JS_ROOT:-$H/t3-run-treeB/juliet-spatial}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "JS STOP: board $BRD" >> "$O"; return 2;;
  esac
  [ "$(git -C /tmp/capstone/wt-jb rev-parse --short=12 HEAD)" = a772f0112ae3 ] && [ -z "$(git -C /tmp/capstone/wt-jb status --porcelain)" ] || { echo "JS STOP: wt-jb is not a clean a772f0112ae3" >> "$O"; return 1; }
  [ "$(sha256sum $JV/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] || { echo "JS STOP: helper" >> "$O"; return 1; }
  [ "$(git -C $(dirname $(dirname $(dirname $JC))) rev-parse --short=12 HEAD 2>/dev/null)" = b3cc8cb4be5a ] && [ "$(sha256sum $JC/judge.py | cut -c1-16)" = 93aa5ad99c891034 ] || { echo "JS STOP: wt-corpus is not b3cc8cb4be5a / judge 93aa5ad9" >> "$O"; return 1; }
  [ "$(sha256sum $J/build-census/build.json | cut -c1-16)" = 806cc24afbc10194 ] && [ "$(sha256sum $J/manifest-spatial-sample.json | cut -c1-16)" = 03f048a19f5ea0f5 ] \
    && [ "$(python3 -c "import json,hashlib,sys; r=json.load(open(sys.argv[1]))['images']; print(hashlib.sha256((chr(10).join(sorted(x['image']+' '+x['sha256'] for x in r))+chr(10)).encode()).hexdigest()[:16])" $J/build-census/build.json)" = 83fddeb7deb79e77 ] \
    || { echo "JS STOP: the build record, the manifest or the image set is not the pre-registered one" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "JS STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "JS STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  local ph rc bid mins
  for ph in run probe; do
    ${JS_BUSY:-$BUSY} > /dev/null || { echo "JS STOP: board $BRD is busy before $ph" >> "$O"; return 1; }
    # The hub's cap is 8 h ("a booking is 15 min to 8 h"); 560 was refused on 2026-10-22, and both run phases took 6.6 h.
    case $ph in run) mins=480;; probe) mins=20;; esac
    bid=$(${JS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD $mins "vp2-tier12 juliet spatial $ph (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "JS STOP: the hub refused the booking (409) before $ph: $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "JS NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "JS $ph BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
    case $ph in
      run)        taskset -c 16-31 timeout 36000 ${JS_RUN:-python3 $JB/run.py} --build $J/build-census --board-kit $K --helper $JV/juliet-timeout --out $R/run > $R/run.log 2>&1; rc=$?;;
      probe)      taskset -c 16-31 timeout 3600 ${JS_PROBE:-python3 $JB/probe.py} --sdk $J/sdk --board-kit $K --helper $JV/juliet-timeout --out $R/probe > $R/probe.log 2>&1; rc=$?;;
    esac
    [ -n "$bid" ] && ${JS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "JS $ph END rc=$rc $(date +%T) :: $(grep -a -c 'completed=True' $R/$ph.log) batches completed, $(grep -a -c 'completed=False' $R/$ph.log) not :: $(tail -1 $R/$ph.log | cut -c1-140)" >> "$O"
  done
  ${JS_JUDGE:-python3 $JC/judge.py} --manifest $J/manifest-spatial-sample.json --juliet $JV/src/C --build $J/build-census --run $R/run \
    --probe $R/probe/probe.json --llvm-bin $H/t3-freeze-r60/llvm/bin --out $R/results > $R/judge.log 2>&1
  echo "JS judge rc=$? :: $(tail -2 $R/judge.log | tr '\n' ' ' | cut -c1-220)" >> "$O"
  python3 - $R/results/spatial-verdicts.tsv >> "$O" 2>&1 <<'PY'
import csv, collections, sys
try: rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
except FileNotFoundError: print("JS verdicts: NO DATA"); sys.exit(2)
print("JS verdicts:", len(rows), dict(collections.Counter((r["verdict"], r["good"], r["cause"]) for r in rows)))
m = [r["case"] for r in rows if "MISSED" in r["verdict"]]
print(f"JS MISSED rows: {len(m)}" + (" TWINS-NEEDED " + " ".join(m[:10]) if m else ""))
PY
  local tb n=0 bad=0
  for tb in $(find $R -path '*/work/board.txt' | sort); do
    n=$((n+1))
    local c="pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb) DRT1 $(grep -a -c '^DRT1:' $tb) FRTY $(grep -a -c 'FRTY:' $tb) Oops $(grep -a -c Oops $tb) INJS $(grep -a -c INJS $tb) INJL $(grep -a -c INJL $tb)"
    [ "$c" = "pc=0 0 DRT1 0 FRTY 0 Oops 0 INJS 0 INJL 0" ] || { bad=$((bad+1)); echo "JS readings ${tb#$R/}: $c" >> "$O"; }
  done
  echo "JS readings: $n boot transcripts, $bad with a nonzero pc=0/DRT1/FRTY/Oops/INJS/INJL count" >> "$O"
  echo "JS board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
