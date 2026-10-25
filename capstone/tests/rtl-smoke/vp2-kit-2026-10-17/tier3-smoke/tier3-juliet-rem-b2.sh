#!/bin/bash
# The Juliet spatial REMAINDER on bitstream 2, one part per call (2026-10-26; each part pre-registered in README.md before
# its boot). The compiler lane cuts the full spatial manifest minus the 755-case sample into 12 parts
# (<mnt-hdd>/capstone/juliet-rem/manifest-partNN.json) and makes each part's QEMU reference with run-parts.sh on the
# 18-bit twin qkit7-b2twin3-q8: partNN/build-census, build-seeded, run/, run-seeded/, probe/ and results/, marked
# .done-partNN, with its lines in RESULT-<lane>.
# This runs the SAME builds on the board's bitstream-2 kit bkit9-b2-b<board>, with the same run.py / probe.py / judge.py
# as the sample (tier3-juliet-spatial-b2.sh). THE GRADE is juliet-obs-compare.py of the board's run/ and run-seeded/
# observations against the part's QEMU observations. No tolerance is applied: the three tolerated CWE170 cases are sample
# cases, and any value-dependent case in a part is reported by name, never folded in. judge.py runs for the record only,
# because it cannot read CAUGHT on silicon. Output: <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/juliet-rem/partNN.
# Usage: tier3-juliet-rem-b2.sh <board 1|2> <out log> <part NN>.
# JR_RUN / JR_PROBE / JR_JUDGE / JR_BUSY / JR_BOOK / JR_ROOT / JR_PARTS stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} NN=${3:?part NN}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone JV=/mnt/hdd/$USER/capstone/juliet-vp2 JB=/tmp/capstone/wt-jb/capstone/bug-corpora/juliet
  local JC=/mnt/hdd/$USER/capstone/wt-corpus/capstone/bug-corpora/juliet SDK=/mnt/hdd/$USER/capstone/juliet-sp/sdk
  local D=${JR_PARTS:-$H/juliet-rem} K bs R BUSY
  case $NN in [0-9][0-9]) ;; *) echo "JR STOP: part $NN" >> "$O"; return 2;; esac
  local P=$D/part$NN M=$D/manifest-part$NN.json
  case $BRD in
    1) K=$H/t3-smoke/bkit9-b2-b1; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit9-b2-b2; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "JR STOP: board $BRD" >> "$O"; return 2;;
  esac
  bs=caplifive_vp2c_dcc6daf6e; R=${JR_ROOT:-$H/t3-run-b2-dcc6daf6e-b$BRD/juliet-rem/part$NN}
  # The part's reference must be complete, made on the twin, by these runners, with every phase rc=0.
  [ -e $D/.done-part$NN ] && [ -s $M ] && [ -s $P/run/observations.json ] && [ -s $P/run-seeded/observations.json ] \
    && [ -s $P/results/spatial-verdicts.tsv ] || { echo "JR STOP: part $NN's reference is not complete" >> "$O"; return 1; }
  local RL; RL=$(grep -l "^PART $NN manifest $(sha256sum $M | cut -c1-16) " $D/RESULT-* 2>/dev/null | head -1)
  [ -n "$RL" ] || { echo "JR STOP: no RESULT line for part $NN with its manifest's hash" >> "$O"; return 1; }
  python3 - $RL $NN <<'PY' >> "$O" || { echo "JR STOP: part $NN's reference phases are not all rc=0" >> "$O"; return 1; }
import re, sys
t = open(sys.argv[1]).read(); nn = sys.argv[2]
seg = t.split(f"PART {nn} manifest", 1)[1].split("\nPART ", 1)[0]
ok = all(f"{k} rc=0" in seg for k in ("build census", "build seeded", "run", "run-seeded", "probe", "judge"))
print(f"JR part {nn} reference: phases {'all rc=0' if ok else 'NOT all rc=0'}; RESULT {sys.argv[1].rsplit('/', 1)[1]}")
sys.exit(0 if ok else 1)
PY
  grep -q "^  K=\$H/t3-smoke/qkit7-b2twin3-q8;" $D/run-parts.sh || { echo "JR STOP: run-parts.sh does not name the 18-bit twin" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-jb rev-parse --short=12 HEAD)" = a772f0112ae3 ] && [ -z "$(git -C /tmp/capstone/wt-jb status --porcelain)" ] \
    && [ "$(sha256sum $JB/run.py | cut -c1-16)" = 4b663651d0638eef ] && [ "$(sha256sum $JB/probe.py | cut -c1-16)" = d6200b39c312f601 ] \
    && [ "$(sha256sum $JC/judge.py | cut -c1-16)" = 93aa5ad99c891034 ] && [ "$(sha256sum $JV/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] \
    && [ "$(sha256sum $SDK/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] \
    || { echo "JR STOP: the runners, helper or SDK are not the sample's" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "JR STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "JR STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  echo "JR part $NN manifest $(sha256sum $M | cut -c1-16) cases $(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cases']))" $M) census images $(find -L $P/build-census -name '*.dom' | wc -l) seeded images $(find -L $P/build-seeded -name '*.dom' | wc -l)" >> "$O"
  local ph rc bid mins
  for ph in run run-seeded probe; do
    ${JR_BUSY:-$BUSY} > /dev/null || { echo "JR STOP: board $BRD is busy before $ph" >> "$O"; return 1; }
    case $ph in run) mins=480;; run-seeded) mins=90;; probe) mins=20;; esac
    bid=$(${JR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD $mins "vp2 bitstream 2 juliet spatial remainder part $NN $ph (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "JR STOP: the hub refused the booking (409) before $ph: $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "JR NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "JR $ph BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
    case $ph in
      run)        taskset -c 16-31 timeout 54000 ${JR_RUN:-python3 $JB/run.py} --build $P/build-census --board-kit $K --helper $JV/juliet-timeout --out $R/run > $R/run.log 2>&1; rc=$?;;
      run-seeded) taskset -c 16-31 timeout 10800 ${JR_RUN:-python3 $JB/run.py} --build $P/build-seeded --board-kit $K --helper $JV/juliet-timeout --out $R/run-seeded > $R/run-seeded.log 2>&1; rc=$?;;
      probe)      taskset -c 16-31 timeout 3600 ${JR_PROBE:-python3 $JB/probe.py} --sdk $SDK --board-kit $K --helper $JV/juliet-timeout --out $R/probe > $R/probe.log 2>&1; rc=$?;;
    esac
    [ -n "$bid" ] && ${JR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "JR $ph END rc=$rc $(date +%T) :: $(grep -a -c 'completed=True' $R/$ph.log) batches completed, $(grep -a -c 'completed=False' $R/$ph.log) not :: $(tail -1 $R/$ph.log | cut -c1-140)" >> "$O"
  done
  ${JR_JUDGE:-python3 $JC/judge.py} --manifest $M --juliet $JV/src/C --build $P/build-census --run $R/run --seeded-build $P/build-seeded \
    --seeded-run $R/run-seeded --probe $R/probe/probe.json --llvm-bin $H/t3-freeze/llvm/bin --out $R/results > $R/judge.log 2>&1
  echo "JR judge rc=$? (not a reading on silicon) :: $(tail -1 $R/judge.log | cut -c1-200)" >> "$O"
  # THE GRADE: juliet-obs-compare against the part's own QEMU observations, census then seeded. No tolerance.
  python3 $S/juliet-obs-compare.py $P/run/observations.json $R/run/observations.json $P/build-census $H/t3-freeze/llvm/bin > $R/obs-census.cmp 2>&1
  echo "JR OBS-COMPARE census rc=$? :: $(tail -1 $R/obs-census.cmp)" >> "$O"
  python3 $S/juliet-obs-compare.py $P/run-seeded/observations.json $R/run-seeded/observations.json $P/build-seeded $H/t3-freeze/llvm/bin > $R/obs-seeded.cmp 2>&1
  echo "JR OBS-COMPARE seeded rc=$? :: $(tail -1 $R/obs-seeded.cmp)" >> "$O"
  python3 - $R/obs-census.cmp $R/obs-seeded.cmp >> "$O" <<'PY'
import sys
for p in sys.argv[1:]:
    for l in open(p):
        if l.startswith("DIFF "):
            k = l.split(":", 1)[0][5:]
            kind = ("uaf-control" if k == "controls:ctl-uaf" else "rand-driven _12 draw" if "_12." in k
                    else "CWE170 (value-dependent candidate)" if k.startswith("CWE170") else "OTHER")
            print(f"JR DIFF [{kind}] {p.rsplit('/', 1)[1]} {l.rstrip()[5:230]}")
PY
  local tb n=0 bad=0
  for tb in $(find $R -path '*/work/board.txt' | sort); do
    n=$((n+1))
    local c="pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb) DRT1 $(grep -a -c '^DRT1:' $tb) FRTY $(grep -a -c 'FRTY:' $tb) Oops $(grep -a -c Oops $tb) INJS $(grep -a -c INJS $tb) INJL $(grep -a -c INJL $tb)"
    [ "$c" = "pc=0 0 DRT1 0 FRTY 0 Oops 0 INJS 0 INJL 0" ] || { bad=$((bad+1)); echo "JR readings ${tb#$R/}: $c" >> "$O"; }
  done
  echo "JR readings: $n boot transcripts, $bad with a nonzero pc=0/DRT1/FRTY/Oops/INJS/INJL count" >> "$O"
  echo "JR part $NN board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
