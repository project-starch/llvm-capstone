#!/bin/bash
# The Juliet TEMPORAL set on one board (2026-10-21; pre-registered in README.md before its first boot): juliet/run.py
# --board-kit on the compiler lane's cw census build (366 cases, 732 images, 12 batches of 32 pairs) and seeded build,
# probe.py --board-kit, then judge.py, all from worktree wt-jb at juliet-board a772f0112ae3 (judge/build/select/common
# byte-identical to the QEMU judging's tree). The record's 10 s per-program limit. Every boot is a cold board boot.
# Twins are built only for MISSED twin-family cases: none are expected, and a MISSED row prints TWINS-NEEDED.
# Usage: tier3-juliet-full.sh <board 1|2> <out log>. Booked on the lab hub per phase (refused on a 409).
# JF_RUN / JF_PROBE / JF_JUDGE / JF_BUSY / JF_BOOK / JF_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local H=/mnt/hdd/$USER/capstone J=/mnt/hdd/$USER/capstone/juliet-vp2 JB=/tmp/capstone/wt-jb/capstone/bug-corpora/juliet K bs R BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${JF_ROOT:-$H/t3-run-treeA/juliet-temporal}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${JF_ROOT:-$H/t3-run-treeB/juliet-temporal}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "JF STOP: board $BRD" >> "$O"; return 2;;
  esac
  [ "$(git -C /tmp/capstone/wt-jb rev-parse --short=12 HEAD)" = a772f0112ae3 ] && [ -z "$(git -C /tmp/capstone/wt-jb status --porcelain)" ] || { echo "JF STOP: wt-jb is not a clean a772f0112ae3" >> "$O"; return 1; }
  [ "$(sha256sum $J/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] || { echo "JF STOP: helper" >> "$O"; return 1; }
  [ "$(sha256sum $J/cw/build-census/build.json | cut -c1-16)" = 3c973076cf93aad6 ] && [ "$(sha256sum $J/cw/build-seeded/build.json | cut -c1-16)" = "${JF_SEEDED_SHA:-c6a8b650f7923845}" ] \
    && [ "$(sha256sum $J/manifest-temporal.json | cut -c1-16)" = "${JF_MANIFEST_SHA:-95ceac7daa991165}" ] || { echo "JF STOP: a build record or the manifest is not the pre-registered one" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "JF STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "JF STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  local ph rc bid mins
  for ph in run run-seeded probe; do
    ${JF_BUSY:-$BUSY} > /dev/null || { echo "JF STOP: board $BRD is busy before $ph" >> "$O"; return 1; }
    case $ph in run) mins=240;; run-seeded) mins=40;; probe) mins=20;; esac
    bid=$(${JF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD $mins "vp2-tier12 juliet temporal $ph (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "JF STOP: the hub refused the booking (409) before $ph: $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "JF NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "JF $ph BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
    case $ph in
      run)        taskset -c 16-31 timeout 28800 ${JF_RUN:-python3 $JB/run.py} --build $J/cw/build-census --board-kit $K --helper $J/juliet-timeout --out $R/run > $R/run.log 2>&1; rc=$?;;
      run-seeded) taskset -c 16-31 timeout 7200 ${JF_RUN:-python3 $JB/run.py} --build $J/cw/build-seeded --board-kit $K --helper $J/juliet-timeout --out $R/run-seeded > $R/run-seeded.log 2>&1; rc=$?;;
      probe)      taskset -c 16-31 timeout 3600 ${JF_PROBE:-python3 $JB/probe.py} --sdk $J/cw/sdk --board-kit $K --helper $J/juliet-timeout --out $R/probe > $R/probe.log 2>&1; rc=$?;;
    esac
    [ -n "$bid" ] && ${JF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "JF $ph END rc=$rc $(date +%T) :: $(grep -a -c 'completed=True' $R/$ph.log) batches completed, $(grep -a -c 'completed=False' $R/$ph.log) not :: $(tail -1 $R/$ph.log | cut -c1-140)" >> "$O"
  done
  ${JF_JUDGE:-python3 $JB/judge.py} --manifest $J/manifest-temporal.json --juliet $J/src/C --build $J/cw/build-census --run $R/run \
    --seeded-build $J/cw/build-seeded --seeded-run $R/run-seeded --probe $R/probe/probe.json --llvm-bin $H/t3-freeze/llvm/bin --out $R/results > $R/judge.log 2>&1
  echo "JF judge rc=$? :: $(tail -2 $R/judge.log | tr '\n' ' ' | cut -c1-220)" >> "$O"
  python3 - $R/results/temporal-verdicts.tsv >> "$O" 2>&1 <<'PY'
import csv, collections, sys
try: rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
except FileNotFoundError: print("JF verdicts: NO DATA"); sys.exit(2)
print("JF verdicts:", len(rows), dict(collections.Counter((r["verdict"], r["good"], r["cause"]) for r in rows)))
m = [r["case"] for r in rows if "MISSED" in r["verdict"]]
print(f"JF MISSED rows: {len(m)}" + (" TWINS-NEEDED " + " ".join(m[:10]) if m else ""))
PY
  local tb n=0 bad=0
  for tb in $(find $R -path '*/work/board.txt' | sort); do
    n=$((n+1))
    local c="pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb) DRT1 $(grep -a -c '^DRT1:' $tb) FRTY $(grep -a -c 'FRTY:' $tb) Oops $(grep -a -c Oops $tb) INJS $(grep -a -c INJS $tb) INJL $(grep -a -c INJL $tb)"
    [ "$c" = "pc=0 0 DRT1 0 FRTY 0 Oops 0 INJS 0 INJL 0" ] || { bad=$((bad+1)); echo "JF readings ${tb#$R/}: $c" >> "$O"; }
  done
  echo "JF readings: $n boot transcripts, $bad with a nonzero pc=0/DRT1/FRTY/Oops/INJS/INJL count" >> "$O"
  echo "JF board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
