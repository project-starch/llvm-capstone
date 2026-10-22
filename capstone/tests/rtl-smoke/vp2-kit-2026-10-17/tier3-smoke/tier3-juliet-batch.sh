#!/bin/bash
# The Juliet TEMPORAL set on the BATCH bitstream c721c1a2a (2026-10-23; pre-registered in README.md before its first boot):
# tier3-juliet-full.sh with what the batch changes:
#   - the kit is the board's batch kit bkit8-batch-b<board> (c721c1a2a, module 21fbfc39, monitor d9da397);
#   - the builds are the compiler lane's r60 rebuilds, juliet-r60/r60/build-{census,seeded} (build.json 53c221ca / c346e5c6;
#     734 and 55 images, the r60 SDK sdk.json f5145cbb, manifest 95ceac7d), and probe.py's SDK is juliet-r60/r60/sdk;
#   - the reference is that lane's batch-kit QEMU judging juliet-r60/judge-batch2 (qkit7-batch2; dev cc95944a056e,
#     pre-registered at lane fd93b0d8ab47: 366 CAUGHT, 366/366 equal to judge-tB-mimic), compared case by case on
#     (verdict, good, cause, site, strict), the lane's own cmp;
#   - output to a NEW directory, <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/juliet-temporal.
# run.py 4b663651 / probe.py d6200b39 / judge.py 93aa5ad9 from wt-jb a772f0112ae3, the reference's hashes. The original header follows.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local H=/mnt/hdd/$USER/capstone J=/mnt/hdd/$USER/capstone/juliet-vp2 JR=/mnt/hdd/$USER/capstone/juliet-r60 JB=/tmp/capstone/wt-jb/capstone/bug-corpora/juliet K bs R BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit8-batch-b1; bs=caplifive_vp2c_c721c1a2a; R=${JF_ROOT:-$H/t3-run-batch-c721c1a2a-b1/juliet-temporal}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit8-batch-b2; bs=caplifive_vp2c_c721c1a2a; R=${JF_ROOT:-$H/t3-run-batch-c721c1a2a-b2/juliet-temporal}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "JF STOP: board $BRD" >> "$O"; return 2;;
  esac
  [ "$(git -C /tmp/capstone/wt-jb rev-parse --short=12 HEAD)" = a772f0112ae3 ] && [ -z "$(git -C /tmp/capstone/wt-jb status --porcelain)" ] || { echo "JF STOP: wt-jb is not a clean a772f0112ae3" >> "$O"; return 1; }
  [ "$(sha256sum $J/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] || { echo "JF STOP: helper" >> "$O"; return 1; }
  [ "$(sha256sum $JR/r60/build-census/build.json | cut -c1-16)" = 53c221caefc4e7e6 ] && [ "$(sha256sum $JR/r60/build-seeded/build.json | cut -c1-16)" = c346e5c6e498828d ] \
    && [ "$(sha256sum $JR/manifest-temporal.json | cut -c1-16)" = 95ceac7daa991165 ] && [ "$(sha256sum $JR/r60/sdk/sdk.json | cut -c1-16)" = f5145cbb88dde6b5 ] \
    || { echo "JF STOP: a build record, the manifest or the SDK is not the pre-registered one" >> "$O"; return 1; }
  [ "$(sha256sum $JB/run.py | cut -c1-16)" = 4b663651d0638eef ] && [ "$(sha256sum $JB/probe.py | cut -c1-16)" = d6200b39c312f601 ] && [ "$(sha256sum $JB/judge.py | cut -c1-16)" = 93aa5ad99c891034 ] \
    && grep -q "^STAMP batch2 run.py 4b663651d0638eef probe.py d6200b39c312f601 judge.py 93aa5ad99c891034 qemu ecaf50bbd5ec44aa module a05c198bfcc5f55c" $JR/RESULT \
    && grep -q "tB-mimic -> batch2: cases 366/366 differ 0" $JR/RESULT && [ -s $JR/judge-batch2/results/temporal-verdicts.tsv ] \
    || { echo "JF STOP: the runners or the batch reference are not the pre-registered ones" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "JF STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "JF STOP: $R exists" >> "$O"; return 1; }
  mkdir -p $R
  local ph rc bid mins
  for ph in run run-seeded probe; do
    ${JF_BUSY:-$BUSY} > /dev/null || { echo "JF STOP: board $BRD is busy before $ph" >> "$O"; return 1; }
    case $ph in run) mins=240;; run-seeded) mins=40;; probe) mins=20;; esac
    bid=$(${JF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD $mins "vp2 batch juliet temporal $ph (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "JF STOP: the hub refused the booking (409) before $ph: $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "JF NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    echo "JF $ph BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
    case $ph in
      run)        taskset -c 16-31 timeout 28800 ${JF_RUN:-python3 $JB/run.py} --build $JR/r60/build-census --board-kit $K --helper $J/juliet-timeout --out $R/run > $R/run.log 2>&1; rc=$?;;
      run-seeded) taskset -c 16-31 timeout 7200 ${JF_RUN:-python3 $JB/run.py} --build $JR/r60/build-seeded --board-kit $K --helper $J/juliet-timeout --out $R/run-seeded > $R/run-seeded.log 2>&1; rc=$?;;
      probe)      taskset -c 16-31 timeout 3600 ${JF_PROBE:-python3 $JB/probe.py} --sdk $JR/r60/sdk --board-kit $K --helper $J/juliet-timeout --out $R/probe > $R/probe.log 2>&1; rc=$?;;
    esac
    [ -n "$bid" ] && ${JF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "JF $ph END rc=$rc $(date +%T) :: $(grep -a -c 'completed=True' $R/$ph.log) batches completed, $(grep -a -c 'completed=False' $R/$ph.log) not :: $(tail -1 $R/$ph.log | cut -c1-140)" >> "$O"
  done
  ${JF_JUDGE:-python3 $JB/judge.py} --manifest $JR/manifest-temporal.json --juliet $J/src/C --build $JR/r60/build-census --run $R/run \
    --seeded-build $JR/r60/build-seeded --seeded-run $R/run-seeded --probe $R/probe/probe.json --llvm-bin $H/t3-freeze/llvm/bin --out $R/results > $R/judge.log 2>&1
  echo "JF judge rc=$? :: $(tail -2 $R/judge.log | tr '\n' ' ' | cut -c1-220)" >> "$O"
  python3 - $R/results/temporal-verdicts.tsv >> "$O" 2>&1 <<'PY'
import csv, collections, sys
try: rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
except FileNotFoundError: print("JF verdicts: NO DATA"); sys.exit(2)
print("JF verdicts:", len(rows), dict(collections.Counter((r["verdict"], r["good"], r["cause"]) for r in rows)))
m = [r["case"] for r in rows if "MISSED" in r["verdict"]]
print(f"JF MISSED rows: {len(m)}" + (" TWINS-NEEDED " + " ".join(m[:10]) if m else ""))
PY
  python3 - $JR/judge-batch2/results/temporal-verdicts.tsv $R/results/temporal-verdicts.tsv >> "$O" 2>&1 <<'PY'
import csv, sys
K = ("verdict", "good", "cause", "site", "strict")
def load(p):
    try: return {r["case"]: tuple(r.get(k, "") for k in K) for r in csv.DictReader(open(p), delimiter="\t")}
    except FileNotFoundError: return {}
a, b = load(sys.argv[1]), load(sys.argv[2])
if not a or not b: print(f"JF COMPARE NO DATA {len(a)}/{len(b)}"); sys.exit(2)
d = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print(f"JF COMPARE batch QEMU -> board: cases {len(a)}/{len(b)} differ {len(d)}" + "".join(f"\n  {k}: {a.get(k)} -> {b.get(k)}" for k in d[:40]))
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
