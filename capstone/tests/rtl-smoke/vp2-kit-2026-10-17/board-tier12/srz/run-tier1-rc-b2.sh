#!/bin/bash
# The srz-rc module c8c8a6f2 (compiler lane srz-rc-arm b976f64ed5aa) on BOARD 2, tree B: ONE arm, t1-rc (fw fc178ea5e3ad5994 on
# image-rev7-rc), judged against t1-old's record. Derived from run-tier1-srzdev-b2.sh by renaming only. Copied from run-tier1-srz-b2.sh, whose header follows.
# The srz module's Tier 1 pair on BOARD 2, tree B (2026-10-20; pre-registered in tier3-smoke/README.md before the boots):
# rev7's Tier 1 boot (b0-stats, b0-hello, vm-contract, vm-suite with sparse-churn last), its oracles and judge, on the retry
# monitor line, two cold boots that differ ONLY in capstone_vm.ko:
#   t1-old: firmware 85fc8dbecd3321a4 on image-rev7 dcfc2e15bf1bbd77 (module 39d8492d);
#   t1-srz: firmware 8502993af385416a on image-rev7-srz 67c2f89aa2e76ec1 (module 607164b9, srevroot-zero).
# Usage: run-tier1-srz-b2.sh t1-old|t1-srz. Ends by itself. T1_* stub for the dry run.
DRV_DIR=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
J=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12
main() {
  set -u
  local A=${1:?t1-old or t1-srz} WANT
  case $A in t1-rc) WANT=fc178ea5e3ad5994;; *) echo "STOP: arm $A"; return 2;; esac
  local D=${T1_OUT:-/tmp/capstone/s26/run-$A-b2} FW=${T1_FW:-/tmp/capstone/s26/fw-$A/fw_payload.bin} v r O=/tmp/capstone/tier1/oracles-rev7
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ${T1_WAIT:-bash /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-free-wait.sh 21600 2} || { echo "STOP: board 2 stayed busy"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER1_SRZ_START $A $(date +%T) fw $WANT bitstream caplifive_vp2c_8de3c48cf board 2" >> $L
  cd $DRV_DIR || return 1
  FPGA_BOARD=2 FPGA_BITSTREAM=caplifive_vp2c_8de3c48cf.bit FPGA_URL="$(cat ${T1_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url-board2})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract vm-suite" LADDER_FPGA_DIR=$O \
    BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
    timeout 9000 ${T1_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  FPGA_BOARD=2 ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!
  wait $runner; local rc=$?
  sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T); watchdog lines $(grep -c '^ALIVE\|^QUIET\|^STALE' $D/watchdog.log)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  for r in b0-stats b0-hello vm-contract vm-suite; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $O/$r.oracle) $([ "${v:-x}" = "$(cat $O/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  python3 $J/boot3/vm-contract-branch3r3.py $D/board.txt >> $L
  echo "FRTY retries $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | wc -l) at $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | sort | uniq -c | tr '\n' ' ')" >> $L
  echo "VM exits: $(grep -a -o '^VM_EXIT:[a-z_-]*:[0-9]*' $D/board.txt | tr '\n' ' ')" >> $L
  echo "VM_BEGIN without VM_EXIT: $(for b in $(grep -a -o '^VM_BEGIN:[a-z_-]*' $D/board.txt | cut -d: -f2); do grep -a -q "^VM_EXIT:$b:" $D/board.txt || echo $b; done | tr '\n' ' ')" >> $L
  python3 $J/rev7/rev7-judge.py $D/board.txt > $D/judge.txt 2>&1; echo "JUDGE rc=$? :: $(tail -1 $D/judge.txt)" >> $L
  sed 's/^/  /' $D/judge.txt >> $L
  python3 $J/boot3/cpmx-diag-read.py $D/board.txt > $D/cpmx-diag.txt 2>&1
  echo "CPMX-DIAG rc=$? :: $(tail -1 $D/cpmx-diag.txt)" >> $L
  echo "S-23: DRT1 $(grep -a -c '^DRT1:' $D/board.txt) DRTX $(grep -a -c '^DRTX:' $D/board.txt) Oops $(grep -a -c Oops $D/board.txt); banner srevroot-rcf-1 $(grep -a -c 'srevroot-rcf-1' $D/board.txt)" >> $L
  python3 - $D/board.txt >> $L <<'PYR'
import re, sys
t = open(sys.argv[1], errors="replace").read()
ctx = len(re.findall(r"CAPSTONE_VM_ADD_OK slot=0 ", t))
rel = re.findall(r"capstone_vm: release retires attempted=(\d+) succeeded=(\d+) first_id=(\d+)", t)
short = [r for r in rel if r[0] != r[1]]
print(f"RELEASE lines {len(rel)} against {ctx} contexts (CAPSTONE_VM_ADD_OK slot=0); succeeded < attempted in {len(short)}; first_id {sorted(set(r[2] for r in rel))}; attempted {[r[0] for r in rel]}")
if len(rel) < ctx: print("RELEASE SHORTFALL: fewer release lines than contexts (a shortfall, never a pass)")
PYR
  echo "TIER1_SRZ_DONE $A $(date +%T)" >> $L
}
main "$@"; exit $?
