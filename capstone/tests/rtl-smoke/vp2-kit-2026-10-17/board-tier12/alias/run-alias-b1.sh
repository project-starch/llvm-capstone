#!/bin/bash
# The alias-load test on BOARD 1, tree A (2026-10-20; pre-registered in tier3-smoke/README.md before the boot): ONE cold boot of
# firmware 08b1a40c7373f381 on image-alias de20785c65780151 (rev7's Tier 1 stage, module 39d8492d, the retry monitor line),
# rungs b0-stats, b0-hello, alias-load. Per arm (inb, alias, outside, past) it reports AL_PRE, AL_LOADED, AL_DONE, any
# domain fault and the exit status. Derived from ff/run-ff-b2.sh. Usage: run-alias-b1.sh. Ends by itself. FF_* stub.
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
main() {
  set -u
  local D=${FF_OUT:-/tmp/capstone/s26/run-alias-b1} FW=${FF_FW:-/tmp/capstone/s26/fw-alias/fw_payload.bin} O=/tmp/capstone/s26/oracles-alias r
  [ "$(sha256sum $FW | cut -c1-16)" = 08b1a40c7373f381 ] || { echo "STOP: the firmware is not 08b1a40c"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: driver"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists"; return 1; }
  mkdir -p $D $O; for r in b0-stats b0-hello alias-load; do echo 0 > $O/$r.oracle; done; local L=$D/SEQ
  ${FF_WAIT:-bash $DRV_DIR/board-free-wait.sh 21600 1} || { echo "STOP: board 1 stayed busy" >> $L; return 1; }
  echo "AL_START $(date +%T) fw 08b1a40c7373f381 bitstream caplifive_vp2c_8522d72fe board 1" >> $L
  cd $DRV_DIR || return 1
  FPGA_BOARD=1 FPGA_BITSTREAM=caplifive_vp2c_8522d72fe.bit FPGA_URL="$(cat $HOME/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello alias-load" LADDER_FPGA_DIR=$O \
    BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
    timeout 9000 ${FF_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  FPGA_BOARD=1 ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!; wait $runner; local rc=$?; sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  for r in b0-stats b0-hello alias-load; do
    echo "RUNG $r retval=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')" >> $L
  done
  python3 - $D/board.txt >> $L <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", "").split("\n")
arm = None; rows = {}
for l in t:
    m = re.match(r"^AL_ARM (\S+) (\S+)", l)
    if m: arm = m.group(1); rows[arm] = {"off": m.group(2), "pre": [], "loaded": [], "done": 0, "fault": [], "exit": None}; continue
    if arm is None: continue
    if re.search(r"AL_PRE", l): rows[arm]["pre"].append(l.strip()[:200])
    m = re.search(r"AL_LOADED value=(\S+)", l)
    if m: rows[arm]["loaded"].append(m.group(1))
    if "AL_DONE" in l: rows[arm]["done"] += 1
    m = re.search(r"domain fault cause=(\d+) pc=(\S+) address=(\S+)", l)
    if m: rows[arm]["fault"].append(f"cause={m.group(1)} address={m.group(3)}")
    m = re.match(r"^AL_EXIT (\S+) (\d+)", l)
    if m: rows[arm]["exit"] = int(m.group(2)); arm = None
if not rows: print("AL NO DATA: no AL_ARM marker"); sys.exit(0)
for a, r in rows.items():
    print(f"AL {a} off={r['off']}: exit {r['exit']} loaded {r['loaded']} done {r['done']} fault {r['fault']}")
    for p in r["pre"]: print(f"AL {a}   {p}")
PY
  echo "module banner x$(grep -a -c 'capstone_vm: loaded' $D/board.txt); FRTY $(grep -a -c '^FRTY:' $D/board.txt) DRT1 $(grep -a -c '^DRT1:' $D/board.txt) Oops $(grep -a -c Oops $D/board.txt)" >> $L
  echo "AL_DONE_RUN $(date +%T)" >> $L
}
main "$@"; exit $?
