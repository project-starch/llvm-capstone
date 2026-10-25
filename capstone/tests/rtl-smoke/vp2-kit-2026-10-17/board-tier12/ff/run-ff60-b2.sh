#!/bin/bash
# The fetch-face sweep on BOARD 2, tree B (2026-10-20; pre-registered in tier3-smoke/README.md before the boot): run-ff-b2.sh
# with firmware fw-ff60 (image-ff60: the pair's stage with the case ff-sweep, INNER 300, node_flat_records 65536; the
# fetch-slot monitor 9697173) and the rung ff-sweep. Per arm it also reports the whole FF_DONE line, the nodes, and every
# (FRSL, FRT2, FRVD) tuple. Usage: run-ff60-b2.sh. Ends by itself. FF_* stub.
DRV_DIR=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
main() {
  set -u
  local D=${FF_OUT:-/tmp/capstone/s26/run-ff60-b2} FW=${FF_FW:-/tmp/capstone/s26/fw-ff60/fw_payload.bin} O=/tmp/capstone/s26/oracles-ff60 r
  [ "$(sha256sum $FW | cut -c1-16)" = 1b7e827176895824 ] || { echo "STOP: the firmware is not 1b7e827176895824"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: driver"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists"; return 1; }
  mkdir -p $D $O; for r in b0-stats b0-hello ff-sweep; do echo 0 > $O/$r.oracle; done; local L=$D/SEQ
  ${FF_WAIT:-bash $DRV_DIR/board-free-wait.sh 21600 2} || { echo "STOP: board 2 stayed busy" >> $L; return 1; }
  echo "FF_START $(date +%T) fw 1b7e827176895824 bitstream caplifive_vp2c_8de3c48cf board 2" >> $L
  cd $DRV_DIR || return 1
  FPGA_BOARD=2 FPGA_BITSTREAM=caplifive_vp2c_8de3c48cf.bit FPGA_URL="$(cat $HOME/.claude-kisp/secrets/fpga-console-url-board2)" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello ff-sweep" LADDER_FPGA_DIR=$O \
    BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
    timeout 9000 ${FF_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  FPGA_BOARD=2 ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!; wait $runner; local rc=$?; sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  for r in b0-stats b0-hello ff-sweep; do
    echo "RUNG $r retval=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')" >> $L
  done
  python3 - $D/board.txt >> $L <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", "").split("\n")
arm = None; rows = {}
for l in t:
    m = re.match(r"^FF_ARM (\S+)", l)
    if m: arm = m.group(1); rows[arm] = {"FRTY": 0, "DRT1": 0, "DRTX": 0, "steps": None, "nodes": None, "done": None, "exit": None, "frs": [], "cur": []}; continue
    if arm is None: continue
    for k in ("FRTY", "DRT1", "DRTX"):
        if l.startswith(k + ":"): rows[arm][k] += 1
    m = re.match(r"^(FRSL|FRT2|FRVD):([0-9a-fA-F]+)", l)
    if m:
        rows[arm]["cur"].append(m.group(2))
        if m.group(1) == "FRVD": rows[arm]["frs"].append(" ".join(rows[arm]["cur"][-3:])); rows[arm]["cur"] = []
    m = re.search(r"CAPSTONE_VM_STATS .*nodes=(\d+) steps=(\d+)", l)
    if m: rows[arm]["nodes"] = int(m.group(1)); rows[arm]["steps"] = int(m.group(2))
    m = re.search(r"FF_DONE (.*)$", l)
    if m: rows[arm]["done"] = m.group(1).strip()
    m = re.match(r"^FF_EXIT (\S+) (\d+)", l)
    if m: rows[arm]["exit"] = int(m.group(2)); arm = None
if not rows: print("FF NO DATA: no FF_ARM marker"); sys.exit(0)
import collections
for a, r in rows.items():
    print(f"FF {a}: exit {r['exit']} steps {r['steps']} nodes {r['nodes']} [{r['done']}] FRTY {r['FRTY']} DRT1 {r['DRT1']} DRTX {r['DRTX']} FRSL-tuples {len(r['frs'])}")
    for k, n in collections.Counter(r["frs"]).most_common(): print(f"FF {a}:   (FRSL FRT2 FRVD) {k} x{n}")
    if r["FRTY"] > 0 and not r["frs"]: print(f"FF {a}: MUST-FIRE FAILED: FRTY > 0 with no FRSL line")
PY
  echo "node_flat_records $(grep -a -o '@@PARAM node_flat_records=[0-9A-Z]*' $D/board.txt | tail -1); srz banner x$(grep -a -c 'diag srevroot-zero-1' $D/board.txt); Oops $(grep -a -c Oops $D/board.txt)" >> $L
  echo "FF_DONE_RUN $(date +%T)" >> $L
}
main "$@"; exit $?
