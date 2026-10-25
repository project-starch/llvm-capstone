#!/bin/bash
# recover-appvm-read.sh <out root> <runner> <runner args...>: finish an appvm arm whose WRAPPER was killed (tier3-appvm-arm.sh
# under a `timeout` too short for the run) while the board driver ran on to the end (2026-10-20, cpython lt0 on board 2).
# Nothing boots: the board's own transcript, written by the driver, is parsed by the SAME code the killed boot phase would
# have run (appvm-session.py boot with BOARDVM_STUB_TRANSCRIPT, boardvm.py's documented dry-run path), then the read phase
# runs as tier3-appvm-arm.sh would. Refuses unless the driver's transcript holds the vbatch END line (the run completed) and
# the session is still in phase collect (the killed boot never advanced it). The original guest dir is kept as guest.board,
# and its firmware.txt hashes are restored into platform.json (the stub's fresh guest dir has none).
main() {
  set -u
  local R=${1:?out root} RUN=${2:?runner}; shift 2
  local S=$R/session G=/tmp/capstone/wt-gate
  local B=$S/guest/board.txt
  [ -s "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "RECOVER STOP: no completed transcript at $B"; return 1; }
  grep -q '"phase": "collect"' $S/state/config.json || { echo "RECOVER STOP: the session is not in phase collect"; return 1; }
  [ -e $S/guest.board ] || [ -e $S/stage.board ] && { echo "RECOVER STOP: $S/guest.board or stage.board exists"; return 1; }
  # the boot phase stages into a fresh stage dir (rebuilt from session/collect) and a fresh guest dir: both move aside, kept
  mv $S/guest $S/guest.board && mv $S/stage $S/stage.board || { echo "RECOVER STOP: could not move guest/stage aside"; return 1; }
  local rc
  BOARDVM_STUB_TRANSCRIPT=$S/guest.board/board.txt python3 $G/capstone/bug-corpora/tools/board/appvm-session.py boot $S > $R/boot-recovered.log 2>&1; rc=$?
  echo "RECOVER boot-parse rc=$rc :: $(tail -1 $R/boot-recovered.log)"
  [ $rc = 0 ] || { echo "RECOVER STOP: the parse did not complete; no read phase"; return 1; }
  python3 - $S/platform.json $S/guest.board/firmware.txt <<'PY'
import json, sys
p, f = sys.argv[1], sys.argv[2]
d = json.load(open(p)); n = 0
for line in open(f).read().splitlines():
    k, _, v = line.partition(" ")
    if k in ("firmware", "kernel") and v: d[k] = v.strip(); n += 1
json.dump(d, open(p, "w"), indent=2, sort_keys=True); open(p, "a").write("\n")
print(f"RECOVER platform.json: {n} hashes restored from firmware.txt")
PY
  python3 $RUN $R/out --state $S/state "$@" --raw $R/raw-read > $R/read.log 2>&1
  echo "RECOVER read rc=$? :: $(grep -a '^--- ' $R/read.log | tail -1)"
}
main "$@"; exit $?
