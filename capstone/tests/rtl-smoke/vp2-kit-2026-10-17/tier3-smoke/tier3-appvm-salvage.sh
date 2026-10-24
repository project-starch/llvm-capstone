#!/bin/bash
# Read a board appvm session whose boot RAN but whose wrapper died before reading it (2026-10-25: the bitstream-2 cpython
# lt1 arm, whose tier3-cpy-b2-arm.sh timeout of 21600 s killed tier3-appvm-arm.sh while the board driver, in its own
# process group, ran on and wrote the transcript). Nothing touches a board: the driver's own transcript is replayed.
#   1. The session's phase must still be "collect", its guest/board.txt must exist and no process may hold it open.
#   2. The session is copied (a real copy, never hardlinks: boot rewrites collect/share/.session-share) to
#      <out root>/session-salvage, without stage/ and guest/, and its config's share path is pointed at the copy.
#   3. appvm-session.py boot runs on the copy with BOARDVM_STUB_TRANSCRIPT = the original board.txt: the same parse
#      (rung_segment, check_params, split) a live boot would have done on the same transcript. The firmware record is
#      then filled from the original guest/firmware.txt, as boardvm.record_firmware does for a live boot.
#   4. The runner reads the copy (--state <copy>/state) into <out root>/out-salvage, as tier3-appvm-arm.sh's read does.
# Usage: tier3-appvm-salvage.sh <arm out root> <runner> <runner args as the arm script received them...>
main() {
  set -u
  local R=${1:?arm out root} RUN=${2:?runner}; shift 2
  local G=/tmp/capstone/wt-gate S=$R/session S2=$R/session-salvage rc
  [ -s $S/guest/board.txt ] || { echo "SALVAGE STOP: no $S/guest/board.txt"; return 1; }
  python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))['session']['phase'] != 'collect')" $S/state/config.json \
    || { echo "SALVAGE STOP: $S is not in phase collect (its boot completed; read it normally)"; return 1; }
  if command -v fuser > /dev/null && fuser $S/guest/board.txt $S/guest/driver.log > /dev/null 2>&1; then
    echo "SALVAGE STOP: a process still holds $S/guest (the driver is running)"; return 1
  fi
  [ -e $S2 ] || [ -e $R/out-salvage ] && { echo "SALVAGE STOP: $S2 or $R/out-salvage exists"; return 1; }
  mkdir -p $S2 && cp -a $S/collect $S/state $S/plan.jsonl $S2/ || { echo "SALVAGE STOP: copy"; return 1; }
  python3 - $S2 <<'PY' || { echo "SALVAGE STOP: config"; return 1; }
import json, sys
from pathlib import Path
s = Path(sys.argv[1]); p = s / "state" / "config.json"; c = json.loads(p.read_text())
c["share"] = str(s / "collect" / "share"); p.write_text(json.dumps(c, indent=2) + "\n")
PY
  BOARDVM_STUB_TRANSCRIPT=$S/guest/board.txt python3 $G/capstone/bug-corpora/tools/board/appvm-session.py boot $S2 > $R/salvage-boot.log 2>&1; rc=$?
  echo "SALVAGE boot-parse rc=$rc :: $(tail -1 $R/salvage-boot.log)"
  [ -s $S2/results.json ] || { echo "SALVAGE STOP: no results.json"; return 1; }
  cp $S/guest/firmware.txt $S2/guest/ 2>/dev/null
  python3 - $S2 <<'PY'
import json, sys
from pathlib import Path
s = Path(sys.argv[1]); p = s / "platform.json"; r = json.loads(p.read_text())
f = s / "guest" / "firmware.txt"
if f.is_file():
    for line in f.read_text().splitlines():
        k, _, v = line.partition(" ")
        if k in ("firmware", "kernel") and v:
            r[k] = v.strip()
p.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
print(f"SALVAGE firmware {r.get('firmware', 'unrecorded')[:16]} kernel {r.get('kernel', 'unrecorded')[:16]}")
PY
  python3 $RUN $R/out-salvage --state $S2/state "$@" --raw $R/raw-salvage > $R/salvage-read.log 2>&1; rc=$?
  echo "APPVM read rc=$rc $(date +%T) :: $(grep -a '^--- ' $R/salvage-read.log | tail -1)"
}
main "$@"; exit $?
