#!/bin/bash
# B's check boot 3 on board 2 (PREREG Addendum 17): b4-t1c, run.py@69fb6cae3db4's Tier 1 SCRIPT with the module's
# resolve_dirty=1 (the S-24 fix, the module owner's decision), then a clock check and a spin pair; no flash.
# Each step gates the next:
# 1. every input by hash: the stage's run.sh (gen-t1c.py), the kit (b4r-boot.sh checks it again), the twin reference
#    F0r, and run.py@69fb6cae3db4 extracted from its commit (not the worktree file, which can move);
# 2. board 2 holds bitstream 4: nv-readback.py reads the resident record by name on a fresh session (read-only);
# 3. the boot: b4r-boot.sh 2 <stage> <out>/t1c 5400 120 (t1b's step ran ~46 min, 91 heartbeats; boot timeout
#    = step + 600 s, inside the 120 min booking);
# 4. the judge: b4-check-t1c.py (exit 0 as predicted, 1 off, 2 no data, 3 read-back failed).
# Usage: b4-t1c.sh <log>. TC_* stub every external step for the dry run (TC_NVRB, TC_BOOT, TC_OUT).
main() {
  set -u
  local O=${1:?log} BRD=2 NAME=caplifive_vp2c_325966ee4 H=/mnt/hdd/$USER/capstone S rc r
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local SHA=f0893218dcdb53f4ad0b9fe452571f9e5c3bc5fa7ffade0025657c662c4da8cc
  local OUT=${TC_OUT:-$H/spec/b4-t1c} D=$H/spec/stages-directed/b4-t1c F0R=$H/f55/t1-F0r/result.json
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  export CAPSTONE_LLVM_BIN=$H/cc-prmerged/bin
  [ -e "$OUT" ] && { echo "TC STOP: $OUT exists" >> "$O"; return 2; }
  mkdir -p "$OUT"
  local RP=$OUT/run.py.69fb6cae3db4
  git -C $H/wt-t1w55 show 69fb6cae3db4:capstone/runtime/virtual/run.py > "$RP" 2>/dev/null \
    && [ "$(sha256sum "$RP" | cut -c1-16)" = 11c47d62fe9f34ae ] || { echo "TC STOP: run.py@69fb6cae3db4" >> "$O"; return 2; }
  [ "$(sha256sum "$D/stage/run.sh" | cut -c1-16)" = df5b86a3fea4a245 ] && [ -f "$D/plan.json" ] \
    || { echo "TC STOP: stage $D is not gen-t1c.py's" >> "$O"; return 2; }
  [ "$(sha256sum "$F0R" | cut -c1-16)" = b7acfc11080f400c ] || { echo "TC STOP: twin reference F0r" >> "$O"; return 2; }
  [ -x "$CAPSTONE_LLVM_BIN/llvm-nm" ] || { echo "TC STOP: no llvm-nm under CAPSTONE_LLVM_BIN" >> "$O"; return 2; }
  echo "TC inputs checked $(date +%T): run.py 11c47d62fe9f34ae, run.sh df5b86a3fea4a245, F0r b7acfc11080f400c" >> "$O"
  ${TC_NVRB:-python3 $S/nv-readback.py} $NAME.bit $SHA >> "$O" 2>&1; rc=$?
  case $rc in
    0|3) echo "TC RESIDENT board 2 holds $NAME.bit (rc=$rc)" >> "$O";;
    *) echo "TC STOP: board 2's resident bitstream is not $NAME.bit (readback rc=$rc)" >> "$O"; return 1;;
  esac
  echo "TC BOOT BEGIN $(date +%T)" >> "$O"
  ${TC_BOOT:-bash $S/b4r-boot.sh} $BRD "$D" "$OUT/t1c" 5400 120 >> "$O" 2>&1; echo "TC BOOT END rc=$? $(date +%T)" >> "$O"
  python3 $S/b4-check-t1c.py "$OUT/t1c/spec.txt" "$RP" "$D/stage/contract.dom" "$F0R" "$OUT/t1c/guest/board.txt" \
    > "$OUT/t1c.judge" 2>&1; r=$?
  sed 's/^/  /' "$OUT/t1c.judge" >> "$O"; echo "TC JUDGE rc=$r" >> "$O"
  echo "TC DONE $(date +%T)" >> "$O"
  return $r
}
main "$@"; exit $?
