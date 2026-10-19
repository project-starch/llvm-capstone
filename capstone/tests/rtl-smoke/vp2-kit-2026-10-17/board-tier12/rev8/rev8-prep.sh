#!/bin/bash
# rev8's stage and firmware build, after the tree A Linux steps end TAL DONE: step 4 (rev7A) reads the Tier 1 stage against
# rev7's record, so rev8 does not touch the stage before then, and steps 5-6 bake in the same buildroot. Stages with
# node_flat_records=4096 (the compiler lane's QEMU readings; 256 refuses the launcher's own image grant), then builds.
# The firmware hash is pinned in the README before the boot, which run-tier12-rev8.sh then checks.
# Usage: rev8-prep.sh <tree A Linux steps' out> <log>. Ends by itself.
main() {
  set -u
  local W=${1:?TAL out} O=${2:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local end=$(( $(date +%s) + 86400 ))
  until grep -a -q "^TAL \(DONE\|STOP\|HOLD\)" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^TAL DONE" "$W" || { echo "REV8 HOLD: the tree A Linux steps did not end DONE ($(grep -a '^TAL' "$W" | tail -1))" >> "$O"; return 1; }
  echo "REV8 STAGE $(date +%T)" >> "$O"
  bash $S/stage-rev8.sh 4096 >> "$O" 2>&1 || { echo "REV8 STOP: staging failed" >> "$O"; return 1; }
  echo "REV8 BUILD $(date +%T)" >> "$O"
  bash $S/tier12-build-rev8.sh >> "$O" 2>&1
  grep -q TIER12_BUILD_REV8_DONE /tmp/capstone/tier1/BUILD-REV8.log || { echo "REV8 STOP: build failed: $(tail -2 /tmp/capstone/tier1/BUILD-REV8.log | tr '\n' ' ')" >> "$O"; return 1; }
  echo "REV8 BUILT $(date +%T): $(grep '^fw_payload' /tmp/capstone/tier1/BUILD-REV8.log | tail -1)" >> "$O"
  echo "REV8 PREP DONE" >> "$O"
}
main "$@"; exit $?
