#!/bin/bash
# Tier 3 postgres/sql-repros vp2-malloc on the S-27 workaround module (2026-10-18): the compiler lane's lt0 image
# (postgres.dom 81ee24a0a1e957c9, CPONVVM6) and native16 fixture through sql-repros' run-arm.py on kit bkit7-fix6s, one
# boot of 21 single-user postgres sessions. bkit7-fix6s is bkit7-fix6 with BOARD_TIMEOUT_MULT=4: the runner budgets
# 600 + 300 s a session, and on vp2 a step runs about 18 times its QEMU time (the QEMU record took about 47 s a session).
# Scored against the QEMU record t3-sql-virtual-malloc/bundle (qkit7, module 5bfc38f8). After the boot the bundle's
# fixture record must be 173aa5b35c1b57b9 (run-arm.py's own hash of the tree) and its image the record's, or the arm
# is VOID. Runs after the fix6b sequence ends. Usage: tier3-sql-arm.sh <fix6b out> <out>. Ends by itself.
# SQL_RUN / SQL_TAG / SQL_DRVPAT / SQL_RUNNER replace the run dir, the tag, the running-driver guard's pattern and the
# runner for the stubbed dry run.
main() {
  set -u
  local W=${1:?fix6b out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6s LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local IMG=/mnt/hdd/$USER/capstone/t3-new6/postgres-lt0/image/postgres.dom FIX=/mnt/hdd/$USER/capstone/t3-new6/tmp/pgfixture/cluster
  local P=$H/t3-sql-virtual-malloc/bundle TAG=${SQL_TAG:-sql0-fix6} R=${SQL_RUN:-$H/t3-run-fix6/sql0-fix6}
  local RUN=${SQL_RUNNER:-/tmp/capstone/wt-gate/capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py} end=$(( $(date +%s) + 86400 )) rc
  until grep -a -q "^FIX6BSEQ DONE\|^FIX6BSEQ HOLD\|^FIX6BSEQ STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "SQL STOP: the kit's module is not d71f207a" >> "$O"; return 1; }
  grep -qx "BOARD_TIMEOUT_MULT=4" $K/board.env && ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "SQL STOP: the kit is not bkit7-fix6s as recorded" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = 81ee24a0a1e957c9 ] || { echo "SQL STOP: the image is not 81ee24a0" >> "$O"; return 1; }
  [ -d $FIX ] && [ -d "$P" ] || { echo "SQL STOP: no fixture or no prediction bundle" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "SQL STOP: $R exists" >> "$O"; return 1; }
  pgrep -f "${SQL_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "SQL STOP: a driver is running" >> "$O"; return 1; }
  mkdir -p "$R/tmp"
  echo "T3PG $TAG BEGIN $(date +%T) (after: $(grep -a '^FIX6BSEQ' "$W" | tail -1))" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/tmp taskset -c 16-31 timeout 36000 python3 "$RUN" --arm virtual-malloc --board-kit $K \
      --image $IMG --fixture $FIX --llvm-bin $LB --out "$R/out" --raw "$R/raw" ) > "$R/run.log" 2>&1; rc=$?
  echo "T3PG $TAG END $(date +%T) rc=$rc :: $(grep -a -o 'virtual boot: [A-Z a-z]*' "$R/run.log" | head -1) :: $(grep -a '^--- ' "$R/run.log" | tail -1)" >> "$O"
  local fr im
  read -r fr im < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" "$R/out/inputs.json" 2>/dev/null)
  if [ "${fr:-}" != 173aa5b35c1b57b9 ] || [ "${im:-}" != 81ee24a0a1e957c9 ]; then
    echo "SQL VOID: fixture record ${fr:-none} (want 173aa5b35c1b57b9), image ${im:-none} (want 81ee24a0a1e957c9)" >> "$O"; return 1
  fi
  echo "SQL inputs as recorded: fixture $fr, image $im" >> "$O"
  KIT_DESC="kit bkit7-fix6s (module d71f207a3af5546b, the S-27 workaround; batch budget x4), pin off" \
    RESULTS_DIR=2026-10-18-vp2-rdx0 bash "$S/score-arm.sh" "$TAG" "$P" "$R" "$O" >> "$O" 2>&1
  echo "SQL DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
