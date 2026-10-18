#!/bin/bash
# Tier 3 postgres/sql-repros vp2-malloc on module revision 39d8492d, second form (2026-10-19). The first form stopped at
# the firmware cap (30,050,824 B over 26,000,000), so: the image is lt0 with --discard-locals (64010720f1a6b3bc; loaded
# bytes, the Capstone sections and every non-.L symbol identical to 81ee24a0); the fixture is the native16 cluster
# trimmed (no template0/template1, a 1 MiB WAL segment; run-arm's record 956173e054762e91); --pg-share is the share the
# native16 record used. One boot of 21 single-user postgres sessions through sql-repros' run-arm.py.
# FIRST a measure-only bake on kit bkit7-rev2m (bkit7-rev2s with a stop after the firmware build; no board access); the
# board boot on kit bkit7-rev2s (module 39d8492d, BOARD_TIMEOUT_MULT=4) runs only if that firmware is under 26,000,000 B.
# Scored against t3-sql-trim-vm/bundle, the QEMU record on these exact inputs (9 of 9 cells equal to the native16 record,
# evidence text included). After the boot the bundle's fixture record must be 956173e054762e91 and its image
# 64010720f1a6b3bc, or the arm is VOID.
# Usage: tier3-sql-arm-rev2b.sh <after> <out>: runs once <after> (the postgres chain's out) reads PGF DONE/STOP/HOLD.
# SQL_RUN / SQL_TAG / SQL_DRVPAT / SQL_RUNNER replace the run dir, the tag, the running-driver guard's pattern and the
# runner for the stubbed dry run. Ends by itself.
main() {
  set -u
  local W=${1:?after} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2s LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local IMG=/mnt/hdd/$USER/capstone/t3-new6/postgres-lt0-dl/image/postgres.dom FIX=/mnt/hdd/$USER/capstone/t3-new6/tmp/pgfixture-trim/cluster
  local SHARE=/mnt/hdd/$USER/capstone/t3-new6/tmp/pgfixture/install/share/postgresql KM=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2m
  local P=$H/t3-sql-trim-vm/bundle TAG=${SQL_TAG:-sql0-rev2b} R=${SQL_RUN:-$H/t3-run-rev2/sql0-rev2b}
  local RUN=${SQL_RUNNER:-/tmp/capstone/wt-gate/capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py} end=$(( $(date +%s) + 86400 )) rc
  until grep -a -q "^PGF DONE\|^PGF STOP\|^PGF HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^PGF" "$W" || { echo "SQL HOLD: the postgres chain did not end" >> "$O"; return 1; }
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] || { echo "SQL STOP: the kit's module is not 39d8492d" >> "$O"; return 1; }
  grep -qx "BOARD_TIMEOUT_MULT=4" $K/board.env && ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "SQL STOP: the kit is not bkit7-rev2s as recorded" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = 64010720f1a6b3bc ] || { echo "SQL STOP: the image is not 64010720" >> "$O"; return 1; }
  [ "$(sha256sum $KM/adapter/module/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] && grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "SQL STOP: the measurement kit is not bkit7-rev2m" >> "$O"; return 1; }
  [ -d $FIX ] && [ -d "$P" ] || { echo "SQL STOP: no fixture or no prediction bundle" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "SQL STOP: $R exists" >> "$O"; return 1; }
  pgrep -f "${SQL_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "SQL STOP: a driver is running" >> "$O"; return 1; }
  mkdir -p "$R/tmp" "$R/measure/tmp"
  # 1. the measure-only bake: the same invocation on kit bkit7-rev2m, which stops after the firmware build
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/measure/tmp taskset -c 16-31 timeout 3600 python3 "$RUN" --arm virtual-malloc --board-kit $KM \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/measure/out" --raw "$R/measure/raw" ) > "$R/measure/run.log" 2>&1
  local sz; sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" "$R/measure/raw/guest/SEQ" 2>/dev/null | grep -o "[0-9]\+" | head -1)
  echo "SQL MEASURE: firmware ${sz:-NONE} B against 26000000 ($(grep -a -o 'manifest [0-9]* files, stage [0-9]* bytes' "$R/measure/raw/guest/SEQ" 2>/dev/null))" >> "$O"
  [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "SQL STOP: the measured firmware is not under the cap; no boot" >> "$O"; return 1; }
  pgrep -f "${SQL_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "SQL STOP: a driver is running" >> "$O"; return 1; }
  echo "T3PG $TAG BEGIN $(date +%T) (trigger: $(head -1 "$W"))" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/tmp taskset -c 16-31 timeout 36000 python3 "$RUN" --arm virtual-malloc --board-kit $K \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/out" --raw "$R/raw" ) > "$R/run.log" 2>&1; rc=$?
  echo "T3PG $TAG END $(date +%T) rc=$rc :: $(grep -a -o 'virtual boot: [A-Z a-z]*' "$R/run.log" | head -1) :: $(grep -a '^--- ' "$R/run.log" | tail -1)" >> "$O"
  local fr im
  read -r fr im < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" "$R/out/inputs.json" 2>/dev/null)
  if [ "${fr:-}" != 956173e054762e91 ] || [ "${im:-}" != 64010720f1a6b3bc ]; then
    echo "SQL VOID: fixture record ${fr:-none} (want 956173e054762e91), image ${im:-none} (want 64010720f1a6b3bc)" >> "$O"; return 1
  fi
  echo "SQL inputs as recorded: fixture $fr, image $im" >> "$O"
  KIT_DESC="kit bkit7-rev2s (module 39d8492d5c18216b, the five-item revision; batch budget x4), pin off" \
    RESULTS_DIR=2026-10-19-vp2-rev2 bash "$S/score-arm.sh" "$TAG" "$P" "$R" "$O" >> "$O" 2>&1
  echo "SQL DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
