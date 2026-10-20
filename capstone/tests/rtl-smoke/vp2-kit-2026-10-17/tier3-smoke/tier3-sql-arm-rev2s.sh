#!/bin/bash
# sql0-dumpB (2026-10-20, BOARD 2, tree B): sql0-nf2B (rev2r) with TWO observation-only changes, declared: the module
# fault-regs-1 0d92d06a (node-floor-2 plus the FREG fatal-fault dump; kits bkit7-rev2xBDF1 / bkit7-rev2xmBDF1) and the image
# DIAG-BTU a2bf0a6ed7c6784b (the recorded postgres build with only nbtinsert.o changed: the duplicate-key ereport gains a HINT
# with both index tuples; compiler lane pgdiag-btu dc9e65694a96). Readings, not scores: the cells are listed as information
# (the image differs from the QEMU record's), every DIAG-BTU HINT and every FREG block with its step. Starts on CHAIN_FR1_GO.
# rev2r's header follows.
# sql0-nf2B (2026-10-20, BOARD 2, tree B): sql0-nf2A (rev2q) with ONE change, the board and bitstream: kits bkit7-rev2xBDN2 /
# bkit7-rev2xmBDN2 (tree B 8de3c48cf, board 2; the same module 54d8a3af and monitor f13dbc5). Scored against the same QEMU
# reference t3-sql-nf2-r1. Reads whether cells 03/07's CREATE EXTENSION failures depend on the bitstream. Starts on
# SQL_NF2B_GO. rev2q's header follows.
# sql0-nf2 SCORED (2026-10-20, BOARD 1, tree A): the first sql0 arm whose cells are graded on silicon. sql0-drt2 (rev2j) with
# the changes that make it scorable: dev's runner 7ce203f9 (the empty-dirs fix as landed; from the dev-pinned
# /tmp/capstone/wt-q15) with its QEMU PASS (t3-sql-nf2-r2/CHECK), kits bkit7-rev2xDN2 / bkit7-rev2xmDN2 (rev2xD/xmD with dev's
# module 54d8a3af: srevroot-zero, the node floor, the release line; Tier 1 PASS on both trees), and the prediction
# t3-sql-nf2-r1/bundle (the compiler lane's QEMU reference on 54d8a3af's twin: CAUGHT 2, MISSED 7, equal to its repeat
# r2). Scored by score-arm.sh. Starts on SQL_NF2A_GO. rev2j's header follows.
# sql0-drt2 (2026-10-19 evening): sql0-drt (rev2g) with ONE change, the runner: run-arm.py 8826a2ca21a33acd (lane sql-empty-dirs:
# each session recreates the cluster's 13 empty directories the board bake drops). Kits and monitor are sql0-drt's (D,
# f13dbc5). The runner's gate is the runtime lane's ruling: (a) the static command diff (records/runner-8826a2ca-static-diff.txt,
# 113 of 113 lines identical but the mkdir clauses) instead of a QEMU check, which cannot exercise the change; (b) this boot's
# must-fire: no directory-open FATAL in any session (narrowed 2026-10-20, see README) and the positive control answered. Cells are graded against the
# single QEMU record, a mismatch reported as 'possibly variance' (sql0 on QEMU is not deterministic). Starts on
# CHAIN_DRT2_GO. rev2g's header follows.
# sql0-drt (2026-10-19 evening): sql0-rf (rev2f) with ONE change, the monitor: kits bkit7-rev2xD/bkit7-rev2xmD are the
# refusal-read kits on capstone-sbi f13dbc5 (03a1ffe + CAPSTONE_S23_DATA_RETRY: S-23b step 1, DRT1/DRTX) (make-drt-kits.sh).
# It starts on CHAIN_DRT_SQL_GO. rev2f's header follows.
# sql0-rf (2026-10-19 evening): sql0-fix (rev2e) with ONE change, the monitor: kits bkit7-rev2xR/bkit7-rev2xmR are the fix
# kits on capstone-sbi 03a1ffe (5b5e86b + CAPSTONE_S23_REFUSAL_READ: RFSL/RFTY/RFVD at a refusal) (make-rf-kits.sh). It starts
# on CHAIN_RF_GO. rev2e's header follows.
# sql0-fix (2026-10-19 evening): sql0-treeA (rev2d) with ONE change, the monitor: kits bkit7-rev2xF/bkit7-rev2xmF are
# bkit7-rev2xA/bkit7-rev2xmA on capstone-sbi 5b5e86b (S-23a's guarded strobe, FRGS/FRGU/FRGD) and wrapper aab63d5 with
# CAPSTONE_QUANTUM_PRINT and CAPSTONE_BOOT_QUANTUM=2000000 (make-fix-kits.sh). It runs only after the monitor's canary
# (run-s26.sh fix1) judged PASS: the chain writes CHAIN_FIX1_PASS. rev2d's header follows.
# sql0-rev2d (2026-10-19): rev2c on tree A (caplifive_vp2c_8522d72fe) with the kit's execute-bit fix. rev2c's boot
# failed in every session at postgres's own check of its binary, which the kit had baked 0644. Kits bkit7-rev2xA and
# bkit7-rev2xmA are bkit7-rev2s and bkit7-rev2m with the tree A bitstream and one recipe line: a staged file whose source
# is executable keeps 0755. The runner, inputs, size gate, prediction and gates are rev2c's. It runs after the tree A
# Linux steps end, and only on TAL DONE (a STOP or HOLD there holds it). rev2c's header follows.
# Tier 3 postgres/sql-repros vp2-malloc on module revision 39d8492d, third form (2026-10-19): rev2b's inputs and kits
# unchanged, with sql run-arm.py's board path fixed. rev2b's boot ran 21/21 sessions and every one failed at
# `cp -a /mnt/vm/pgcluster` (the QEMU kit's mount; the board stages under /test-domains/vbatch), and the bake's 0775
# directories would have failed postgres's data-directory check next. The runner now takes the cluster path from the
# batch and chmods the copy 700. It runs only if the runner is that fix (sha256/16 below) and the QEMU check of it
# (<mnt-hdd>/capstone/t3-sql-trim-vm-r2/CHECK) reads PASS for the same hash.
#
# rev2b's header follows.
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
# Usage: tier3-sql-arm-rev2d.sh <after> <out>: runs once <after> (the tree A Linux steps' out) reads TAL DONE.
# SQL_RUN / SQL_TAG / SQL_BUSY / SQL_RUNNER replace the run dir, the tag, the board-1 busy check (board-busy.py 1) and the
# runner for the stubbed dry run. Ends by itself.
main() {
  set -u
  local W=${1:?after} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2xBDF1 LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local IMG=/mnt/hdd/$USER/capstone/t3-cw/postgres-diag/image/postgres.dom FIX=/mnt/hdd/$USER/capstone/t3-new6/tmp/pgfixture-trim/cluster
  local SHARE=/mnt/hdd/$USER/capstone/t3-new6/tmp/pgfixture/install/share/postgresql KM=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2xmBDF1
  local P=$H/t3-sql-nf2-r1/bundle TAG=${SQL_TAG:-sql0-dumpB} R=${SQL_RUN:-$H/t3-run-treeB/sql0-dumpB}
  local RUN=${SQL_RUNNER:-/tmp/capstone/wt-q15/capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py} end=$(( $(date +%s) + 86400 )) rc
  until grep -a -q "^CHAIN_FR1_GO\|^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CHAIN_FR1_GO" "$W" || { echo "SQL HOLD: no CHAIN_FR1_GO ($(tail -1 "$W"))" >> "$O"; return 1; }
  local k; for k in $K $KM; do
    grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf" $k/board.env && grep -qx "FPGA_BOARD=2" $k/board.env && grep -qx "BOARD_MONITOR_COMMIT=f13dbc5" $k/board.env && grep -q 'if \[ -x "$STAGE/$f" \]; then m=0755' $k/board-run.sh \
      || { echo "SQL STOP: $k is not a tree B board-2 kit with the execute-bit rule" >> "$O"; return 1; }
  done
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 0d92d06a2abc6298 ] || { echo "SQL STOP: the kit's module is not 0d92d06a" >> "$O"; return 1; }
  grep -qx "BOARD_TIMEOUT_MULT=4" $K/board.env && ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "SQL STOP: the kit is not bkit7-rev2xBDF1 as recorded" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = a2bf0a6ed7c6784b ] || { echo "SQL STOP: the image is not a2bf0a6e" >> "$O"; return 1; }
  [ "$(sha256sum $KM/adapter/module/capstone_vm.ko | cut -c1-16)" = 0d92d06a2abc6298 ] && grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "SQL STOP: the measurement kit is not bkit7-rev2xmBDF1" >> "$O"; return 1; }
  local RH; RH=$(sha256sum "$RUN" | cut -c1-16)
  [ "$RH" = "${SQL_RUNNER_SHA:-7ce203f9fd6a3cce}" ] || { echo "SQL STOP: the runner is $RH, not dev's 7ce203f9" >> "$O"; return 1; }
  grep -qx "PASS $RH" "${SQL_QCHECK:-$H/t3-sql-nf2-r2/CHECK}" 2>/dev/null || { echo "SQL STOP: no QEMU PASS for runner $RH" >> "$O"; return 1; }
  [ -d $FIX ] && [ -d "$P" ] || { echo "SQL STOP: no fixture or no prediction bundle" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "SQL STOP: $R exists" >> "$O"; return 1; }
  ${SQL_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "SQL STOP: board 2 is busy" >> "$O"; return 1; }
  mkdir -p "$R/tmp" "$R/measure/tmp"
  # 1. the measure-only bake: the same invocation on kit bkit7-rev2m, which stops after the firmware build
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/measure/tmp taskset -c 16-31 timeout 3600 python3 "$RUN" --arm virtual-malloc --board-kit $KM \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/measure/out" --raw "$R/measure/raw" ) > "$R/measure/run.log" 2>&1
  local sz; sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" "$R/measure/raw/guest/SEQ" 2>/dev/null | grep -o "[0-9]\+" | head -1)
  echo "SQL MEASURE: firmware ${sz:-NONE} B against 26000000 ($(grep -a -o 'manifest [0-9]* files, stage [0-9]* bytes' "$R/measure/raw/guest/SEQ" 2>/dev/null))" >> "$O"
  [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "SQL STOP: the measured firmware is not under the cap; no boot" >> "$O"; return 1; }
  ${SQL_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "SQL STOP: board 2 is busy" >> "$O"; return 1; }
  echo "T3PG $TAG BEGIN $(date +%T) (trigger: $(head -1 "$W"))" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/tmp taskset -c 16-31 timeout 36000 python3 "$RUN" --arm virtual-malloc --board-kit $K \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/out" --raw "$R/raw" ) > "$R/run.log" 2>&1; rc=$?
  echo "T3PG $TAG END $(date +%T) rc=$rc :: $(grep -a -o 'virtual boot: [A-Z a-z]*' "$R/run.log" | head -1) :: $(grep -a '^--- ' "$R/run.log" | tail -1)" >> "$O"
  local fr im
  read -r fr im < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" "$R/out/inputs.json" 2>/dev/null)
  if [ "${fr:-}" != 956173e054762e91 ] || [ "${im:-}" != a2bf0a6ed7c6784b ]; then
    echo "SQL VOID: fixture record ${fr:-none} (want 956173e054762e91), image ${im:-none} (want a2bf0a6ed7c6784b)" >> "$O"; return 1
  fi
  echo "SQL inputs as recorded: fixture $fr, image $im" >> "$O"
  # (b) the must-fire: no session stopped on a directory-open FATAL, and the positive control answered. Its controls: a planted
  # pg_notify FATAL must trip it, and the passing QEMU record's transcript (t3-sql-trim-vm-r2) must not.
  local B=$R/raw/guest/board.txt nd
  nd=$(grep -a -c "FATAL: .*could not open directory" $B 2>/dev/null)   # a FATAL only (narrowed 2026-10-20, the runtime lane's ruling: a zoneinfo LOG is in every QEMU run)
  if [ "${nd:-x}" != 0 ] || grep -a -q "POSITIVE CONTROL FAILED" "$R/run.log"; then
    echo "SQL MUST-FIRE FAIL: directory-open FATAL x${nd:-NO-DATA}, positive control $(grep -a -c 'POSITIVE CONTROL FAILED' $R/run.log) failure line(s)" >> "$O"; return 1
  fi
  echo "SQL MUST-FIRE OK: no directory-open FATAL in any session; positive control answered" >> "$O"
  echo "SQL S-23: FRTY $(grep -a -c '^FRTY:' $B) DRT1 $(grep -a -c '^DRT1:' $B) DRTX $(grep -a -c '^DRTX:' $B) Oops $(grep -a -c Oops $B); banner fault-regs-1 x$(grep -a -c 'fault-regs-1' $B); release lines $(grep -a -o '\[ *[0-9.]*\] capstone_vm: release retires[^ ]* [^ ]* [^ ]*' $B | sort -u | wc -l), with succeeded < attempted $(grep -a -o 'release retires attempted=[0-9]* succeeded=[0-9]*' $B | awk -F'[= ]' '$4!=$6' | wc -l)" >> "$O"
  python3 "$S/tier3-cells.py" "$R/out" 2>/dev/null | sed 's/^/SQL CELL /' >> "$O"
  echo "SQL DIAG-BTU hints $(grep -a -c 'DIAG-BTU' $B 2>/dev/null)" >> "$O"
  awk '/@@STEP-BEGIN/{s=$2} /DIAG-BTU/{sub(/.*HINT: */,""); print "SQL DIAG-BTU " s " :: " $0}' $B | cut -c1-400 >> "$O"
  echo "SQL FREG blocks $(grep -a -c 'capstone_vm: FREG thread=' $B 2>/dev/null)" >> "$O"
  awk '/@@STEP-BEGIN/{s=$2} /capstone_vm: FREG/{sub(/.*capstone_vm: /,""); print "SQL FREG " s " :: " $0}' $B | cut -c1-260 >> "$O"
  echo "SQL DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
