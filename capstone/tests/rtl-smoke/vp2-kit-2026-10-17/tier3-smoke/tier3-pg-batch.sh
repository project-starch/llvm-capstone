#!/bin/bash
# postgres sql-repros pg-pools (lt1) on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md before its
# boot). tier3-sql-arm-rev2u.sh's sql0 path with what the batch arm changes:
#   - kits: the flashed board's batch kit bkit8-batch-b<board> and its MEASURE-ONLY twin bkit8-batch-mb<board>
#     (make-batch-kit.sh, MK_M=1): c721c1a2a, batch module 21fbfc39, monitor d9da397 + FETCH_RETRY_CAP;
#   - --arm virtual-pg-pools, image t3-images-r60/q15/postgres-lt1 (dc2a1356dce735e0);
#   - the trimmed fixture and share the sql0 board path uses, which the compiler lane's batch reference also ran:
#     pgp-trim/bundle on qkit7-batch2 (fixture record 956173e054762e91, image dc2a1356; 9 of 9 cells equal to the untrimmed
#     batch reference; controls fault; SWITCHES movc_null_scalar=OFF, the batch mimic);
#   - dev's runner 7ce203f9 from wt-q15, with its QEMU PASS (t3-sql-nf2-r2/CHECK), as the reference ran;
#   - scored by score-arm.sh into a NEW results directory named for the bitstream (RESULTS_DIR 2026-10-22-batch-c721c1a2a-b<board>).
# First the measure-only bake: the boot runs only if the batch firmware with this payload is under 26,000,000 B.
# Usage: tier3-pg-batch.sh <board 1|2> <out log>. PG_RUNNER / PG_RUNNER_SHA / PG_QCHECK / PG_BUSY / PG_ROOT / PG_TAG / PG_SCORE
# stub the dry run (PG_SCORE keeps the dry run from filing anything into the corpus's results).
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local K=$H/t3-smoke/bkit8-batch-b$BRD KM=$H/t3-smoke/bkit8-batch-mb$BRD
  local IMG=$H/t3-images-r60/q15/postgres-lt1/image/postgres.dom FIX=$H/t3-new6/tmp/pgfixture-trim/cluster
  local SHARE=$H/t3-new6/tmp/pgfixture/install/share/postgresql P=$H/pgp-trim/bundle ST=$H/pgp-trim/pg-pools-trim.stamp
  local TAG=${PG_TAG:-batch-pg-pools-b$BRD} R=${PG_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/pg-pools} BUSY rc
  local RUN=${PG_RUNNER:-/tmp/capstone/wt-q15/capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py}
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PG STOP: board $BRD" >> "$O"; return 2;;
  esac
  local k; for k in $K $KM; do
    ( cd $k && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $k/board.env \
      && [ "$(sha256sum $k/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] && grep -q 'if \[ -x "$STAGE/$f" \]; then m=0755' $k/board-run.sh \
      || { echo "PG STOP: $k is not a batch kit with the execute-bit rule" >> "$O"; return 1; }
  done
  grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "PG STOP: $KM is not the measure-only kit" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = dc2a1356dce735e0 ] || { echo "PG STOP: the image is not dc2a1356" >> "$O"; return 1; }
  grep -q "^STAMP pg-pools-trim runner 7ce203f9fd6a3cce .*image dc2a1356dce735e0 fixture pgfixture-trim/cluster" $ST \
    && grep -q "SWITCHES movc_null_scalar=OFF compact_rounding=ON" $ST && [ -f $P/inputs.json ] || { echo "PG STOP: the reference's stamp, switches or bundle" >> "$O"; return 1; }
  [ "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" $P/inputs.json)" = "956173e054762e91 dc2a1356dce735e0" ] \
    || { echo "PG STOP: the reference bundle's fixture or image record" >> "$O"; return 1; }
  local RH; RH=$(sha256sum "$RUN" | cut -c1-16)
  [ "$RH" = "${PG_RUNNER_SHA:-7ce203f9fd6a3cce}" ] || { echo "PG STOP: the runner is $RH, not dev's 7ce203f9" >> "$O"; return 1; }
  grep -qx "PASS $RH" "${PG_QCHECK:-$H/t3-sql-nf2-r2/CHECK}" 2>/dev/null || { echo "PG STOP: no QEMU PASS for runner $RH" >> "$O"; return 1; }
  [ -d $FIX ] || { echo "PG STOP: no fixture" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "PG STOP: $R exists" >> "$O"; return 1; }
  ${PG_BUSY:-$BUSY} > /dev/null || { echo "PG STOP: board $BRD is busy" >> "$O"; return 1; }
  mkdir -p "$R/tmp" "$R/measure/tmp"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/measure/tmp taskset -c 16-31 timeout 3600 python3 "$RUN" --arm virtual-pg-pools --board-kit $KM \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/measure/out" --raw "$R/measure/raw" ) > "$R/measure/run.log" 2>&1
  local sz; sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" "$R/measure/raw/guest/SEQ" 2>/dev/null | grep -o "[0-9]\+" | head -1)
  echo "PG MEASURE: firmware ${sz:-NONE} B against 26000000" >> "$O"
  [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "PG STOP: the measured firmware is not under the cap; no boot" >> "$O"; return 1; }
  ${PG_BUSY:-$BUSY} > /dev/null || { echo "PG STOP: board $BRD is busy" >> "$O"; return 1; }
  echo "T3PG $TAG BEGIN $(date +%T)" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/tmp taskset -c 16-31 timeout 36000 python3 "$RUN" --arm virtual-pg-pools --board-kit $K \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/out" --raw "$R/raw" ) > "$R/run.log" 2>&1; rc=$?
  echo "T3PG $TAG END $(date +%T) rc=$rc :: $(grep -a '^--- ' "$R/run.log" | tail -1)" >> "$O"
  local fr im
  read -r fr im < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" "$R/out/inputs.json" 2>/dev/null)
  if [ "${fr:-}" != 956173e054762e91 ] || [ "${im:-}" != dc2a1356dce735e0 ]; then
    echo "PG VOID: fixture record ${fr:-none} (want 956173e054762e91), image ${im:-none} (want dc2a1356dce735e0)" >> "$O"; return 1
  fi
  echo "PG inputs as recorded: fixture $fr, image $im" >> "$O"
  local B=$R/raw/guest/board.txt nd
  nd=$(grep -a -c "FATAL: .*could not open directory" $B 2>/dev/null)
  if [ "${nd:-x}" != 0 ] || grep -a -q "POSITIVE CONTROL FAILED" "$R/run.log"; then
    echo "PG MUST-FIRE FAIL: directory-open FATAL x${nd:-NO-DATA}, positive control $(grep -a -c 'POSITIVE CONTROL FAILED' $R/run.log) failure line(s)" >> "$O"; return 1
  fi
  echo "PG MUST-FIRE OK: no directory-open FATAL in any session; positive control answered" >> "$O"
  echo "PG readings: FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) DRT1 $(grep -a -c '^DRT1:' $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) Oops $(grep -a -c Oops $B) pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) collect-skipped $(grep -a -c 'collect skipped' $B) banner gen-push x$(grep -a -c 'diag gen-push' $B)" >> "$O"
  KIT_DESC="runner 7ce203f9fd6a3cce (dev; QEMU PASS t3-sql-nf2-r2); image dc2a1356dce735e0 (postgres lt1, pg-pools); scored against pgp-trim/bundle (QEMU qkit7-batch2, module a05c198b, movc_null_scalar OFF, the trimmed fixture); kit bkit8-batch-b$BRD (batch bitstream caplifive_vp2c_c721c1a2a, module 21fbfc397e7c7a42, monitor capstone-sbi d9da397 + FETCH_RETRY_CAP) on board $BRD" \
    RESULTS_DIR=2026-10-22-batch-c721c1a2a-b$BRD ARM_DIR=vp2-pg-pools ${PG_SCORE:-bash "$S/score-arm.sh"} "$TAG" "$P" "$R" "$O" >> "$O" 2>&1
  echo "PG DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
