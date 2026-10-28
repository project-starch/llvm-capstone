#!/bin/bash
# BITSTREAM 3 version (2026-10-27; pre-registered in README.md before its boot): tier3-pg-b2-arm.sh with only the board
# side changed: the bitstream-3 kits bkit11-b3-b<board> / bkit11-b3-mb<board> (make-b3-kit.sh copies of the bitstream-2
# kits, BOARD_BITSTREAM=caplifive_vp2c_e7af48457, module 215a09be unchanged), output t3-run-b3-e7af48457-b<board>, results
# under RESULTS_DIR 2026-10-27-b3-e7af48457-b<board>, and a lab-hub booking around the boot (PG_BOOK stubs it). The
# references, image, runner, fixture and grade are bitstream 2's, unchanged. The bitstream-2 version's header follows.
# BITSTREAM 2 version (2026-10-25; pre-registered in README.md before its boot): tier3-pg-batch-arm.sh on the board's
# bitstream-2 kits bkit9-b2-b<board> / bkit9-b2-mb<board> (caplifive_vp2c_dcc6daf6e, module 215a09be), against the compiler
# lane's bitstream-2 references on the 18-bit twin qkit7-b2twin3-q8: sql0 batch-refs-b2/ref-b2/sql0/bundle, pg-pools the
# TRIMMED batch-refs-b2/pgp-trim/bundle; results filed under RESULTS_DIR 2026-10-25-b2-dcc6daf6e-b<board>. The batch
# version's header follows.
# tier3-pg-batch.sh generalised to an arm tag (2026-10-22 18:0x, a new file because pg-pools runs from the original): <tag>
# pg-pools runs exactly as tier3-pg-batch.sh; <tag> sql0 runs postgres sql-repros lt0 (virtual-malloc) on the batch kit,
# image t3-nomovc-r60/postgres-lt0 (cf61cb4333d35cb6), scored against the compiler lane's batch reference
# batch-refs-r60/ref-batch/sql0/bundle (qkit7-batch2; stamp runner 7ce203f9, image cf61cb43, SWITCHES
# movc_null_scalar=OFF ... generations=ON; the trimmed fixture 956173e054762e91), filed under ARM_DIR vp2-malloc-nomovc.
# pg-pools's original header follows. tier3-sql-arm-rev2u.sh's sql0 path with what the batch arm changes:
#   - kits: the flashed board's batch kit bkit8-batch-b<board> and its MEASURE-ONLY twin bkit8-batch-mb<board>
#     (make-batch-kit.sh, MK_M=1): c721c1a2a, batch module 21fbfc39, monitor d9da397 + FETCH_RETRY_CAP;
#   - --arm virtual-pg-pools, image t3-images-r60/q15/postgres-lt1 (dc2a1356dce735e0);
#   - the trimmed fixture and share the sql0 board path uses, which the compiler lane's batch reference also ran:
#     pgp-trim/bundle on qkit7-batch2 (fixture record 956173e054762e91, image dc2a1356; 9 of 9 cells equal to the untrimmed
#     batch reference; controls fault; SWITCHES movc_null_scalar=OFF, the batch mimic);
#   - dev's runner 7ce203f9 from wt-q15, with its QEMU PASS (t3-sql-nf2-r2/CHECK), as the reference ran;
#   - scored by score-arm.sh into a NEW results directory named for the bitstream (RESULTS_DIR 2026-10-22-batch-c721c1a2a-b<board>).
# First the measure-only bake: the boot runs only if the batch firmware with this payload is under 26,000,000 B.
# Usage: tier3-pg-batch-arm.sh <board 1|2> <out log> <pg-pools|sql0>. PG_RUNNER / PG_RUNNER_SHA / PG_QCHECK / PG_BUSY / PG_ROOT / PG_TAG / PG_SCORE
# stub the dry run (PG_SCORE keeps the dry run from filing anything into the corpus's results).
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} ARMT=${3:?tag pg-pools|sql0}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local K=$H/t3-smoke/bkit11-b3-b$BRD KM=$H/t3-smoke/bkit11-b3-mb$BRD
  local IMG IMS ARM AD FIX=$H/t3-new6/tmp/pgfixture-trim/cluster
  local SHARE=$H/t3-new6/tmp/pgfixture/install/share/postgresql P ST STP
  case $ARMT in
    pg-pools) IMG=$H/t3-images-r60/q15/postgres-lt1/image/postgres.dom IMS=dc2a1356dce735e0 ARM=virtual-pg-pools AD=vp2-pg-pools
              P=$H/batch-refs-b2/pgp-trim/bundle ST=$H/batch-refs-b2/pgp-trim/pg-pools-trim.stamp STP="^STAMP pg-pools-trim runner 7ce203f9fd6a3cce .*qemu 708bbff5663f0e86 module 3325473537072a3a .*image dc2a1356dce735e0";;
    sql0)     IMG=$H/t3-nomovc-r60/postgres-lt0/image/postgres.dom IMS=cf61cb4333d35cb6 ARM=virtual-malloc AD=vp2-malloc-nomovc
              P=$H/batch-refs-b2/ref-b2/sql0/bundle ST=$H/batch-refs-b2/ref-b2/sql0.stamp STP="^STAMP sql0 runner 7ce203f9fd6a3cce .*qemu 708bbff5663f0e86 module 3325473537072a3a .*image cf61cb4333d35cb6";;
    *) echo "PG STOP: tag $ARMT" >> "$O"; return 2;;
  esac
  local TAG=${PG_TAG:-b3-$ARMT-b$BRD} R=${PG_ROOT:-$H/t3-run-b3-e7af48457-b$BRD/$ARMT} BUSY rc bid
  local RUN=${PG_RUNNER:-/tmp/capstone/wt-q15/capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py}
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PG STOP: board $BRD" >> "$O"; return 2;;
  esac
  local k; for k in $K $KM; do
    ( cd $k && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_e7af48457" $k/board.env \
      && [ "$(sha256sum $k/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] && grep -q 'if \[ -x "$STAGE/$f" \]; then m=0755' $k/board-run.sh \
      || { echo "PG STOP: $k is not a bitstream-3 kit with the execute-bit rule" >> "$O"; return 1; }
  done
  grep -q "MEASURE-ONLY" $KM/board-run.sh || { echo "PG STOP: $KM is not the measure-only kit" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = $IMS ] || { echo "PG STOP: the image is not $IMS" >> "$O"; return 1; }
  grep -q "$STP" $ST && grep -q "SWITCHES movc_null_scalar=OFF compact_rounding=ON.* generations=ON.*index=18" $ST && [ -f $P/inputs.json ] \
    || { echo "PG STOP: the reference's stamp, switches or bundle" >> "$O"; return 1; }
  [ "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" $P/inputs.json)" = "956173e054762e91 $IMS" ] \
    || { echo "PG STOP: the reference bundle's fixture or image record" >> "$O"; return 1; }
  local RH; RH=$(sha256sum "$RUN" | cut -c1-16)
  [ "$RH" = "${PG_RUNNER_SHA:-7ce203f9fd6a3cce}" ] || { echo "PG STOP: the runner is $RH, not dev's 7ce203f9" >> "$O"; return 1; }
  grep -qx "PASS $RH" "${PG_QCHECK:-$H/t3-sql-nf2-r2/CHECK}" 2>/dev/null || { echo "PG STOP: no QEMU PASS for runner $RH" >> "$O"; return 1; }
  [ -d $FIX ] || { echo "PG STOP: no fixture" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "PG STOP: $R exists" >> "$O"; return 1; }
  ${PG_BUSY:-$BUSY} > /dev/null || { echo "PG STOP: board $BRD is busy" >> "$O"; return 1; }
  mkdir -p "$R/tmp" "$R/measure/tmp"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/measure/tmp taskset -c 16-31 timeout 3600 python3 "$RUN" --arm $ARM --board-kit $KM \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/measure/out" --raw "$R/measure/raw" ) > "$R/measure/run.log" 2>&1
  local sz; sz=$(grep -a -o "MEASURE-ONLY: firmware [0-9]* B\|STOP: firmware [0-9]* B" "$R/measure/raw/guest/SEQ" 2>/dev/null | grep -o "[0-9]\+" | head -1)
  echo "PG MEASURE: firmware ${sz:-NONE} B against 26000000" >> "$O"
  [ -n "$sz" ] && [ "$sz" -le 26000000 ] || { echo "PG STOP: the measured firmware is not under the cap; no boot" >> "$O"; return 1; }
  ${PG_BUSY:-$BUSY} > /dev/null || { echo "PG STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${PG_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2 bitstream 3 postgres $ARMT (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "PG STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "PG NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "T3PG $TAG BEGIN $(date +%T) booking ${bid:-none}" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1
    CAPSTONE_TMP_ROOT=$R/tmp taskset -c 16-31 timeout 36000 python3 "$RUN" --arm $ARM --board-kit $K \
      --image $IMG --fixture $FIX --pg-share $SHARE --llvm-bin $LB --out "$R/out" --raw "$R/raw" ) > "$R/run.log" 2>&1; rc=$?
  [ -n "$bid" ] && ${PG_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "T3PG $TAG END $(date +%T) rc=$rc :: $(grep -a '^--- ' "$R/run.log" | tail -1)" >> "$O"
  local fr im
  read -r fr im < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['fixture']['cluster_tree_sha256'][:16], d['image']['sha256'][:16])" "$R/out/inputs.json" 2>/dev/null)
  if [ "${fr:-}" != 956173e054762e91 ] || [ "${im:-}" != $IMS ]; then
    echo "PG VOID: fixture record ${fr:-none} (want 956173e054762e91), image ${im:-none} (want $IMS)" >> "$O"; return 1
  fi
  echo "PG inputs as recorded: fixture $fr, image $im" >> "$O"
  local B=$R/raw/guest/board.txt nd
  nd=$(grep -a -c "FATAL: .*could not open directory" $B 2>/dev/null)
  if [ "${nd:-x}" != 0 ] || grep -a -q "POSITIVE CONTROL FAILED" "$R/run.log"; then
    echo "PG MUST-FIRE FAIL: directory-open FATAL x${nd:-NO-DATA}, positive control $(grep -a -c 'POSITIVE CONTROL FAILED' $R/run.log) failure line(s)" >> "$O"; return 1
  fi
  echo "PG MUST-FIRE OK: no directory-open FATAL in any session; positive control answered" >> "$O"
  echo "PG readings: FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) DRT1 $(grep -a -c '^DRT1:' $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) Oops $(grep -a -c Oops $B) pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) collect-skipped $(grep -a -c 'collect skipped' $B) banner gen-push x$(grep -a -c 'diag gen-push' $B)" >> "$O"
  KIT_DESC="runner 7ce203f9fd6a3cce (dev; QEMU PASS t3-sql-nf2-r2); image $IMS (postgres, $ARM); scored against ${P#$H/} (QEMU qkit7-b2twin3-q8, module 33254735, the 18-bit twin, movc_null_scalar OFF, the trimmed fixture); kit bkit11-b3-b$BRD (bitstream 3 caplifive_vp2c_e7af48457, module 215a09be5a89360c, launcher fad465ce4dea7121) on board $BRD" \
    RESULTS_DIR=2026-10-27-b3-e7af48457-b$BRD ARM_DIR=$AD ${PG_SCORE:-bash "$S/score-arm.sh"} "$TAG" "$P" "$R" "$O" >> "$O" 2>&1
  echo "PG DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
