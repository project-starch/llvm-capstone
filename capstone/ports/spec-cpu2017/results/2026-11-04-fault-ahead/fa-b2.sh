#!/bin/bash
# fa-b2.sh: adaptive fault-ahead (wt-fa a6fba52f75c3) on board 2, ObjBounds 603528b8b, the Spatial images of out-objb.
# Pre-registration: spec-silicon results/2026-11-04-fault-ahead/PREREG.md (FA_PREREG names its commit; must be on origin).
#  1. Wait until board 2's chain (pid FA_CHAINPID) has stopped at the sentinel `.../STOP-for-fa` in spatial-b2.list.
#  2. Control arm: imagick test on kit bkit18-faoff-objb-b2 (the new module, fault-ahead off) into out-fa-b2-off.
#  3. Only if fa-grade.py reads the OFF arm as the baseline (F1-F3: control, outputs, instret, no params; same faults;
#     real time within 5 %): the six stages on kit bkit18-fa-objb-b2 (fault-ahead on, read
#     back through sysfs) into out-fa-b2, cheapest first.
#  4. Always: restart board 2's chain on the list's lines after the sentinel, with its own kit (bkit15-objb-b2).
# Stubs for the dry run: FA_CHAIN FA_CHAINPID FA_POLL_S FA_O FA_LIST FA_CHROOT FA_REST FA_OUTROOT.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${FA_O:-$H/perf/fa} L
  L=$O/fa-b2.log
  local CHAIN=${FA_CHAIN:-bash $O/board/chain-board.sh} ORIG=${FA_ORIGCHAIN:-bash $H/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh}
  local LIST=${FA_LIST:-$H/spec/board/spatial-b2.list} CHROOT=${FA_CHROOT:-$H/spec/board/out-objb} SENT=STOP-for-fa
  local OUTR=${FA_OUTROOT:-$H/spec/board} K=$H/t3-smoke MOD=6e854c46c654c843 rc=x on=skipped
  local SK=${FA_SLOTKIT:-$H/perf/slot/kit-e1a835aa} RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local RUN=${FA_RUN:-python3 $RUNNER} BOOK=${FA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} bid i src
  log() { echo "FA $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  for k in bkit18-fa-objb-b2 bkit18-faoff-objb-b2; do
    ( cd "$K/$k" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$K/$k/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
      || { log "STOP: kit $k"; return 2; }
  done
  grep -qx "fault_ahead_max=64" "$K/bkit18-fa-objb-b2/adapter/module/params" && [ ! -e "$K/bkit18-faoff-objb-b2/adapter/module/params" ] \
    || { log "STOP: kit params"; return 2; }
  ( cd "$SK/images" && sha256sum -c --quiet SHA256SUMS ) && [ -f "$SK/grade/slot-grade.py" ] || { log "STOP: slot kit"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 2; }
  [ -n "${FA_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$FA_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${FA_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  [ "$(grep -c "/$SENT " "$LIST")" = 1 ] || { log "STOP: the sentinel is not in $LIST exactly once"; return 2; }
  log "preconditions ok; waiting for board 2's chain to stop at the sentinel"
  local cpid=${FA_CHAINPID:-688544} waited=0
  until ! kill -0 "$cpid" 2>/dev/null; do
    [ $waited -ge "${FA_WAIT_S:-86400}" ] && { log "STOP: chain pid $cpid still running after $waited s"; return 2; }
    sleep "${FA_POLL_S:-60}"; waited=$((waited + ${FA_POLL_S:-60}))
  done
  tail -3 "$CHROOT/chain-b2.log" | grep -q "CB STOP after $SENT" || { log "STOP: board 2's chain ended, but not at the sentinel"; tail -5 "$CHROOT/chain-b2.log" >> "$L"; return 2; }
  log "chain stopped at the sentinel"

  # The rtl lane's slot-bare kit (land/rtl-slot-kit e1a835aa466e) on ObjBounds, one boot per image, slot-9 first.
  # Informational: it gates nothing here.
  bid=; for i in $(seq 1 30); do bid=$($BOOK book 2 40 "vp2 slot-bare kit on ObjBounds (board 2)") && break; bid=; sleep 60; done
  mkdir -p "$O/slot-objb"
  for i in slot-9 slot-6 slot-5 slot-7 slot-8 slot-1 slot-2 slot-3 slot-4; do
    log "SLOT $i BEGIN"
    ( export FPGA_URL; FPGA_URL=$(cat "$HOME/.claude-kisp/secrets/fpga-console-url-board2")
      FPGA_BOARD=2 FPGA_BITSTREAM=caplifive_vp2c_603528b8b.bit SUP_IMG=$SK/images/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$SK/images/$i.rec") \
        SUP_OUT=$O/slot-objb/$i SUP_TIMEOUT=90 timeout 900 $RUN > "$O/slot-objb/$i.stderr" 2>&1 ); src=$?
    echo $src > "$O/slot-objb/$i.rc"; log "SLOT $i END rc=$src"
    grep -q 'HARD STOP' "$O/slot-objb/$i.stderr" "$O/slot-objb/$i/run.log" 2>/dev/null && { log "SLOT HARD STOP at $i"; break; }
  done
  [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
  python3 "$SK/grade/slot-grade.py" "$O/slot-objb" objb > "$O/slot-objb.grade.txt" 2>&1; log "SLOT GRADE objb rc=$?"; sed 's/^/  /' "$O/slot-objb.grade.txt" >> "$L"

  SB_KIT=$K/bkit18-faoff-objb-b2 SB_BITSTREAM=603528b8b SB_MODULE=$MOD $CHAIN 2 "$O/fa-off-b2.list" "$OUTR/out-fa-b2-off" >> "$L" 2>&1; rc=$?
  log "OFF arm chain rc=$rc"
  local offline=
  [ $rc = 0 ] && [ -e "$OUTR/out-fa-b2-off/.chain-b2-done" ] && offline=$(CAPSTONE_WORK=$H python3 "$O/fa-grade.py" 2>&1 | grep "^off ")
  log "OFF arm grade: ${offline:-none}"
  if echo "$offline" | grep -q "F1 holds.*F2 holds.*F3 holds"; then
    SB_KIT=$K/bkit18-fa-objb-b2 SB_BITSTREAM=603528b8b SB_MODULE=$MOD $CHAIN 2 "$O/fa-on-b2.list" "$OUTR/out-fa-b2" >> "$L" 2>&1; on=$?
    log "ON arm chain rc=$on"
  else log "OFF arm not clean (control, rc, outputs, faults or time differ from the baseline): the ON arm is not run"; fi
  CAPSTONE_WORK=$H python3 "$O/fa-grade.py" > "$O/grade.txt" 2>&1; log "GRADE rc=$?"; sed 's/^/  /' "$O/grade.txt" >> "$L"

  local rest=${FA_REST:-$H/spec/board/b2-after-fa.list}
  awk -v s="/$SENT " 'f{print} index($0,s){f=1}' "$LIST" > "$rest"
  log "restarting board 2's chain on $(grep -c -v '^#' "$rest") lines of $rest"
  SB_KIT=$K/bkit15-objb-b2 SB_BITSTREAM=603528b8b setsid nohup $ORIG 2 "$rest" "$CHROOT" > "$O/chain-restart.nohup" 2>&1 < /dev/null &
  log "DONE off $rc on $on; chain pid $!"
}
main "$@"; exit $?
