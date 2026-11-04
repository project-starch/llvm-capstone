#!/bin/bash
# fa-b1.sh: adaptive fault-ahead (wt-fa a6fba52f75c3) on board 1, bitstream 3, the 3 Nov stages and images (baselines in
# spec/board/out). Pre-registration: spec-silicon results/2026-11-04-fault-ahead/PREREG.md, Addendum 4 (FA_PREREG).
#  1. Wait until board 1's chain (FA_CHAINPID, default 3629844) has stopped at the sentinel STOP-for-c5a-flash.
#  1b. The rtl lane's slot-bare kit on bitstream 3 (informational).
#  2. OFF arm: imagick test (both arms) on kit bkit18-faoff-b3-b1 into out-fa-b1-off.
#  3. Only if fa-grade-b1.py reads both OFF rows as the baseline (F1-F3): the six stages on bkit18-fa-b3-b1 into out-fa-b1.
#  4. Always: launch perf/c5a-first/c5a-b1.sh (the C.5a session; it then restarts board 1's chain at the end).
# Stubs for the dry run: FA_CHAIN FA_CHAINPID FA_POLL_S FA_O FA_LIST FA_CHROOT FA_OUTROOT FA_C5.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${FA_O:-$H/perf/fa} L
  L=$O/fa-b1.log
  local CHAIN=${FA_CHAIN:-bash $O/board/chain-board.sh} ORIG=${FA_ORIGCHAIN:-bash $H/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh}
  local LIST=${FA_LIST:-$H/spec/board/b1-resume-oom.list} CHROOT=${FA_CHROOT:-$H/spec/board/out-hfl} SENT=STOP-for-c5a-flash
  local OUTR=${FA_OUTROOT:-$H/spec/board} K=$H/t3-smoke MOD=6e854c46c654c843 rc=x on=skipped
  local SK=${FA_SLOTKIT:-$H/perf/slot/kit-e1a835aa} RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local RUN=${FA_RUN:-python3 $RUNNER} BOOK=${FA_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} bid i src
  log() { echo "FA $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  for k in bkit18-fa-b3-b1 bkit18-faoff-b3-b1; do
    ( cd "$K/$k" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$K/$k/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
      || { log "STOP: kit $k"; return 2; }
  done
  grep -qx "fault_ahead_max=64" "$K/bkit18-fa-b3-b1/adapter/module/params" && [ ! -e "$K/bkit18-faoff-b3-b1/adapter/module/params" ] \
    || { log "STOP: kit params"; return 2; }
  ( cd "$SK/images" && sha256sum -c --quiet SHA256SUMS ) && [ -f "$SK/grade/slot-grade.py" ] || { log "STOP: slot kit"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 2; }
  [ -n "${FA_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$FA_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${FA_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  [ "$(grep -c "/$SENT " "$LIST")" = 1 ] || { log "STOP: the sentinel is not in $LIST exactly once"; return 2; }
  log "preconditions ok; waiting for board 1's chain to stop at the sentinel"
  local cpid=${FA_CHAINPID:-3629844} waited=0
  until ! kill -0 "$cpid" 2>/dev/null; do
    [ $waited -ge "${FA_WAIT_S:-86400}" ] && { log "STOP: chain pid $cpid still running after $waited s"; return 2; }
    sleep "${FA_POLL_S:-60}"; waited=$((waited + ${FA_POLL_S:-60}))
  done
  tail -3 "$CHROOT/chain-b1.log" | grep -q "CB STOP after $SENT" || { log "STOP: board 1's chain ended, but not at the sentinel"; tail -5 "$CHROOT/chain-b1.log" >> "$L"; return 2; }
  log "chain stopped at the sentinel"

  # The rtl lane's slot-bare kit (land/rtl-slot-kit e1a835aa466e) on bitstream 3 first: the DYN-unit decision waits on
  # it. One boot per image, slot-9 first; informational, gates nothing. (c5a-b1.sh runs it again on bs3: N = 2.)
  bid=; for i in $(seq 1 30); do bid=$($BOOK book 1 40 "vp2 slot-bare kit on bitstream 3 (board 1)") && break; bid=; sleep 60; done
  mkdir -p "$O/slot-bs3-b1"
  for i in slot-9 slot-6 slot-5 slot-7 slot-8 slot-1 slot-2 slot-3 slot-4; do
    log "SLOT $i BEGIN"
    ( export FPGA_URL; FPGA_URL=$(cat "$HOME/.claude-kisp/secrets/fpga-console-url")
      FPGA_BOARD=1 FPGA_BITSTREAM=caplifive_vp2c_e7af48457.bit SUP_IMG=$SK/images/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$SK/images/$i.rec") \
        SUP_OUT=$O/slot-bs3-b1/$i SUP_TIMEOUT=90 timeout 900 $RUN > "$O/slot-bs3-b1/$i.stderr" 2>&1 ); src=$?
    echo $src > "$O/slot-bs3-b1/$i.rc"; log "SLOT $i END rc=$src"
    grep -q 'HARD STOP' "$O/slot-bs3-b1/$i.stderr" "$O/slot-bs3-b1/$i/run.log" 2>/dev/null && { log "SLOT HARD STOP at $i"; break; }
  done
  [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
  python3 "$SK/grade/slot-grade.py" "$O/slot-bs3-b1" bs3 > "$O/slot-bs3-b1.grade.txt" 2>&1; log "SLOT GRADE bs3 rc=$?"; sed 's/^/  /' "$O/slot-bs3-b1.grade.txt" >> "$L"

  SB_KIT=$K/bkit18-faoff-b3-b1 SB_BITSTREAM=e7af48457 SB_MODULE=$MOD $CHAIN 1 "$O/fa-off-b1.list" "$OUTR/out-fa-b1-off" >> "$L" 2>&1; rc=$?
  log "OFF arm chain rc=$rc"
  local offline=
  [ $rc = 0 ] && [ -e "$OUTR/out-fa-b1-off/.chain-b1-done" ] && offline=$(CAPSTONE_WORK=$H python3 "$O/fa-grade-b1.py" 2>&1 | grep "^off ")
  log "OFF arm grade: ${offline:-none}"
  if [ "$(echo "$offline" | grep -c "F1 holds.*F2 holds.*F3 holds")" = 2 ]; then
    SB_KIT=$K/bkit18-fa-b3-b1 SB_BITSTREAM=e7af48457 SB_MODULE=$MOD $CHAIN 1 "$O/fa-on-b1.list" "$OUTR/out-fa-b1" >> "$L" 2>&1; on=$?
    log "ON arm chain rc=$on"
  else log "OFF arm not clean (control, rc, outputs, faults or time differ from the baseline): the ON arm is not run"; fi
  CAPSTONE_WORK=$H python3 "$O/fa-grade-b1.py" > "$O/grade-b1.txt" 2>&1; log "GRADE rc=$?"; sed 's/^/  /' "$O/grade-b1.txt" >> "$L"

  log "launching the C.5a session (perf/c5a-first/c5a-b1.sh)"
  ( cd "$H/perf/c5a-first" && C5_PREREG=${FA_C5PREREG:-$FA_PREREG} setsid nohup ${FA_C5:-bash $H/perf/c5a-first/c5a-b1.sh} >> "$H/perf/c5a-first/c5a-b1.nohup" 2>&1 < /dev/null & )
  log "DONE off $rc on $on"
}
main "$@"; exit $?
