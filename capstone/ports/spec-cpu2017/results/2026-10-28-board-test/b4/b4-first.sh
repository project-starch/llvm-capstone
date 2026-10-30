#!/bin/bash
# Bitstream 4's (B: capstone-ariane 325966ee4, 55-bit lifetime ids) flash and first boots on board 2, on the lead's own
# word in the runtime session (2026-10-30: "Даю добро на перепрошивку, как будет готово"; "Wns не важен"). Pre-registered
# in PREREG.md Addendum 14 before launch; the predictions are module-ids55 PREREG section 2 + addendum 2 (e08dfaa95f22).
# flash-and-first-bs3.sh's shape, with these changes:
# - the wait is for board 2's chain to end: continue-b2.log's "continue b2: chain rc=" line (written on every exit path
#   of chain-board.sh after the list was cut at its read offset), then for the board to be free;
# - the fallback that must stay registered is bitstream 3, caplifive_vp2c_e7af48457.bit;
# - after the flash, the resident record is re-read on a fresh session (nv-readback.py: name, and sha256 if exposed);
# - the kit is make-b4-kit.sh's three-change copy of bkit11-b3-b2;
# - boot 1 (b4-t1a: Tier 1 contract + live 1000/400000/1100000, 1100000 last) is judged by b4-judge-t1a.py; boot 2
#   (b4-t1b: run.py@69fb6cae3db4's Tier 1 SCRIPT) boots only when boot 1 reads AS PREDICTED, judged by b4-check-t1b.py.
# Each step gates the next. Usage: b4-first.sh <log>. FB_* stub every external step for the dry run.
main() {
  set -u
  local O=${1:?log} BRD=2 NAME=caplifive_vp2c_325966ee4 H=/mnt/hdd/$USER/capstone S
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local BIT=${FB_BIT:-$H/bitstreams/$NAME.bit} SHA=f0893218dcdb53f4ad0b9fe452571f9e5c3bc5fa7ffade0025657c662c4da8cc
  local OUT=${FB_OUT:-$H/spec/b4-first} WAITF=${FB_WAITF:-$H/spec/board/out/continue-b2.log} D=$H/spec/stages-directed
  local RP=$H/wt-t1w55/capstone/runtime/virtual/run.py F0=$H/f55/t1-F0/result.json t0 rc bid r
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  export CAPSTONE_LLVM_BIN=$H/cc-prmerged/bin
  local BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"
  [ -e $OUT ] && { echo "FB STOP: $OUT exists" >> $O; return 2; }
  # Only a line written after launch counts: a chain rc line already in the log belongs to an earlier chain.
  local n0; n0=$(wc -l < $WAITF 2>/dev/null || echo 0)
  grep -q "continue b2: chain rc=" $WAITF 2>/dev/null && echo "FB NOTE: $WAITF already has a chain rc line; waiting for a new one" >> $O
  t0=$(date +%s)
  until tail -n +$((n0 + 1)) $WAITF 2>/dev/null | grep -q " continue b2: chain rc="; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_WAITMAX:-172800} ] && { echo "FB STOP: board 2's chain did not end" >> $O; return 1; }
    sleep ${FB_SLEEP:-60}
  done
  echo "FB board 2's chain ended $(date +%T): $(tail -n +$((n0 + 1)) $WAITF | grep 'continue b2: chain rc=' | tail -1)" >> $O
  t0=$(date +%s)
  until ${FB_BUSY:-$BUSY} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_BUSYMAX:-14400} ] && { echo "FB STOP: board 2 stayed busy" >> $O; return 1; }
    sleep ${FB_SLEEP:-60}
  done
  echo "FB board 2 free $(date +%T)" >> $O
  ${FB_STORE:-python3 $H/wt-spec/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/batch/store-has.py} no-such-bitstream-$$.bit >> $O 2>&1; rc=$?
  [ $rc = 1 ] || { echo "FB STOP: store-has control read rc=$rc, not 1" >> $O; return 1; }
  ${FB_STORE:-python3 $H/wt-spec/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/batch/store-has.py} caplifive_vp2c_e7af48457.bit >> $O 2>&1 \
    || { echo "FB STOP: the fallback (bitstream 3) is not registered" >> $O; return 1; }
  [ "$(sha256sum $BIT | cut -d' ' -f1)" = $SHA ] || { echo "FB STOP: local bitstream hash" >> $O; return 1; }
  bid=$(${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 bitstream 4 $NAME REFLASH (board $BRD, the lead's word)"); rc=$?
  [ $rc = 3 ] && { echo "FB STOP: the hub refused the flash booking (409): $bid" >> $O; return 1; }
  [ $rc = 0 ] || { echo "FB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> $O; bid=; }
  echo "FB FLASH BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> $O
  ${FB_FLASH:-python3 $H/wt-spec/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} $NAME.bit $BIT $SHA >> $O 2>&1; rc=$?
  [ -n "$bid" ] && ${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> $O 2>&1
  echo "FB FLASH END rc=$rc $(date +%T)" >> $O
  [ $rc = 0 ] || { echo "FB STOP: the flash did not read back $NAME.bit" >> $O; return 1; }
  ${FB_NVRB:-python3 $S/nv-readback.py} $NAME.bit $SHA >> $O 2>&1; rc=$?
  case $rc in
    0) echo "FB READBACK name and sha256 match" >> $O;;
    3) echo "FB READBACK name matches; the console exposes no sha256" >> $O;;
    *) echo "FB STOP: readback rc=$rc" >> $O; return 1;;
  esac
  ${FB_KIT:-bash $S/make-b4-kit.sh} ${FB_KSRC:-$H/t3-smoke}/bkit11-b3-b2 ${FB_KDST:-$H/t3-smoke}/bkit13-b4-b2 $O \
    || { echo "FB STOP: kit" >> $O; return 1; }
  mkdir -p $OUT
  echo "FB BOOT1 BEGIN $(date +%T)" >> $O
  ${FB_BOOT:-bash $S/b4-boot.sh} $BRD $D/b4-t1a $OUT/t1a 3600 120 >> $O 2>&1; echo "FB BOOT1 END rc=$? $(date +%T)" >> $O
  ${FB_J1:-python3 $S/b4-judge-t1a.py} $OUT/t1a/spec.txt > $OUT/t1a.judge 2>&1; r=$?
  sed 's/^/  /' $OUT/t1a.judge >> $O; echo "FB JUDGE1 rc=$r" >> $O
  if [ $r != 0 ]; then
    echo "FB HOLD: boot 1 did not read as predicted; boot 2 not booted" >> $O; echo "FB DONE $(date +%T)" >> $O; return 1
  fi
  echo "FB BOOT2 BEGIN $(date +%T)" >> $O
  ${FB_BOOT:-bash $S/b4-boot.sh} $BRD $D/b4-t1b $OUT/t1b 7200 240 >> $O 2>&1; echo "FB BOOT2 END rc=$? $(date +%T)" >> $O
  ${FB_J2:-python3 $S/b4-check-t1b.py} $OUT/t1b/spec.txt $RP $D/b4-t1b/stage/contract.dom $F0 > $OUT/t1b.judge 2>&1; r=$?
  sed 's/^/  /' $OUT/t1b.judge >> $O; echo "FB JUDGE2 rc=$r" >> $O
  echo "FB DONE $(date +%T)" >> $O
  return $r
}
main "$@"; exit $?
