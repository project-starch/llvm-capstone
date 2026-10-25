#!/bin/bash
# BITSTREAM 2's (18-bit index) first boots on the flashed board (2026-10-23, before the flash; pre-registered in README.md):
# run-batch-first.sh with: the kit bkit9-b2-b<board> (make-b2-kit.sh: module 215a09be5a89360c, launcher fad465ce4dea7121);
# the Tier 1 firmware 0c24496f80ef40d5 (b2-prep.sh: the batch Tier 1 stage with only capstone_vm.ko = 215a09be, image
# b052598d70aba6f9); s23-cpmp-rearm graded --expect batch (the RTL lane: its vector is unchanged on bitstream 2, no id is
# printed); a NEW second step, va-alloc-bare graded --expect b2 (the RTL lane's PREREG, dev cd30ce0e0876, carried to
# 68d0ffaf8 at dev 816fd44f5849), anything but PASS stopping the chain before Linux like s23; and Tier 1 with the
# CAPSTONE_VM_STATS line of every VM rung printed. run-batch-first.sh's header follows.
# The batch bitstream's FIRST boots on the flashed board, in order, each gating the next (written 2026-10-22, before the
# flash; run only after the lead's reflash word, the flash, make-batch-kit.sh, and a pre-registration pushed for that
# board and bitstream):
#  1. s23-cpmp-rearm as a bare JTAG boot (tier3-s23-bare.sh with the batch kit), graded --expect batch by s23-grade.py
#     (the RTL lane's PREREG at 7ae1d8491466: re-arm probes all 1, total 14). Anything but AS EXPECTED stops the chain:
#     VOID is an instrument reading, FINDING is item 1 not behaving as simulated, and either is surfaced before Linux.
#  2. Tier 1 on the batch image 64ec46d1f8d9461b / firmware 6423545a567b2f3c (batch-prep.sh; batch-t1c): b0-stats,
#     b0-hello, vm-contract, vm-suite, one cold boot, the board watchdog beside the driver. Rung retvals are printed
#     beside rev7's oracles for reference only: the batch boot is graded per test against the compiler lane's QEMU Tier 1
#     on the batch kits (33/45, the 12 known failures; sparse-churn predicted to complete), which the pre-registration
#     states. The S-23 counters, FRTY/FRTC and the release lines are printed as the srz runners print them.
# Usage: run-batch-first.sh <board 1|2> <bitstream name, no .bit> <out dir>. Ends by itself.
# BF_S23 / BF_DRIVER / BF_BUSY / BF_BOOK / BF_WD / BF_KIT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} BS=${2:?bitstream name} D=${3:?out dir}
  local H=/mnt/hdd/$USER/capstone S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local J; J=$(cd $S/.. && pwd)
  local T3; T3=$(cd $S/../../tier3-smoke && pwd)
  local K=${BF_KIT:-$H/t3-smoke/bkit9-b2-b$BRD} FW=$H/batch-t1-b2/fw/fw_payload.bin O=/tmp/capstone/tier1/oracles-rev7 L rc bid v r
  [ -e $D ] && { echo "BF STOP: $D exists"; return 1; }
  mkdir -p $D; L=$D/SEQ
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$BS" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] \
    && [ "$(sha256sum $K/adapter/capstone-vexec | cut -c1-16)" = fad465ce4dea7121 ] \
    || { echo "BF STOP: kit $K is not the bitstream-2 kit for $BS" >> $L; return 1; }
  [ "$(sha256sum $FW | cut -c1-16)" = 0c24496f80ef40d5 ] || { echo "BF STOP: the firmware is not 0c24496f" >> $L; return 1; }
  local drv url fb dsha
  drv=$(sed -n 's/^BOARD_DRIVER_DIR=//p' $K/board.env); dsha=$(sed -n 's/^BOARD_DRIVER_SHA=//p' $K/board.env)
  url=$(sed -n 's/^BOARD_URL_FILE=//p' $K/board.env); url=${url:-$HOME/.claude-kisp/secrets/fpga-console-url}
  fb=$(sed -n 's/^FPGA_BOARD=//p' $K/board.env); fb=${fb:-1}
  [ "$fb" = "$BRD" ] && [ "$(sha256sum $drv/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$dsha" ] && [ -s "$url" ] \
    || { echo "BF STOP: board.env (board $fb, driver $drv)" >> $L; return 1; }
  echo "BF START $(date +%T) board $BRD bitstream $BS kit ${K##*/} fw 0c24496f" >> $L
  # 1. s23 bare, the batch vector
  S23_KIT=$K S23_ROOT=$D/s23-bare ${BF_S23:-bash $T3/tier3-s23-bare.sh} $BRD $L batch
  grep -q "^S23 GRADE rc=0$" $L || { echo "BF STOP: s23 did not read AS EXPECTED (batch); Tier 1 not booted" >> $L; return 1; }
  # 1b. va-alloc-bare, bitstream 2's vector (the generation field moved to [29:18])
  S23_KIT=$K S23_ROOT=$D/va-alloc-bare ${BF_VA:-bash $T3/tier3-va-alloc-bare.sh} $BRD $L b2
  grep -q "^VA GRADE rc=0$" $L || { echo "BF STOP: va-alloc-bare did not read PASS (bitstream-2 vector); Tier 1 not booted" >> $L; return 1; }
  # 2. Tier 1
  ${BF_BUSY:-python3 $drv/board-busy.py $BRD} > /dev/null || { echo "BF STOP: board $BRD is busy before Tier 1" >> $L; return 1; }
  bid=$(${BF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 150 "vp2 batch bitstream Tier 1 (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "BF STOP: the hub refused the Tier 1 booking (409): $bid" >> $L; return 1; }
  [ $rc = 0 ] || { echo "BF NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> $L; bid=; }
  echo "BF TIER1 BEGIN $(date +%T) booking ${bid:-none}" >> $L
  # The URL is read into the environment only, never echoed.
  ( cd "$drv" && FPGA_BOARD=$fb FPGA_BITSTREAM=$BS.bit FPGA_URL="$(cat "$url")" FPGA_FW=$FW \
      BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract vm-suite" LADDER_FPGA_DIR=$O \
      BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
      timeout 9000 ${BF_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 ) &
  local runner=$!
  ( cd "$drv" && FPGA_BOARD=$fb ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 ${BF_WD:-bash board-watchdog.sh} $D/driver.log 900 $runner > $D/watchdog.log 2>&1 ) &
  local wd=$!
  wait $runner; rc=$?
  sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  [ -n "$bid" ] && ${BF_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> $L 2>&1
  echo "BF TIER1 DRIVER rc=$rc $(date +%T); watchdog lines $(grep -c '^ALIVE\|^QUIET\|^STALE' $D/watchdog.log)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  for r in b0-stats b0-hello vm-contract vm-suite; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} (rev7 oracle $(cat $O/$r.oracle), reference only)" >> $L
  done
  echo "VM exits: $(grep -a -o '^VM_EXIT:[a-z_-]*:[0-9]*' $D/board.txt | tr '\n' ' ')" >> $L
  grep -a -o '^CAPSTONE_VM_STATS .*' $D/board.txt | sed 's/ launch_ns=.*//; s/^/  /' >> $L
  echo "SPARSE: $(grep -a -o 'VIRTUAL_SPARSE_CHURN_OK[^[:cntrl:]]*' $D/board.txt | tail -1)" >> $L
  echo "VM_BEGIN without VM_EXIT: $(for b in $(grep -a -o '^VM_BEGIN:[a-z_-]*' $D/board.txt | cut -d: -f2); do grep -a -q "^VM_EXIT:$b:" $D/board.txt || echo $b; done | tr '\n' ' ')" >> $L
  echo "FRTY $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | wc -l) FRTC $(grep -a -c 'FRTC' $D/board.txt) DRT1 $(grep -a -c '^DRT1:' $D/board.txt) INJS $(grep -a -c INJS $D/board.txt) INJL $(grep -a -c INJL $D/board.txt) Oops $(grep -a -c Oops $D/board.txt) pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $D/board.txt)" >> $L
  python3 $J/boot3/cpmx-diag-read.py $D/board.txt > $D/cpmx-diag.txt 2>&1; echo "CPMX-DIAG rc=$? :: $(tail -1 $D/cpmx-diag.txt)" >> $L
  echo "BF board$BRD DONE $(date +%T)" >> $L
}
main "$@"; exit $?
