#!/bin/bash
# Bitstream 2's board kit (the 18-bit index), made AFTER the lead's reflash word and the flash, never before (the kit's
# board.env names the flashed bitstream): a copy of the flashed board's BATCH kit with exactly three changes, checked by
# diff against it:
#   BOARD_BITSTREAM=<the flashed bitstream's name>  (the only board.env line that may differ);
#   adapter/module/capstone_vm.ko = the 18-bit board module 215a09be5a89360c (runtime lane: runtime/flat-index-18-gen
#     9e7bf6e0b584 = the batch module 21fbfc39 plus -DCAPSTONE_VM_INDEX_BITS=18; 0 RVC; index_bits reads 18);
#   adapter/capstone-vexec = launcher fad465ce4dea7121 (c14933904de4 with only CV_MAX_ARENAS 512 -> 8192).
# The monitor, firmware defines and board-run.sh stay the batch kit's (d9da397 + FETCH_RETRY_CAP). SHA256SUMS regenerated
# and checked. Usage: make-b2-kit.sh <board 1|2> <bitstream name, no .bit> <log>. MK_N=1 bases it on the board's
# NORANDMAPS batch kit (writing bkit9-b2-nb<board>), MK_M=1 on its MEASURE-ONLY batch kit (bkit9-b2-mb<board>); plain
# writes bkit9-b2-b<board>. MK_X stubs the output root; MK_MOD / MK_VEX override the parts' paths for the dry run.
main() {
  set -u
  local BRD=${1:?board} BS=${2:?bitstream name} O=${3:?log} H=/mnt/hdd/$USER/capstone S D
  local M=${MK_MOD:-$H/board-modules/capstone_vm-215a09be5a89360c.ko} V=${MK_VEX:-$H/board-modules/capstone-vexec-fad465ce4dea7121}
  local MS=${MK_MOD_SHA:-215a09be5a89360c} VS=${MK_VEX_SHA:-fad465ce4dea7121}
  local n=""; [ "${MK_N:-0}" = 1 ] && n=n; [ "${MK_M:-0}" = 1 ] && n=m
  case $BRD in 1|2) S=$H/t3-smoke/bkit8-batch-${n}b$BRD;; *) echo "MK STOP: board $BRD" >> "$O"; return 2;; esac
  D=${MK_X:-$H/t3-smoke}/bkit9-b2-${n}b$BRD
  case $BS in caplifive_*) ;; *) echo "MK STOP: bitstream name $BS" >> "$O"; return 1;; esac
  ( cd $S && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $S/board.env \
    && [ "$(sha256sum $S/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] \
    || { echo "MK STOP: base kit $S is not the batch kit" >> "$O"; return 1; }
  [ "$(sha256sum $M 2>/dev/null | cut -c1-16)" = "$MS" ] || { echo "MK STOP: the module $M is not $MS" >> "$O"; return 1; }
  [ "$(sha256sum $V 2>/dev/null | cut -c1-16)" = "$VS" ] || { echo "MK STOP: the launcher $V is not $VS" >> "$O"; return 1; }
  [ -e $D ] && { echo "MK STOP: $D exists" >> "$O"; return 1; }
  cp -a $S $D && cp $M $D/adapter/module/capstone_vm.ko && cp $V $D/adapter/capstone-vexec && chmod 755 $D/adapter/capstone-vexec \
    || { echo "MK STOP: copy" >> "$O"; return 1; }
  python3 - $D/board.env "$BS" <<'PY' || { echo "MK STOP: board.env edit" >> "$O"; return 1; }
import re, sys
p, bs = sys.argv[1:]
s = open(p).read()
s, n = re.subn(r"^BOARD_BITSTREAM=.*$", f"BOARD_BITSTREAM={bs}", s, flags=re.M)
assert n == 1
open(p, "w").write(s)
PY
  ( cd $D && sha256sum ./adapter/capstone-vexec ./adapter/module/capstone_vm.ko ./bake-manifest.sh ./base.manifest ./board.env ./board-run.sh > SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) || { echo "MK STOP: sums" >> "$O"; return 1; }
  local dl; dl=$(diff $S/board.env $D/board.env | grep -c '^[<>]')
  [ "$dl" = 2 ] && diff $S/board-run.sh $D/board-run.sh > /dev/null || { echo "MK STOP: board.env differs in $dl lines, or board-run.sh differs" >> "$O"; return 1; }
  echo "MK kit $D: board.env diff against $S:" >> "$O"; diff $S/board.env $D/board.env | grep '^[<>]' | grep -v -i 'url\|token' >> "$O"
  echo "MK module $(sha256sum $D/adapter/module/capstone_vm.ko | cut -c1-16) launcher $(sha256sum $D/adapter/capstone-vexec | cut -c1-16) DONE" >> "$O"
}
main "$@"; exit $?
