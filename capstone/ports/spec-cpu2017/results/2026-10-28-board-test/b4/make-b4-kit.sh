#!/bin/bash
# Bitstream 4's (capstone-ariane 325966ee4, option B: 55-bit lifetime ids) board kit for board 2: a copy of the
# bitstream-3 Tier 1 kit bkit11-b3-b2 with EXACTLY three changes, checked by hash:
# - board.env: BOARD_BITSTREAM=caplifive_vp2c_325966ee4 (one line);
# - adapter/module/capstone_vm.ko: the flat ids55 board module 6422805f... (module-ids55 3523f8833531, PREREG addendum 2
#   at e08dfaa95f22);
# - adapter/capstone-vexec: the twin grid's launcher 62753dd6... (not the perf kits' 978c09d6).
# BAKE_REPLACE_KO (the b0 controls' process-ABI /capstone.ko) is unchanged. ariane_pkg.sv's memory constants are equal
# at e7af48457 and 325966ee4 (the diff touches only types and comments), so the firmware recipe and its DTB stand.
# Usage: make-b4-kit.sh <source kit> <new kit> <log>.
main() {
  set -u
  local S=${1:?source kit} D=${2:?new kit} O=${3:?log}
  local BS=caplifive_vp2c_325966ee4 KO=/mnt/hdd/$USER/capstone/f55/board-module/capstone_vm.ko
  local VX=/mnt/hdd/$USER/capstone/f55/kit-f55-nocsr/adapter/capstone-vexec
  local KOH=6422805f49533ea5494b943421e3a4cb8901405a3e8fade8f5f36c9edf841438
  local VXH=62753dd693cc4a2573b2c7defa68301cad4630b2785cf3c5321ada0f7610804a
  [ "$(sha256sum $KO | cut -d' ' -f1)" = $KOH ] && [ "$(sha256sum $VX | cut -d' ' -f1)" = $VXH ] \
    || { echo "MK4 STOP: module or launcher hash" >> "$O"; return 1; }
  ( cd $S && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_e7af48457" $S/board.env \
    || { echo "MK4 STOP: source kit $S does not verify or is not a bitstream-3 kit" >> "$O"; return 1; }
  [ -e $D ] && { echo "MK4 STOP: $D exists" >> "$O"; return 1; }
  cp -a $S $D || { echo "MK4 STOP: copy" >> "$O"; return 1; }
  cp $KO $D/adapter/module/capstone_vm.ko && cp $VX $D/adapter/capstone-vexec || { echo "MK4 STOP: adapter copy" >> "$O"; return 1; }
  python3 - $D/board.env "$BS" <<'PY' || { echo "MK4 STOP: board.env edit" >> "$O"; return 1; }
import re, sys
p, bs = sys.argv[1:]
s = open(p).read()
s, n = re.subn(r"^BOARD_BITSTREAM=.*$", f"BOARD_BITSTREAM={bs}", s, flags=re.M)
assert n == 1
open(p, "w").write(s)
PY
  ( cd $D && awk '{print $2}' SHA256SUMS | xargs sha256sum > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) \
    || { echo "MK4 STOP: sums" >> "$O"; return 1; }
  local diffs dl
  diffs=$(diff <(sort -k2 $S/SHA256SUMS) <(sort -k2 $D/SHA256SUMS) | grep '^>' | awk '{print $3}' | sort | tr '\n' ' ')
  dl=$(diff $S/board.env $D/board.env | grep -c '^[<>]')
  [ "$diffs" = "./adapter/capstone-vexec ./adapter/module/capstone_vm.ko ./board.env " ] && [ "$dl" = 2 ] \
    || { echo "MK4 STOP: changed sums [$diffs], board.env differs in $dl lines" >> "$O"; return 1; }
  grep -q " $KOH\| ./adapter/module/capstone_vm.ko" $D/SHA256SUMS && grep -q "^$KOH  ./adapter/module/capstone_vm.ko$" $D/SHA256SUMS \
    && grep -q "^$VXH  ./adapter/capstone-vexec$" $D/SHA256SUMS || { echo "MK4 STOP: the copy's sums do not name the module and launcher" >> "$O"; return 1; }
  echo "MK4 kit $D from ${S##*/}: changed [$diffs]; board.env diff:" >> "$O"
  diff $S/board.env $D/board.env | grep '^[<>]' | grep -v -i 'url\|token' >> "$O"
  echo "MK4 ${D##*/} DONE" >> "$O"
}
main "$@"; exit $?
