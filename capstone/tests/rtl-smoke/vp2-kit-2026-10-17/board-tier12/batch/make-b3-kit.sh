#!/bin/bash
# Bitstream 3's board kits (capstone-ariane e7af48457: bitstream 2 plus the R-62/R-63 divider fixes; the RTL diff from
# dcc6daf6e is core/fpu_div_sqrt_fix/norm_div_sqrt_mvp.sv and its Flist line, and ariane_pkg.sv's memory map is
# identical), made AFTER the flash, never before (board.env names the flashed bitstream). A copy of a bitstream-2 kit
# with EXACTLY one change, BOARD_BITSTREAM=<the flashed name>, checked three ways:
# - the source kit verifies against its SHA256SUMS and names caplifive_vp2c_dcc6daf6e;
# - the copy's SHA256SUMS covers the same file list, with every hash but board.env's equal to the source's;
# - board.env differs from the source's in exactly that one line.
# Usage: make-b3-kit.sh <source kit dir> <new kit dir> <bitstream name, no .bit> <log>.
main() {
  set -u
  local S=${1:?source kit} D=${2:?new kit} BS=${3:?bitstream name} O=${4:?log}
  case $BS in caplifive_*e7af48457*) ;; *) echo "MK3 STOP: bitstream name $BS is not an e7af48457 build" >> "$O"; return 1;; esac
  ( cd $S && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_dcc6daf6e" $S/board.env \
    || { echo "MK3 STOP: source kit $S does not verify or is not a bitstream-2 kit" >> "$O"; return 1; }
  [ -e $D ] && { echo "MK3 STOP: $D exists" >> "$O"; return 1; }
  cp -a $S $D || { echo "MK3 STOP: copy" >> "$O"; return 1; }
  python3 - $D/board.env "$BS" <<'PY' || { echo "MK3 STOP: board.env edit" >> "$O"; return 1; }
import re, sys
p, bs = sys.argv[1:]
s = open(p).read()
s, n = re.subn(r"^BOARD_BITSTREAM=.*$", f"BOARD_BITSTREAM={bs}", s, flags=re.M)
assert n == 1
open(p, "w").write(s)
PY
  ( cd $D && awk '{print $2}' SHA256SUMS | xargs sha256sum > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) \
    || { echo "MK3 STOP: sums" >> "$O"; return 1; }
  local other dl
  other=$(diff <(grep -v ' \./board.env$\| board.env$' $S/SHA256SUMS) <(grep -v ' \./board.env$\| board.env$' $D/SHA256SUMS) | grep -c '^[<>]')
  dl=$(diff $S/board.env $D/board.env | grep -c '^[<>]')
  [ "$other" = 0 ] && [ "$dl" = 2 ] || { echo "MK3 STOP: $other other sum lines differ, board.env differs in $dl lines" >> "$O"; return 1; }
  echo "MK3 kit $D from ${S##*/}: board.env diff:" >> "$O"; diff $S/board.env $D/board.env | grep '^[<>]' | grep -v -i 'url\|token' >> "$O"
  echo "MK3 ${D##*/} DONE" >> "$O"
}
main "$@"; exit $?
