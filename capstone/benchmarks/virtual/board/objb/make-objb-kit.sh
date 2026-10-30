#!/bin/bash
# make-objb-kit.sh <source kit> <new kit> <bitstream name> <log>: the ObjBounds board kit for board 2, a copy of the
# bitstream-3 perf kit bkit12-perf-b3-b2 with EXACTLY one change, checked by hash: board.env's
# BOARD_BITSTREAM=<bitstream name> (the sealed .bit's name without .bit; the driver's resident check reads it).
# Everything else stays: the perf module 9c60fb53f80ce67e (bench counters), the launcher 978c09d61f774315, the firmware
# recipe, its defines and DTB. objbounds-b changes no memory constant: rtl's record (land/rtl-objb c8be354d28ef) reads
# ariane_pkg.sv's generated hash equal to e7af48457's, and git diff 1782d20bd 603528b8b -- core touches only the config
# package's define and two lines of the DYN unit's Anvil source.
main() {
  set -u
  local S=${1:?source kit} D=${2:?new kit} BS=${3:?bitstream name} O=${4:?log}
  case $BS in caplifive_vp2c_*) ;; *) echo "MKO STOP: $BS is not a caplifive_vp2c_ name" >> "$O"; return 1;; esac
  case $BS in *e7af48457*|*.bit) echo "MKO STOP: $BS is bitstream 3 or carries .bit" >> "$O"; return 1;; esac
  ( cd "$S" && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_e7af48457" "$S/board.env" \
    && [ "$(sha256sum "$S/adapter/module/capstone_vm.ko" | cut -c1-16)" = 9c60fb53f80ce67e ] \
    && [ "$(sha256sum "$S/adapter/capstone-vexec" | cut -c1-16)" = 978c09d61f774315 ] \
    || { echo "MKO STOP: source kit $S does not verify or is not the bitstream-3 perf kit" >> "$O"; return 1; }
  [ -e "$D" ] && { echo "MKO STOP: $D exists" >> "$O"; return 1; }
  cp -a "$S" "$D" || { echo "MKO STOP: copy" >> "$O"; return 1; }
  python3 - "$D/board.env" "$BS" <<'PY' || { echo "MKO STOP: board.env edit" >> "$O"; return 1; }
import re, sys
p, bs = sys.argv[1:]
s = open(p).read()
s, n = re.subn(r"^BOARD_BITSTREAM=.*$", f"BOARD_BITSTREAM={bs}", s, flags=re.M)
assert n == 1
open(p, "w").write(s)
PY
  ( cd "$D" && awk '{print $2}' SHA256SUMS | xargs sha256sum > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) \
    || { echo "MKO STOP: sums" >> "$O"; return 1; }
  local diffs dl
  diffs=$(diff <(sort -k2 "$S/SHA256SUMS") <(sort -k2 "$D/SHA256SUMS") | grep '^>' | awk '{print $3}' | sort | tr '\n' ' ')
  dl=$(diff "$S/board.env" "$D/board.env" | grep -c '^[<>]')
  [ "$diffs" = "./board.env " ] && [ "$dl" = 2 ] && grep -qx "BOARD_BITSTREAM=$BS" "$D/board.env" \
    || { echo "MKO STOP: changed sums [$diffs], board.env differs in $dl lines" >> "$O"; return 1; }
  echo "MKO kit $D from ${S##*/}: changed [$diffs]; board.env: BOARD_BITSTREAM=$BS" >> "$O"
  echo "MKO ${D##*/} DONE" >> "$O"
}
main "$@"; exit $?
