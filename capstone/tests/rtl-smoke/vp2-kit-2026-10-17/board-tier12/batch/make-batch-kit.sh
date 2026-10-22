#!/bin/bash
# The batch bitstream's board kit, made AFTER the lead's reflash word and the flash, never before: a copy of the flashed
# board's current kit with exactly three changes, checked by diff against it:
#   BOARD_BITSTREAM=<the flashed bitstream's name>;
#   BOARD_FW_MON / BOARD_MONITOR_COMMIT = the batch monitor line (capstone-sbi vp2/batch d9da397, /tmp/capstone/csbi-batch)
#   and BOARD_FW_DEFINES + -DCAPSTONE_FETCH_RETRY_CAP (batch-prep.sh's recipe);
#   adapter/module/capstone_vm.ko = the batch module 21fbfc397e7c7a42 (runtime lane module-gen-push 64747600ecec).
# SHA256SUMS regenerated and checked. Usage: make-batch-kit.sh <board 1|2> <bitstream name, no .bit> <log>. MK_X stubs the root.
main() {
  set -u
  local BRD=${1:?board} BS=${2:?bitstream name} O=${3:?log} H=/mnt/hdd/$USER/capstone S D
  local M=/mnt/hdd/$USER/capstone/t3-smoke/bkit8-batch-parts/capstone_vm.ko MON=/tmp/capstone/csbi-batch
  case $BRD in 1) S=$H/t3-smoke/bkit7-rev2xDN2;; 2) S=$H/t3-smoke/bkit7-rev2xBDN2;; *) echo "MK STOP: board $BRD" >> "$O"; return 2;; esac
  D=${MK_X:-$H/t3-smoke}/bkit8-batch-b$BRD
  case $BS in caplifive_*) ;; *) echo "MK STOP: bitstream name $BS" >> "$O"; return 1;; esac
  ( cd $S && sha256sum -c --quiet SHA256SUMS ) || { echo "MK STOP: base kit $S" >> "$O"; return 1; }
  [ "$(sha256sum $M | cut -c1-16)" = 21fbfc397e7c7a42 ] || { echo "MK STOP: the batch module is not 21fbfc39" >> "$O"; return 1; }
  [ "$(git -C $MON rev-parse --short=7 HEAD)" = d9da397 ] && [ -z "$(git -C $MON status --porcelain)" ] || { echo "MK STOP: $MON is not a clean d9da397" >> "$O"; return 1; }
  [ -e $D ] && { echo "MK STOP: $D exists" >> "$O"; return 1; }
  cp -a $S $D && cp $M $D/adapter/module/capstone_vm.ko || { echo "MK STOP: copy" >> "$O"; return 1; }
  python3 - $D/board.env "$BS" "$MON" <<'PY' || { echo "MK STOP: board.env edit" >> "$O"; return 1; }
import re, sys
p, bs, mon = sys.argv[1:]
s = open(p).read()
for pat, rep in [(r"^BOARD_BITSTREAM=.*$", f"BOARD_BITSTREAM={bs}"), (r"^BOARD_FW_MON=.*$", f"BOARD_FW_MON={mon}"),
                 (r"^BOARD_MONITOR_COMMIT=.*$", "BOARD_MONITOR_COMMIT=d9da397"),
                 (r'^BOARD_FW_DEFINES="(.*)"$', r'BOARD_FW_DEFINES="\1 -DCAPSTONE_FETCH_RETRY_CAP"')]:
    s, n = re.subn(pat, rep, s, flags=re.M)
    assert n == 1, pat
open(p, "w").write(s)
PY
  ( cd $D && sha256sum ./adapter/capstone-vexec ./adapter/module/capstone_vm.ko ./bake-manifest.sh ./base.manifest ./board.env ./board-run.sh > SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) || { echo "MK STOP: sums" >> "$O"; return 1; }
  echo "MK kit $D: board.env diff against $S:" >> "$O"; diff $S/board.env $D/board.env | grep '^[<>]' | grep -v -i 'url\|token' >> "$O"
  echo "MK module $(sha256sum $D/adapter/module/capstone_vm.ko | cut -c1-16) DONE" >> "$O"
}
main "$@"; exit $?
