#!/bin/bash
# The S-23a/S-26 monitor revision's kits (2026-10-19), each a copy of a recorded kit with board.env's lines changed and
# SHA256SUMS regenerated; nothing else in a kit changes.
#   bkit7-rev2xF, bkit7-rev2xmF  sql0-treeA's two kits (bkit7-rev2xA, bkit7-rev2xmA) on the fix monitor.
#   bkit7-pb10A                  pb10's panic kit (bkit7-fix6t: module d71f207a, BOARD_STEP_S=300) on tree A, old monitor.
#   bkit7-pb10F                  bkit7-pb10A on the fix monitor.
# "The fix monitor": capstone-sbi 5b5e86b (vp2/s23-strobe-guard: S-23a's guarded strobe on 007d81d) in /tmp/capstone/csbi-fix,
# wrapper aab63d5 (s26-boot-quantum on 1269c77) in /tmp/capstone/wt-wrap-s26, with CAPSTONE_QUANTUM_PRINT and
# CAPSTONE_BOOT_QUANTUM=2000000 added to the defines. Usage: make-fix-kits.sh. Refuses to overwrite a kit.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke
  local FIXD='BOARD_FW_DEFINES="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000"'
  [ "$(git -C /tmp/capstone/csbi-fix rev-parse --short=7 HEAD)" = 5b5e86b ] && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] \
    || { echo "STOP: the fix monitor or wrapper is not at 5b5e86b / aab63d5"; return 1; }
  mk() {   # mk <from> <to> <fix 0|1> <tree A 0|1>
    local F=$H/$1 T=$H/$2
    [ -e "$T" ] && { echo "STOP: $T exists"; return 1; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F does not match its SHA256SUMS"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    if [ "$4" = 1 ]; then sed -i 's|^BOARD_BITSTREAM=.*|BOARD_BITSTREAM=caplifive_vp2c_8522d72fe|' "$T/board.env"; fi
    if [ "$3" = 1 ]; then
      sed -i 's|^BOARD_FW_WRAP=.*|BOARD_FW_WRAP=/tmp/capstone/wt-wrap-s26|; s|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-fix|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=5b5e86b|' "$T/board.env"
      sed -i "s|^BOARD_FW_DEFINES=.*|$FIXD|" "$T/board.env"
    fi
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-rev2xA bkit7-rev2xF 1 0
  mk bkit7-rev2xmA bkit7-rev2xmF 1 0
  mk bkit7-fix6t bkit7-pb10A 0 1
  mk bkit7-pb10A bkit7-pb10F 1 0
}
main "$@"; exit $?
