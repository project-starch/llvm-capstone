#!/bin/bash
# The refusal-read kits (2026-10-19 evening): the fix kits on capstone-sbi 03a1ffe (vp2/s23-refusal-read: 5b5e86b plus
# CAPSTONE_S23_REFUSAL_READ, the LCC type and validity of each loaded slot covering mtval at a refusal), with that define
# added. bkit7-rev2xR, bkit7-rev2xmR from bkit7-rev2xF, bkit7-rev2xmF; bkit7-pb10R from bkit7-pb10F. Nothing else changes.
# Usage: make-rf-kits.sh. Refuses to overwrite a kit.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke
  local FIXD='BOARD_FW_DEFINES="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000 -DCAPSTONE_S23_REFUSAL_READ"'
  [ "$(git -C /tmp/capstone/csbi-refusal rev-parse --short=7 HEAD)" = 03a1ffe ] && [ -z "$(git -C /tmp/capstone/csbi-refusal status --porcelain)" ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "STOP: the refusal monitor or wrapper is not at 03a1ffe / aab63d5"; return 1; }
  mk() {   # mk <from> <to> <fix 0|1> <tree A 0|1>
    local F=$H/$1 T=$H/$2
    [ -e "$T" ] && { echo "STOP: $T exists"; return 1; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F does not match its SHA256SUMS"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    if [ "$4" = 1 ]; then sed -i 's|^BOARD_BITSTREAM=.*|BOARD_BITSTREAM=caplifive_vp2c_8522d72fe|' "$T/board.env"; fi
    if [ "$3" = 1 ]; then
      sed -i 's|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-refusal|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=03a1ffe|' "$T/board.env"
      sed -i "s|^BOARD_FW_DEFINES=.*|$FIXD|" "$T/board.env"
    fi
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-rev2xF bkit7-rev2xR 1 0
  mk bkit7-rev2xmF bkit7-rev2xmR 1 0
  mk bkit7-pb10F bkit7-pb10R 1 0
}
main "$@"; exit $?
