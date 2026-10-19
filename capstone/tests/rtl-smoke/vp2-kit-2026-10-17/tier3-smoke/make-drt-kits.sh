#!/bin/bash
# The data-retry kits (2026-10-19 evening): the refusal-read kits on capstone-sbi f13dbc5 (vp2/s23-data-retry: 03a1ffe plus
# CAPSTONE_S23_DATA_RETRY, S-23b step 1, DRT1/DRTX), with that define added. bkit7-rev2xD, bkit7-rev2xmD from
# bkit7-rev2xR, bkit7-rev2xmR; bkit7-pb10D from bkit7-pb10R. Nothing else changes. Usage: make-drt-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke
  local FIXD='BOARD_FW_DEFINES="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT -DCAPSTONE_BOOT_QUANTUM=2000000 -DCAPSTONE_S23_REFUSAL_READ -DCAPSTONE_S23_DATA_RETRY"'
  [ "$(git -C /tmp/capstone/csbi-drt rev-parse --short=7 HEAD)" = f13dbc5 ] && [ -z "$(git -C /tmp/capstone/csbi-drt status --porcelain)" ] \
    && [ "$(git -C /tmp/capstone/wt-wrap-s26 rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "STOP: the data-retry monitor or wrapper is not at f13dbc5 / aab63d5"; return 1; }
  mk() {   # mk <from> <to> <fix 0|1> <tree A 0|1>
    local F=$H/$1 T=$H/$2
    [ -e "$T" ] && { echo "STOP: $T exists"; return 1; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F does not match its SHA256SUMS"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    if [ "$4" = 1 ]; then sed -i 's|^BOARD_BITSTREAM=.*|BOARD_BITSTREAM=caplifive_vp2c_8522d72fe|' "$T/board.env"; fi
    if [ "$3" = 1 ]; then
      sed -i 's|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-drt|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=f13dbc5|' "$T/board.env"
      sed -i "s|^BOARD_FW_DEFINES=.*|$FIXD|" "$T/board.env"
    fi
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-rev2xR bkit7-rev2xD 1 0
  mk bkit7-rev2xmR bkit7-rev2xmD 1 0
  mk bkit7-pb10R bkit7-pb10D 1 0
}
main "$@"; exit $?
