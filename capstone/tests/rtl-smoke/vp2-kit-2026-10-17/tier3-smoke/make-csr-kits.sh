#!/bin/bash
# The refusal-CSR kits (2026-10-19 evening; sql0's two added for sql0-csr): bkit7-pb10C is bkit7-pb10D, and bkit7-rev2xC /
# bkit7-rev2xmC are bkit7-rev2xD / bkit7-rev2xmD, (tree A, module d71f207a, BOARD_STEP_S=300, the data
# retry) on capstone-sbi c3917e8 (vp2/s23-csr-read: f13dbc5 plus RFSR/RFSH/RFCT at an S/U data refusal and FRTM beside FRTY,
# under the same CAPSTONE_S23_REFUSAL_READ). Only the monitor directory and commit change. Usage: make-csr-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke F T
  [ "$(git -C /tmp/capstone/csbi-csr rev-parse --short=7 HEAD)" = c3917e8 ] && [ -z "$(git -C /tmp/capstone/csbi-csr status --porcelain)" ] || { echo "STOP: csbi-csr"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|; s|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-csr|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=c3917e8|" "$T/board.env"
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-pb10D bkit7-pb10C
  mk bkit7-rev2xD bkit7-rev2xC
  mk bkit7-rev2xmD bkit7-rev2xmC
}
main "$@"; exit $?
