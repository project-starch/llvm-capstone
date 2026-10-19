#!/bin/bash
# The fetch-slot kits (2026-10-20; sql0-frs): bkit7-rev2xBFZ / bkit7-rev2xmBFZ are bkit7-rev2xBDZ / bkit7-rev2xmBDZ (tree B,
# board 2, the srevroot-zero module 607164b9, the data retry) on capstone-sbi 9697173 (vp2/s23-fetch-slots: c3917e8 plus
# FRSL/FRT2/FRVD at a refused fetch, under CAPSTONE_S23_FETCH_SLOTS), with that define added. Only the monitor directory,
# its commit and the one define change. Usage: make-frs-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke F T
  [ "$(git -C /tmp/capstone/csbi-frs rev-parse --short=7 HEAD)" = 9697173 ] && [ -z "$(git -C /tmp/capstone/csbi-frs status --porcelain)" ] || { echo "STOP: csbi-frs"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|; s|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-frs|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=9697173|; s|^\(BOARD_FW_DEFINES=\".*-DCAPSTONE_S23_DATA_RETRY\)\"$|\1 -DCAPSTONE_S23_FETCH_SLOTS\"|" "$T/board.env"
    grep -q '^BOARD_FW_DEFINES=".* -DCAPSTONE_S23_DATA_RETRY -DCAPSTONE_S23_FETCH_SLOTS"$' "$T/board.env" || { echo "STOP: $T's defines did not take the new one"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-rev2xBDZ bkit7-rev2xBFZ
  mk bkit7-rev2xmBDZ bkit7-rev2xmBFZ
}
main "$@"; exit $?
