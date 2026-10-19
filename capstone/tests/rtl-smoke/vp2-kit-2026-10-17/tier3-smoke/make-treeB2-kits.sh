#!/bin/bash
# Tree B / board 2 kits (2026-10-19 evening; the runtime lane's discriminator for tree B's flash): the data-retry kits
# (monitor f13dbc5) with board.env changed to tree B's bitstream (caplifive_vp2c_8de3c48cf), board 2 (FPGA_BOARD=2, its URL
# file) and dev's driver directory (/tmp/capstone/wt-b2: the same run_baked_rungs_fpga.py fd38816a90c64c22, with the
# lease client in fpga_console.py). bkit7-rev2xBD, bkit7-rev2xmBD from bkit7-rev2xD, bkit7-rev2xmD; bkit7-pb10BD from
# bkit7-pb10D. board.env is sourced under set -a, so FPGA_BOARD and BOARD_URL_FILE reach the driver. Usage: make-treeB2-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke D=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke
  [ "$(sha256sum $D/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = fd38816a90c64c22 ] && grep -q lease_take $D/fpga_driver/fpga_console.py \
    || { echo "STOP: $D is not dev's driver with the lease client"; return 1; }
  mk() {
    local F=$H/$1 T=$H/$2
    [ -e "$T" ] && { echo "STOP: $T exists"; return 1; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F does not match its SHA256SUMS"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|; s|^BOARD_BITSTREAM=.*|BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf|; s|^BOARD_DRIVER_DIR=.*|BOARD_DRIVER_DIR=$D|" "$T/board.env"
    printf 'FPGA_BOARD=2\nBOARD_URL_FILE=%s\n' "$HOME/.claude-kisp/secrets/fpga-console-url-board2" >> "$T/board.env"
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  }
  mk bkit7-rev2xD bkit7-rev2xBD
  mk bkit7-rev2xmD bkit7-rev2xmBD
  mk bkit7-pb10D bkit7-pb10BD
}
main "$@"; exit $?
