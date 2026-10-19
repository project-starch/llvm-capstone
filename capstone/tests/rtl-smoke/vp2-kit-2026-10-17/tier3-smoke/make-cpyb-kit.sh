#!/bin/bash
# The cpython tree B kit (2026-10-20; cpyp-DB): bkit7-rev2xnBD is bkit7-rev2xnD (monitor f13dbc5, module 39d8492d, the
# norandmaps bootargs) with ONLY the board fields changed as bkit7-rev2xBD changed bkit7-rev2xD: tree B's bitstream
# caplifive_vp2c_8de3c48cf, board 2's driver directory (dev's lease client), FPGA_BOARD=2 and board 2's URL file.
# Usage: make-cpyb-kit.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke F T
  F=$H/bkit7-rev2xnD; T=$H/bkit7-rev2xnBD
  [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
  ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
  cp -a "$F" "$T"
  sed -i "s|^BOARD_BITSTREAM=.*|BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf|; s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|; s|^BOARD_DRIVER_DIR=.*|BOARD_DRIVER_DIR=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke|" "$T/board.env"
  printf 'FPGA_BOARD=2\nBOARD_URL_FILE=%s/.claude-kisp/secrets/fpga-console-url-board2\n' "$HOME" >> "$T/board.env"
  ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
  echo "KIT bkit7-rev2xnBD: $(diff "$F/board.env" "$T/board.env" | grep -c '^>') changed board.env lines"
}
main "$@"; exit $?
