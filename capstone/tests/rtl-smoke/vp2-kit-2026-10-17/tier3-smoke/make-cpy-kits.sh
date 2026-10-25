#!/bin/bash
# cpython lt0's board kit on the S-23 retry monitor (2026-10-19 evening): bkit7-rev2xnD is bkit7-rev2xnA (tree A, module
# 39d8492d, norandmaps bootargs through the bake's saved dtb, execute-bit rule, BOARD_TIMEOUT_MULT=4) with ONLY the monitor
# fields changed to the data-retry kits' (bkit7-rev2xD): wrapper wt-wrap-s26 (aab63d5), capstone-sbi f13dbc5 in csbi-drt,
# and its defines. Usage: make-cpy-kits.sh. Refuses to overwrite.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke F T
  F=$H/bkit7-rev2xnA; T=$H/bkit7-rev2xnD
  [ -e "$T" ] && { echo "STOP: $T exists"; return 1; }
  [ "$(git -C /tmp/capstone/csbi-drt rev-parse --short=7 HEAD)" = f13dbc5 ] && [ -z "$(git -C /tmp/capstone/csbi-drt status --porcelain)" ] \
    || { echo "STOP: csbi-drt is not f13dbc5, clean"; return 1; }
  ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F does not match its SHA256SUMS"; return 1; }
  cp -a "$F" "$T"
  local D; D=$(grep '^BOARD_FW_DEFINES=' $H/bkit7-rev2xD/board.env)
  sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|; s|^BOARD_FW_WRAP=.*|BOARD_FW_WRAP=/tmp/capstone/wt-wrap-s26|; s|^BOARD_FW_MON=.*|BOARD_FW_MON=/tmp/capstone/csbi-drt|; s|^BOARD_MONITOR_COMMIT=.*|BOARD_MONITOR_COMMIT=f13dbc5|" "$T/board.env"
  sed -i "s|^BOARD_FW_DEFINES=.*|$D|" "$T/board.env"
  ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
  echo "KIT bkit7-rev2xnD from bkit7-rev2xnA: $(diff "$F/board.env" "$T/board.env" | grep '^>' | tr '\n' ' ')"
  diff <(grep -v '^BOARD_BASE_MANIFEST\|^BOARD_DTB' "$T/board.env") <(grep -v '^BOARD_BASE_MANIFEST' $H/bkit7-rev2xD/board.env) && echo "board.env equals bkit7-rev2xD's but for the manifest path and the bootargs"
}
main "$@"; exit $?
