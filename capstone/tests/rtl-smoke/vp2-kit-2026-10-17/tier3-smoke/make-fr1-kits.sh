#!/bin/bash
# The fault-regs-1 kits (2026-10-20; the sql0 dump arm): bkit7-rev2xBDF1 / bkit7-rev2xmBDF1 are bkit7-rev2xBDN2 /
# bkit7-rev2xmBDN2 (tree B, board 2, monitor f13dbc5; the kits of sql0-nf2B) with ONLY adapter/module/capstone_vm.ko changed:
# node-floor-2 54d8a3af replaced by fault-regs-1 0d92d06a (compiler lane module-fault-regs a486a9666fd9: node-floor-2 plus
# the FREG fatal-fault dump). The module file is replaced by removal and checked by hash. Usage: make-fr1-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/modfix/fr1-board-flat/module/capstone_vm.ko F T
  [ "$(sha256sum $M | cut -c1-16)" = 0d92d06a2abc6298 ] || { echo "STOP: the module is not 0d92d06a"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    rm -f "${T:?}/adapter/module/capstone_vm.ko"; cp "$M" "$T/adapter/module/capstone_vm.ko"
    [ "$(sha256sum "$T/adapter/module/capstone_vm.ko" | cut -c1-16)" = 0d92d06a2abc6298 ] || { echo "STOP: $T's module did not take"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff -r "$F" "$T" 2>&1 | grep -v SHA256SUMS | grep -c .) differing lines/files outside SHA256SUMS"
  }
  mk bkit7-rev2xBDN2 bkit7-rev2xBDF1
  mk bkit7-rev2xmBDN2 bkit7-rev2xmBDF1
}
main "$@"; exit $?
