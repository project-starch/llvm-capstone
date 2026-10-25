#!/bin/bash
# The cpython srz kit (2026-10-20; cpyp-DZ): bkit7-rev2xnDZ is bkit7-rev2xnD (tree A, board 1, monitor f13dbc5, the cpython
# arm's norandmaps kit) with ONLY adapter/module/capstone_vm.ko changed: 39d8492d replaced by the srevroot-zero module
# 607164b9 (the module of sql0-srz-treeB and pb10-srz). Derived from make-nf-kits.sh. Usage: make-cpyz-kit.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/modfix/srz-board-flat/module/capstone_vm.ko F T
  [ "$(sha256sum $M | cut -c1-16)" = 607164b9c1407c7c ] || { echo "STOP: the module is not 607164b9"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    rm -f "${T:?}/adapter/module/capstone_vm.ko"; cp "$M" "$T/adapter/module/capstone_vm.ko"
    [ "$(sha256sum "$T/adapter/module/capstone_vm.ko" | cut -c1-16)" = 607164b9c1407c7c ] || { echo "STOP: $T's module did not take"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff -r "$F" "$T" 2>&1 | grep -v SHA256SUMS | grep -c .) differing lines/files outside SHA256SUMS"
  }
  mk bkit7-rev2xnD bkit7-rev2xnDZ
}
main "$@"; exit $?
