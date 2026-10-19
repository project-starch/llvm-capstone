#!/bin/bash
# The node-floor-2 kits (2026-10-20; sql0-nf2-treeB): bkit7-rev2xBDN2 / bkit7-rev2xmBDN2 are bkit7-rev2xBDZ / bkit7-rev2xmBDZ (tree B,
# board 2, monitor f13dbc5) with ONLY adapter/module/capstone_vm.ko changed: the srevroot-zero module 607164b9 replaced by the
# node-floor module 54d8a3af (runtime lane c2b843ad88ad; dev's srz module plus the context-table index floor). Usage:
# make-nf-kits.sh. The module file is replaced by removal (it may be read-only) and checked by hash.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/modfix/nf2-board-flat/module/capstone_vm.ko F T
  [ "$(sha256sum $M | cut -c1-16)" = 54d8a3af1158c3f4 ] || { echo "STOP: the module is not 54d8a3af"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    rm -f "${T:?}/adapter/module/capstone_vm.ko"; cp "$M" "$T/adapter/module/capstone_vm.ko"
    [ "$(sha256sum "$T/adapter/module/capstone_vm.ko" | cut -c1-16)" = 54d8a3af1158c3f4 ] || { echo "STOP: $T's module did not take"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff -r "$F" "$T" 2>&1 | grep -v SHA256SUMS | grep -c .) differing lines/files outside SHA256SUMS"
  }
  mk bkit7-rev2xBDZ bkit7-rev2xBDN2
  mk bkit7-rev2xmBDZ bkit7-rev2xmBDN2
}
main "$@"; exit $?
