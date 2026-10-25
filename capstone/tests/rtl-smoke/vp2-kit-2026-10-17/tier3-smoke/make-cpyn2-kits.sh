#!/bin/bash
# The cpython node-floor-2 kits (2026-10-20; the rebuilt-image cpython arms): bkit7-rev2xnDN2 is bkit7-rev2xnD (tree A,
# board 1) and bkit7-rev2xnBDN2 is bkit7-rev2xnBD (tree B, board 2), both monitor f13dbc5 with the norandmaps bootargs, with
# ONLY adapter/module/capstone_vm.ko changed: 39d8492d replaced by dev's module 54d8a3af (srz + the node floor; Tier 1 PASS on
# both trees), taken from the sql0 kit bkit7-rev2xDN2. Derived from make-cpyz-kit.sh. Usage: make-cpyn2-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2xDN2/adapter/module/capstone_vm.ko F T
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
  mk bkit7-rev2xnD bkit7-rev2xnDN2
  mk bkit7-rev2xnBD bkit7-rev2xnBDN2
}
main "$@"; exit $?
