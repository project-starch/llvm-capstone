#!/bin/bash
# The dev-srz kits (2026-10-20; sql0-srzdev-treeB, the node floor's one-variable control): bkit7-rev2xBDZD / bkit7-rev2xmBDZD are
# bkit7-rev2xBDZ / bkit7-rev2xmBDZ with ONLY adapter/module/capstone_vm.ko changed, to dev's srevroot-zero module 90bf036f
# (dev 178e005899b0), which differs from the node-floor module a4aa5668 only by the floor commit (770daec2dbd6).
# Derived from make-nf-kits.sh. Usage: make-srzd-kits.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/modfix/srzdev-board-flat/module/capstone_vm.ko F T
  [ "$(sha256sum $M | cut -c1-16)" = 90bf036f59da1231 ] || { echo "STOP: the module is not 90bf036f"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    rm -f "${T:?}/adapter/module/capstone_vm.ko"; cp "$M" "$T/adapter/module/capstone_vm.ko"
    [ "$(sha256sum "$T/adapter/module/capstone_vm.ko" | cut -c1-16)" = 90bf036f59da1231 ] || { echo "STOP: $T's module did not take"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff -r "$F" "$T" 2>&1 | grep -v SHA256SUMS | grep -c .) differing lines/files outside SHA256SUMS"
  }
  mk bkit7-rev2xBDZ bkit7-rev2xBDZD
  mk bkit7-rev2xmBDZ bkit7-rev2xmBDZD
}
main "$@"; exit $?
