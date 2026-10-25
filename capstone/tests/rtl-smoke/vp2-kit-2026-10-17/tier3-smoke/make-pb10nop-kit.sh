#!/bin/bash
# The pb10 srz-nop kit (2026-10-20; the runtime lane's pb10 anomaly pair): bkit7-pb10BN is bkit7-pb10BZ (tree B, board 2, monitor
# f13dbc5, the srz module 607164b9 of pb10-srz) with ONLY adapter/module/capstone_vm.ko changed, to the compiler lane's srz-nop
# variant 2a8f85f7 (lane srz-nop-pair 5260ec650af4: .text differs in 3 words, csrw 0x5c1,x0 -> nop). Usage: make-pb10nop-kit.sh.
main() {
  set -euo pipefail
  local H=/mnt/hdd/$USER/capstone/t3-smoke M=/mnt/hdd/$USER/capstone/modfix/srznop-board-flat/module/capstone_vm.ko F T
  [ "$(sha256sum $M | cut -c1-16)" = 2a8f85f70d8efcdd ] || { echo "STOP: the module is not 2a8f85f7"; return 1; }
  mk() {
    F=$H/$1; T=$H/$2
    [ -e "$T" ] && { echo "SKIP: $T exists"; return 0; }
    ( cd "$F" && sha256sum -c --quiet SHA256SUMS ) || { echo "STOP: $F"; return 1; }
    cp -a "$F" "$T"
    sed -i "s|^BOARD_BASE_MANIFEST=.*|BOARD_BASE_MANIFEST=$T/base.manifest|" "$T/board.env"
    rm -f "${T:?}/adapter/module/capstone_vm.ko"; cp "$M" "$T/adapter/module/capstone_vm.ko"
    [ "$(sha256sum "$T/adapter/module/capstone_vm.ko" | cut -c1-16)" = 2a8f85f70d8efcdd ] || { echo "STOP: $T's module did not take"; return 1; }
    ( cd "$T" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
    echo "KIT $2 from $1: $(diff -r "$F" "$T" 2>&1 | grep -v SHA256SUMS | grep -c .) differing lines/files outside SHA256SUMS"
  }
  mk bkit7-pb10BZ bkit7-pb10BN
}
main "$@"; exit $?
