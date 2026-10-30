#!/bin/bash
# Bitstream 4's board kit with the S-24 fix on (PREREG Addendum 17): a copy of make-b4-kit.sh's bkit13-b4-b2 with
# EXACTLY one change, checked by hash: a new adapter/module/params holding `resolve_dirty=1` (the S-24 fix) and
# `resolve_pte=1` (it only adds the PTE value to S-24's bound line, if that line fires). boardvm (virtualvm's
# module_params) writes each through sysfs right after insmod, prints `@@PARAM <name>=<read back>` and refuses the run
# unless each reads Y. The module (6422805f49533ea5), launcher (62753dd693cc4a25), board.env and recipe are bkit13's.
# Usage: make-b4r-kit.sh <bkit13 dir> <new kit> <log>.
main() {
  set -u
  local S=${1:?source kit} D=${2:?new kit} O=${3:?log} diffs
  ( cd $S && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_325966ee4" $S/board.env \
    && [ "$(sha256sum $S/adapter/module/capstone_vm.ko | cut -c1-16)" = 6422805f49533ea5 ] && [ ! -e $S/adapter/module/params ] \
    || { echo "MK4R STOP: $S is not bkit13-b4 (sums, bitstream, module, or it already has params)" >> "$O"; return 1; }
  [ -e $D ] && { echo "MK4R STOP: $D exists" >> "$O"; return 1; }
  cp -a $S $D || { echo "MK4R STOP: copy" >> "$O"; return 1; }
  printf '# PREREG Addendum 17: the S-24 fix (resolve_dirty), the module owner'"'"'s decision, and the PTE print on its\n# bound line (resolve_pte, print only); both read back by boardvm\nresolve_dirty=1\nresolve_pte=1\n' \
    > $D/adapter/module/params
  ( cd $D && sha256sum ./adapter/module/params >> SHA256SUMS && sha256sum -c --quiet SHA256SUMS ) || { echo "MK4R STOP: sums" >> "$O"; return 1; }
  diffs=$(diff <(sort -k2 $S/SHA256SUMS) <(sort -k2 $D/SHA256SUMS) | grep '^[<>]' | awk '{print $1 $3}' | tr '\n' ' ')
  [ "$diffs" = ">./adapter/module/params " ] || { echo "MK4R STOP: changed sums [$diffs]" >> "$O"; return 1; }
  echo "MK4R kit ${D##*/} from ${S##*/}: added adapter/module/params $(sha256sum $D/adapter/module/params | cut -c1-16) (resolve_dirty=1 resolve_pte=1)" >> "$O"
}
main "$@"; exit $?
