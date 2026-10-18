#!/bin/bash
# vp2 Tier 1 for module 39d8492d: replace exactly three staged files -- the module (d71f207a -> 39d8492d), vm-contract.sh and
# vm-suite-inner.sh (rev7's) -- with the module through the kit gates that read a module, the old files moved to the
# attic by name (nothing deleted), and stage.SHA256SUMS rewritten with exactly three lines changed. Ends by itself.
main() {
  set -u
  local T=/tmp/capstone/tier1 S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local KO=/mnt/hdd/$USER/capstone/board-modules/capstone_vm-39d8492d5c18216b.ko A=$T/stage-attic/rev7-$(date +%m%d-%H%M)
  local G=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17 f
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not its record"; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "STOP: the staged module is not d71f207a"; return 1; }
  [ "$(sha256sum $KO | cut -c1-16)" = 39d8492d5c18216b ] || { echo "STOP: $KO is not 39d8492d"; return 1; }
  python3 $G/qemu-only-csr-census.py "$KO" || { echo "STOP: qemu-only-csr-census is not 0"; return 1; }
  python3 $G/vp2-silicon-gate.py "$KO" || { echo "STOP: vp2-silicon-gate is not clean"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm; cp $T/stage.SHA256SUMS $A/stage-before.SHA256SUMS
  for f in capstone_vm.ko vm-contract.sh vm-suite-inner.sh; do mv $T/stage/vm/$f $A/vm/$f; done
  cp "$KO" $T/stage/vm/capstone_vm.ko; cp $S/vm-contract.sh $S/vm-suite-inner.sh $T/stage/vm/
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED module 39d8492d and rev7's two scripts; the previous files in $A"
  diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
  [ "$(diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep -c '^>')" = 3 ] || { echo "STOP: not exactly three files changed"; return 1; }
}
main "$@"; exit $?
