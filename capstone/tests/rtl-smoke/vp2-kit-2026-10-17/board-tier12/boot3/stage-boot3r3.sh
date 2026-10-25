#!/bin/bash
# vp2 Tier 1 boot 3-T1's THIRD run: the repeat's stage with ONE file changed, the board module. The compiler lane's
# revision 457b78f27e10 (Q-26: contexts serialized on one slot; Q-30: the reply slot overwritten, not required empty).
# The repeat's module moves to stage-attic/ by explicit name (nothing is deleted); the new one is checked against the
# hash the handoff names, run through the kit gates that read a module, and stage.SHA256SUMS is rewritten.
# Usage: stage-boot3r3.sh <board ko path> <board ko sha256/16>
main() {
  set -u
  local KOP=${1:?board ko path} KO=${2:?board ko hash}
  local T=${T2_T:-/tmp/capstone/tier1}; local A=$T/stage-attic/vm-boot3r-$(date +%m%d-%H%M)
  local G=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not the repeat's record"; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = ed24f827b5beb373 ] || { echo "STOP: the staged module is not the repeat's ed24f827"; return 1; }
  [ -f "$KOP" ] && [ "$(sha256sum "$KOP" | cut -c1-16)" = "$KO" ] || { echo "STOP: $KOP is not $KO"; return 1; }
  # The kit gates that read a module (the images are the repeat's, already gated): 0 QEMU-only CSR accesses, and the
  # silicon gate in its trusted-S column.
  python3 $G/qemu-only-csr-census.py "$KOP" || { echo "STOP: qemu-only-csr-census is not 0 (or no data)"; return 1; }
  python3 $G/vp2-silicon-gate.py "$KOP" || { echo "STOP: vp2-silicon-gate is not clean on the module (or no data)"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm
  cp $T/stage.SHA256SUMS $A/stage-boot3r.SHA256SUMS
  mv $T/stage/vm/capstone_vm.ko $A/vm/capstone_vm.ko
  cp "$KOP" $T/stage/vm/capstone_vm.ko
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED module $KO; the repeat's stage record and module in $A"
  echo "changed from the repeat:"; diff <(sort -k2 $A/stage-boot3r.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
  [ "$(diff <(sort -k2 $A/stage-boot3r.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep -c '^>')" = 1 ] || { echo "STOP: more than one file changed"; return 1; }
}
main "$@"; exit $?
