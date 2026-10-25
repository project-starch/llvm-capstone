#!/bin/bash
# vp2 Tier 1: replace the staged board module, and nothing else. The current module must be the one named (so a stage
# cannot drift), the new one is checked against its hash and through the kit gates that read a module, the old one
# moves to stage-attic/ by explicit name (nothing is deleted), and stage.SHA256SUMS is rewritten with exactly one line
# changed. (Generalizes stage-boot3r3.sh.)
# Usage: stage-module.sh <current ko sha256/16> <new ko path> <new ko sha256/16> <attic label>
main() {
  set -u
  local CUR=${1:?current ko hash} KOP=${2:?new ko path} KO=${3:?new ko hash} LBL=${4:?attic label}
  local T=${T2_T:-/tmp/capstone/tier1}; local A=$T/stage-attic/vm-$LBL-$(date +%m%d-%H%M)
  local G=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not its record"; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = "$CUR" ] || { echo "STOP: the staged module is not $CUR"; return 1; }
  [ -f "$KOP" ] && [ "$(sha256sum "$KOP" | cut -c1-16)" = "$KO" ] || { echo "STOP: $KOP is not $KO"; return 1; }
  python3 $G/qemu-only-csr-census.py "$KOP" || { echo "STOP: qemu-only-csr-census is not 0 (or no data)"; return 1; }
  python3 $G/vp2-silicon-gate.py "$KOP" || { echo "STOP: vp2-silicon-gate is not clean on the module (or no data)"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm
  cp $T/stage.SHA256SUMS $A/stage-before.SHA256SUMS
  mv $T/stage/vm/capstone_vm.ko $A/vm/capstone_vm.ko
  cp "$KOP" $T/stage/vm/capstone_vm.ko
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED module $KO; the previous stage record and module in $A"
  diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
  [ "$(diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep -c '^>')" = 1 ] || { echo "STOP: more than one file changed"; return 1; }
}
main "$@"; exit $?
