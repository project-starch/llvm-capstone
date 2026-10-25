#!/bin/bash
# vp2 Tier 1 rev8 for module 39d8492d: from rev7's stage, replace vm-contract.sh and vm-suite-inner.sh (rev8's, with
# @NFR@ filled in) and add contract-dense.dom (the compiler lane's dense-churn image), the image through the kit gates
# first. The old files are moved to the attic by name, nothing is deleted, and stage.SHA256SUMS is rewritten with
# exactly three lines new. Usage: stage-rev8.sh <node_flat_records>. Ends by itself.
main() {
  set -u
  local NFR=${1:?node_flat_records} T=/tmp/capstone/tier1 S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local DOM=/mnt/hdd/$USER/capstone/dense-churn/contract-dense.dom A=$T/stage-attic/rev8-$(date +%m%d-%H%M)
  local G=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17 f
  case "$NFR" in ''|*[!0-9]*) echo "STOP: node_flat_records '$NFR' is not a number"; return 1;; esac
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not its record"; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] || { echo "STOP: the staged module is not 39d8492d"; return 1; }
  for f in vm-contract.sh vm-suite-inner.sh; do
    cmp -s $T/stage/vm/$f $G/board-tier12/rev7/$f || { echo "STOP: the staged $f is not rev7's"; return 1; }
  done
  [ -e $T/stage/vm/contract-dense.dom ] && { echo "STOP: contract-dense.dom is already staged"; return 1; }
  [ "$(sha256sum $DOM | cut -c1-16)" = b6a11ff3ba726646 ] || { echo "STOP: $DOM is not b6a11ff3"; return 1; }
  python3 $G/qemu-only-csr-census.py "$DOM" || { echo "STOP: qemu-only-csr-census is not 0 on the image"; return 1; }
  python3 $G/vp2-silicon-gate.py --attributions $S/silicon-gate-attributions.tsv "$DOM" || { echo "STOP: vp2-silicon-gate is not clean on the image"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm; cp $T/stage.SHA256SUMS $A/stage-before.SHA256SUMS
  for f in vm-contract.sh vm-suite-inner.sh; do
    mv $T/stage/vm/$f $A/vm/$f
    sed "s/@NFR@/$NFR/g" $S/$f > $T/stage/vm/$f; chmod 0755 $T/stage/vm/$f
    grep -q "@NFR@" $T/stage/vm/$f && { echo "STOP: @NFR@ left in $f"; return 1; }
  done
  cp "$DOM" $T/stage/vm/contract-dense.dom
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED rev8 (node_flat_records=$NFR); the previous files in $A"
  diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
  [ "$(diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep -c '^>')" = 3 ] || { echo "STOP: not exactly three lines new"; return 1; }
}
main "$@"; exit $?
