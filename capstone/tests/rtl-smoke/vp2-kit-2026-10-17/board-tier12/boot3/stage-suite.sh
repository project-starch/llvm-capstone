#!/bin/bash
# vp2 Tier 2: replace the staged guest suite (vm/vm-suite-inner.sh), and nothing else. As stage-module.sh: the current suite
# must be the one named, the new one is checked against its hash and by `sh -n`, the old one moves to stage-attic/ by
# explicit name (nothing is deleted), and stage.SHA256SUMS is rewritten with exactly one line changed.
# Usage: stage-suite.sh <current suite sha256/16> <new suite path> <new suite sha256/16> <attic label>
main() {
  set -u
  local CUR=${1:?current suite hash} SP=${2:?new suite path} SH=${3:?new suite hash} LBL=${4:?attic label}
  local T=${T2_T:-/tmp/capstone/tier1}; local A=$T/stage-attic/suite-$LBL-$(date +%m%d-%H%M)
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not its record"; return 1; }
  [ "$(sha256sum $T/stage/vm/vm-suite-inner.sh | cut -c1-16)" = "$CUR" ] || { echo "STOP: the staged suite is not $CUR"; return 1; }
  [ -f "$SP" ] && [ "$(sha256sum "$SP" | cut -c1-16)" = "$SH" ] || { echo "STOP: $SP is not $SH"; return 1; }
  sh -n "$SP" || { echo "STOP: $SP does not parse"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm
  cp $T/stage.SHA256SUMS $A/stage-before.SHA256SUMS
  mv $T/stage/vm/vm-suite-inner.sh $A/vm/vm-suite-inner.sh
  cp "$SP" $T/stage/vm/vm-suite-inner.sh; chmod 755 $T/stage/vm/vm-suite-inner.sh
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED suite $SH; the previous stage record and suite in $A"
  diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
  [ "$(diff <(sort -k2 $A/stage-before.SHA256SUMS) <(sort -k2 $T/stage.SHA256SUMS) | grep -c '^>')" = 1 ] || { echo "STOP: more than one file changed"; return 1; }
}
main "$@"; exit $?
