#!/bin/bash
# vp2 Tier 1 boot 3: re-stage /tmp/capstone/tier1/stage from the compiler lane's C-27-fixed kit. Boot 2's stage moves to
# stage-attic/ by explicit name (nothing is deleted), the new files are copied by explicit name, each is checked against
# the hash the handoff names, and stage.SHA256SUMS is rewritten (boot 1's kept beside the attic copy).
# Usage: stage-boot3.sh <kit out dir> <board ko sha256/16> <contract.dom sha256/16> <sqlite3.dom sha256/16> <mruby sha256/16>
main() {
  set -u
  local O=${1:?kit out dir} KO=${2:?board ko hash} CT=${3:?contract.dom hash} SQ=${4:?sqlite3.dom hash} MR=${5:?mruby hash}
  local T=${T2_T:-/tmp/capstone/tier1}; local A=$T/stage-attic/vm-boot2-$(date +%m%d-%H%M)
  local L=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the current stage is not boot 2's record"; return 1; }
  # sources, by explicit name
  local -A SRC=(
    [vm/capstone_vm.ko]=$O/adapter-board/module/capstone_vm.ko
    [vm/capstone-vexec]=$O/adapter-board/capstone-vexec
    [vm/capstone-job]=$O/adapter-board/capstone-job
    [vm/contract.dom]=$O/contract.dom
    [vm/sqlite3.dom]=$O/sqlite/sqlite3.dom
    [vm/mruby.dom]=$O/mruby/src/mruby/build/capstone/bin/mruby
    [vm/vm-contract.sh]=$L/vm-contract.sh
    [vm/vm-suite-inner.sh]=$L/vm-suite-inner.sh
    [vmrun.sh]=$L/vmrun.sh )
  local k
  for k in "${!SRC[@]}"; do [ -f "${SRC[$k]}" ] || { echo "STOP: missing ${SRC[$k]}"; return 1; }; done
  [ "$(sha256sum ${SRC[vm/capstone_vm.ko]} | cut -c1-16)" = "$KO" ] || { echo "STOP: the kit's board ko is not $KO"; return 1; }
  [ "$(sha256sum ${SRC[vm/sqlite3.dom]} | cut -c1-16)" = "$SQ" ] || { echo "STOP: the kit's sqlite3.dom is not $SQ"; return 1; }
  [ "$(sha256sum ${SRC[vm/contract.dom]} | cut -c1-16)" = "$CT" ] || { echo "STOP: the kit's contract.dom is not $CT"; return 1; }
  [ "$(sha256sum ${SRC[vm/mruby.dom]} | cut -c1-16)" = "$MR" ] || { echo "STOP: the kit's mruby is not $MR"; return 1; }
  [ -e "$A" ] && { echo "STOP: $A exists"; return 1; }
  mkdir -p $A/vm
  for k in "${!SRC[@]}"; do mv $T/stage/$k $A/$k; done
  cp $T/stage.SHA256SUMS $A/stage-boot2.SHA256SUMS
  [ -z "$(find $T/stage -type f)" ] || { echo "STOP: unexpected files left in the stage: $(find $T/stage -type f)"; return 1; }
  for k in "${!SRC[@]}"; do cp "${SRC[$k]}" $T/stage/$k; done
  chmod 0755 $T/stage/vmrun.sh $T/stage/vm/capstone-vexec $T/stage/vm/capstone-job $T/stage/vm/vm-contract.sh $T/stage/vm/vm-suite-inner.sh
  ( cd $T/stage && find . -type f | sed 's|^\./||' | sort | xargs sha256sum ) > $T/stage.SHA256SUMS
  echo "STAGED from $O; boot 2's stage in $A"
  cat $T/stage.SHA256SUMS | cut -c1-16,65-
  echo "changed from boot 2:"; diff <(cut -c1-64,65- $A/stage-boot2.SHA256SUMS | sort -k2) <(sort -k2 $T/stage.SHA256SUMS) | grep '^[<>]'
}
main "$@"; exit $?
