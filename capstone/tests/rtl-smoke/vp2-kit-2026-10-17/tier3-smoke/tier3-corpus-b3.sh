#!/bin/bash
# The bug corpus on BITSTREAM 3 (caplifive_vp2c_e7af48457), first arms (2026-10-27; pre-registered in README.md before any
# boot): the arms bitstream 2 read on its first night, re-read on the bitstream that fixes R-62/R-63, against the same
# bitstream-2 QEMU references. Board 1 runs cpython lt1 (tier3-cpy-b3-arm.sh); board 2 runs postgres sql0
# (tier3-pg-b3-arm.sh) and then cpython lt0. Each arm is its own cold boot, so lt0 does not depend on sql0's outcome; it
# runs once sql0's script has returned, whatever its status, as on bitstream 2.
# Usage: tier3-corpus-b3.sh <log>. CO_PG / CO_CPY stub an arm script for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  arm() {   # arm <pg|cpy> <board> <tag>
    local kind=$1 b=$2 t=$3 L rc
    L=/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass/corpus-b3-$t-b$b.out; : > $L
    echo "CO $t on board $b BEGIN $(date +%T)" >> "$O"
    if [ $kind = pg ]; then ${CO_PG:-bash $S/tier3-pg-b3-arm.sh} $b $L $t; else ${CO_CPY:-bash $S/tier3-cpy-b3-arm.sh} $b $L $t; fi
    rc=$?; echo "CO $t on board $b END rc=$rc $(date +%T)" >> "$O"
    sed 's/^/  /' $L >> "$O"; return $rc
  }
  arm cpy 1 lt1 & local p1=$!
  arm pg 2 sql0; arm cpy 2 lt0
  wait $p1
  echo "CO DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
