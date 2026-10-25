#!/bin/bash
# Three firmware builds on one saved Image: A rfence-pair e1db99e (current defines); B the batch monitor, cap OFF
# (current defines); C the batch monitor, cap ON. A vs B: the generated monitor must be byte-identical. C: the gate.
main() {
  local B=/mnt/hdd/$USER/capstone/batch-fw R=/tmp/capstone/r55-table/build-fpga-fw.sh
  local DC; DC=$(cat $B/defines-current.txt); local DB; DB=$(cat $B/defines-batch.txt)
  for v in "A /tmp/capstone/csbi-rfence $DC" "B /tmp/capstone/csbi-batch $DC" "C /tmp/capstone/csbi-batch $DB"; do
    set -- $v; local t=$1 mon=$2; shift 2
    FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=$mon FW_IMAGE_DIR=$B/img taskset -c 16-31 bash $R $B/fw-$t "$*" > $B/fw-$t.log 2>&1
    echo "BUILD $t rc=$? mon $(git -C $mon rev-parse --short=7 HEAD) dirty=$(git -C $mon status --porcelain | wc -l) $(grep '^gate:' $B/fw-$t.log | tail -1)"
  done
  echo BUILD3 DONE
}
main "$@"; exit $?
