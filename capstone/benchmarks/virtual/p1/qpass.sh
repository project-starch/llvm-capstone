#!/usr/bin/env bash
# qpass.sh: QEMU functional pass of the seven P1 kits' native stages (ns/<kit>-ns), slots 29-31, records 524288.
main() {
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local F=$H/perf/faultesc P=$H/perf/p1suite I=$H/perf/natio/images
  it() { for k in "$@"; do printf ' p1q-%s:%s/ns/%s-ns:524288' $k $P $k; done; }
  (bash $F/run.sh 29 $I $(it p1a1 p1k1 p1k2); echo QP-29-DONE) &
  (bash $F/run.sh 30 $I $(it p1a2 p1a4); echo QP-30-DONE) &
  (bash $F/run.sh 31 $I $(it p1a3 p1k3); echo QP-31-DONE) &
  wait; echo QP-ALL-DONE
}
main "$@"; exit
