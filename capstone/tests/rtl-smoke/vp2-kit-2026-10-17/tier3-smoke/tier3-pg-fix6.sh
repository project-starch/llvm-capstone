#!/bin/bash
# The postgres Tier 3 arms (pgc, pgm0, pgm1) on the S-27 workaround module (2026-10-19), after the nine-arm re-run:
# tier3-pg-arm.sh unchanged with kit bkit7-fix6t (module d71f207a3af5546b, the batch budget scaled), scored against the
# compiler lane's QEMU records (<mnt-hdd>/capstone/t3-<arm>/bundle; qkit7 wrapper, svadu=false), records <arm>-fix6b,
# bundles under results/2026-10-18-vp2-rdx0/. Runs after the re-run chain ends (FIX6BSEQ DONE, STOP or HOLD).
# Usage: tier3-pg-fix6.sh <re-run out> <out>. Ends by itself. PGF_ARM / PGF_RUN / PGF_SUF / PGF_RESULTS / PGF_DRVPAT
# replace the arm script, the run root, the tag suffix, the results directory and the driver guard for the dry run.
main() {
  set -u
  local W=${1:?re-run out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6t
  local ARM=${PGF_ARM:-$S/tier3-pg-arm.sh} R=${PGF_RUN:-$H/t3-run-fix6pg} SUF=${PGF_SUF:--fix6b}
  local RD=${PGF_RESULTS:-2026-10-18-vp2-rdx0} end=$(( $(date +%s) + 43200 )) t
  until grep -a -q "^FIX6BSEQ DONE\|^FIX6BSEQ STOP\|^FIX6BSEQ HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^FIX6BSEQ" "$W" || { echo "PGF HOLD: the re-run chain did not end" >> "$O"; return 1; }
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] && ( cd $K && sha256sum -c --quiet SHA256SUMS ) \
    || { echo "PGF STOP: the kit is not bkit7-fix6t as recorded" >> "$O"; return 1; }
  for t in pgc pgm0 pgm1; do [ -d "$H/t3-$t/bundle" ] || { echo "PGF STOP: no prediction bundle for $t" >> "$O"; return 1; }; done
  pgrep -f "${PGF_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "PGF STOP: a driver is running" >> "$O"; return 1; }
  echo "PGF BEGIN $(date +%T) (after: $(grep -a '^FIX6BSEQ' "$W" | tail -1))" >> "$O"
  ( cd /tmp/capstone/wt-gate || exit 1; taskset -c 16-31 bash "$ARM" $K "$R" $I/pg pgc pgm0 pgm1 ) >> "$O" 2>&1
  echo "PGF ARMS END rc=$? $(date +%T)" >> "$O"
  for t in pgc pgm0 pgm1; do
    grep -a -q "T3PG $t END" "$O" || { echo "PGF $t not run" >> "$O"; continue; }
    KIT_DESC="kit bkit7-fix6t (module d71f207a3af5546b, the S-27 workaround; batch budget 300 s per step), pin off" \
      RESULTS_DIR=$RD bash "$S/score-arm.sh" "$t$SUF" "$H/t3-$t/bundle" "$R/$t" "$O" >> "$O" 2>&1
  done
  echo "PGF DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
