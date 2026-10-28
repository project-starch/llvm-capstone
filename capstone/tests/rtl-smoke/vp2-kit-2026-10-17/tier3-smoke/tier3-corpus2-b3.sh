#!/bin/bash
# The bug corpus on BITSTREAM 3, second wave (2026-10-28; pre-registered in README.md before any boot): the arms bitstream 2
# read after its re-read, on the bitstream-3 kits, against the same bitstream-2 QEMU references. Board 1: mruby stock, mruby
# Sublet, perl stock and Sublet, then Juliet temporal. Board 2: the new arms (xlang, plane x2, capi in six groups), then the
# Juliet spatial sample. Each board starts only when its 23-arm re-read (tier3-reread-b3.sh, log given as $2 / $3) has
# printed "RR seq-<board> DONE"; a re-read that STOPs or HOLDs, or has not ended within 6 h, holds that board. Within a
# board, each runner starts only when the previous one returned 0, as on bitstream 2; a runner's STOP holds the board.
# Usage: tier3-corpus2-b3.sh <log> <re-read log board 1> <re-read log board 2>. C2_RUN stubs every runner for the dry run
# (it is called as `$C2_RUN <runner> <board> <log> [tags]`); C2_POLL / C2_WAITMAX stub the wait.
main() {
  set -u
  local O=${1:?log} R1=${2:?re-read log 1} R2=${3:?re-read log 2} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  waitrr() {   # waitrr <board> <re-read log>: 0 once "RR seq-<board> DONE" is in it
    local b=$1 L=$2 t0; t0=$(date +%s)
    echo "C2 board $b WAIT for re-read DONE $(date +%T)" >> "$O"
    until grep -a -q "^RR seq-$b DONE" "$L" 2>/dev/null; do
      grep -a -q "^RR \(STOP\|HOLD\)\|^RR .* STOP" "$L" 2>/dev/null && {
        echo "C2 board $b HOLD: the re-read stopped :: $(grep -a "STOP\|HOLD" "$L" | tail -1 | cut -c1-160)" >> "$O"; return 1; }
      [ $(( $(date +%s) - t0 )) -gt ${C2_WAITMAX:-21600} ] && {
        echo "C2 board $b HOLD: the re-read did not end within ${C2_WAITMAX:-21600} s" >> "$O"; return 1; }
      sleep ${C2_POLL:-60}
    done
    echo "C2 board $b FREE: $(grep -a "^RR seq-$b DONE" "$L" | cut -c1-80)" >> "$O"
  }
  step() {     # step <board> <runner> [tags]: one runner, its log appended to $O
    local b=$1 r=$2 rc L; shift 2
    L=/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass/c2b3-${r%.sh}-b$b.out; : > $L
    echo "C2 $r on board $b BEGIN $(date +%T) $*" >> "$O"
    case $r in
      tier3-newarms-b3.sh) ${C2_RUN:-bash} $S/$r $b - x $L; rc=$?;;
      *) ${C2_RUN:-bash} $S/$r $b $L "$@"; rc=$?;;
    esac
    echo "C2 $r on board $b END rc=$rc $(date +%T)" >> "$O"; sed 's/^/  /' $L >> "$O"
    [ $rc = 0 ] || echo "C2 board $b HOLD after $r (rc=$rc): nothing further runs on this board" >> "$O"
    return $rc
  }
  { waitrr 2 "$R2" && step 2 tier3-newarms-b3.sh && step 2 tier3-juliet-spatial-b3.sh; } & local p2=$!
  waitrr 1 "$R1" && step 1 tier3-mp-stock-b3.sh && step 1 tier3-mp-sublet-b3.sh \
    && step 1 tier3-mp-perl-b3.sh perl-stock perl-sublet && step 1 tier3-juliet-b3.sh
  wait $p2
  echo "C2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
