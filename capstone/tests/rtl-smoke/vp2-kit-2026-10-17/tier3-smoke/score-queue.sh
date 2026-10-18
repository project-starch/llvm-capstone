#!/bin/bash
# score-queue.sh <seq out> <tag>...: wait for each tag's END line in <seq out> (T3SEQ, T3PG or T3APPVM) and score it with
# score-arm.sh against <mnt-hdd>/capstone/t3-<tag>/bundle (or t3-pred-<tag>/out for the board lane's own predictions).
# Ends when every tag is scored, when the sequence prints DONE or STOP with tags left unstarted, or after 4 hours.
main() {
  set -u
  local Q=${1:?seq out}; shift; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone end=$(( $(date +%s) + 14400 )) t P
  for t in "$@"; do
    until grep -a -q "T3SEQ $t END\|T3PG $t END\|T3APPVM $t END" "$Q" 2>/dev/null; do
      [ "$(date +%s)" -ge $end ] && { echo "SCOREQ timeout before $t"; return 1; }
      if grep -a -q "T3SEQ DONE\|T3SEQ STOP\|T3PG DONE\|T3PG STOP\|QUEUE DONE\|^STOP" "$Q" 2>/dev/null \
         && ! grep -a -q "T3SEQ $t BEGIN\|T3PG $t BEGIN\|T3APPVM $t BEGIN" "$Q"; then echo "SCOREQ $t never started"; return 1; fi
      sleep 20
    done
    P=$H/t3-$t/bundle; [ -d "$P" ] || P=$H/t3-pred-$t/out
    bash "$S/score-arm.sh" "$t" "$P" "$H/t3-run/$t" "$Q"
  done
  echo "SCOREQ DONE $(date +%T)"
}
main "$@"; exit $?
