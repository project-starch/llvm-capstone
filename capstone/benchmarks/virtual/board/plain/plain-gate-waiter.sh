#!/bin/bash
# plain-gate-waiter.sh <chain pid>: the Plain pilot gate for board 1's LIVE chain (b1-cont.list), in place of
# plain-chain.sh, which would start a second chain on the same board.
# The live list carries perf-plain-pp1 early and p1..p6 later (disarmed: no plan.json, so spec-board.sh refuses them and
# chain-board.sh stops). This waits for a NEW "END perf-plain-pp1" line in chain-b1.log, runs
# check-plain-boot.py --pilot on the chain's out dir, and arms p1..p6 (make-plain-stages.py --arm) only on exit 0.
# Any other gate exit leaves p1..p6 disarmed, so the chain stops at p1 with SB STOP / CB STOP and chain-xz.sh does not
# start: the board waits for a look. Ends on: the gate ran; the chain pid gone with no pp1 END; or PG_WAITMAX s.
# Stubs: PG_LOG (chain log), PG_ROOT (chain out dir), PG_STAGES, PG_PILOT_KIT, PG_GATE / PG_ARM (commands).
main() {
  set -u
  local CPID=${1:?chain pid} W=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} HERE n0 t0 rc
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local LOG=${PG_LOG:-$W/spec/board/out/chain-b1.log} ROOT=${PG_ROOT:-$W/spec/board/out}
  local STAGES=${PG_STAGES:-$W/perf/T0b/board/stages} PK=${PG_PILOT_KIT:-$W/perf/T0b/kits-pilot/pp1}
  local GATE=${PG_GATE:-python3 $HERE/check-plain-boot.py} ARM=${PG_ARM:-python3 $HERE/make-plain-stages.py}
  local O=$ROOT/plain-gate-b1.log
  n0=$(wc -l < "$LOG"); t0=$(date +%s)
  echo "$(date '+%F %T') PG waiting for END perf-plain-pp1 after line $n0 of $LOG (chain pid $CPID)" >> "$O"
  until tail -n +$((n0 + 1)) "$LOG" | grep -q " END perf-plain-pp1 rc="; do
    kill -0 "$CPID" 2>/dev/null || { echo "$(date '+%F %T') PG STOP: chain pid $CPID gone with no pp1 END; p1..p6 left disarmed" >> "$O"; return 2; }
    [ $(( $(date +%s) - t0 )) -gt ${PG_WAITMAX:-172800} ] && { echo "$(date '+%F %T') PG STOP: no pp1 END in ${PG_WAITMAX:-172800} s" >> "$O"; return 2; }
    sleep ${PG_POLL:-60}
  done
  sleep ${PG_SETTLE:-20}   # the READ lines follow the END line
  $GATE "$ROOT/perf-plain-pp1" "$PK" --pilot > "$ROOT/pilot-gate.txt" 2>&1; rc=$?
  sed 's/^/    /' "$ROOT/pilot-gate.txt" >> "$O"
  if [ $rc != 0 ]; then
    echo "$(date '+%F %T') PG STOP: pilot gate exit $rc (1 VOID, 2 NO DATA, 3 STOP counters, 4 HOLD instruments); p1..p6 left disarmed" >> "$O"
    return 1
  fi
  $ARM --arm --out "$STAGES" >> "$O" 2>&1 || { echo "$(date '+%F %T') PG STOP: arming failed" >> "$O"; return 1; }
  echo "$(date '+%F %T') PG pilot PASS: p1..p6 armed" >> "$O"
}
main "$@"; exit $?
