#!/bin/bash
# plain-chain.sh <board> <list> <out root>: the Plain boots through the SPEC lane's chain, pilot FIRST and ALONE, then the
# main kits only if the pilot passes check-plain-boot.py --pilot.
#
# The list's first boot line must be the pilot stage (perf-plain-pp1). The chain runs in two calls of chain-board.sh:
#   1. a list holding the pilot line alone;
#   2. the pilot gate: check-plain-boot.py <root>/perf-plain-pp1 <pp1 kit> --pilot. Exit 0 arms p1..p6
#      (make-plain-stages.py --arm); any other exit -- VOID 1, NO DATA 2, STOP 3 (counters unusable: monitor ILLX, or no
#      PLAINPROBE access=ok), HOLD 4 (the two instruments disagree) -- ends here with .plain-chain-stopped and the
#      main stages left disarmed;
#   3. the whole list (chain-board.sh skips the pilot, whose out dir exists).
# Independently of this script, a disarmed main stage has no plan.json, so spec-board.sh refuses it (SB STOP) and
# chain-board.sh stops: a chain started on the whole list by hand cannot run p1..p6 before the gate.
# Every board call stubs through the environment exactly as chain-board.sh/spec-board.sh document (CB_BUSY, SB_BUSY,
# SB_BOOK, SB_RUN, BOARDVM_STUB_TRANSCRIPT). PC_CHAIN overrides the chain script, PC_STAGES the stage root (default: the pilot line's directory).
main() {
  set -u
  local BRD=${1:?board} LIST=${2:?list} ROOT=${3:?out root}
  local W=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} HERE first pk rc
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local CHAIN=${PC_CHAIN:-$W/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh}
  local PK=${PC_PILOT_KIT:-$W/perf/T0b/kits-pilot/pp1}
  mkdir -p "$ROOT"; local log=$ROOT/plain-chain-b$BRD.log
  first=$(grep -v -E '^[[:space:]]*(#|$)' "$LIST" | head -1)
  case $(basename "${first%% *}") in perf-plain-pp1) ;; *) echo "PC STOP: the list's first boot is not perf-plain-pp1 ($first)" | tee -a "$log"; return 2;; esac
  grep -v -E '^[[:space:]]*(#|$)' "$LIST" | head -1 > "$ROOT/pilot.list"
  local STAGES=${PC_STAGES:-$(dirname "${first%% *}")}   # the stages are armed where the list points
  echo "$(date +%T) PC pilot: $first" >> "$log"
  bash "$CHAIN" "$BRD" "$ROOT/pilot.list" "$ROOT"; rc=$?
  echo "$(date +%T) PC pilot chain rc=$rc" >> "$log"
  python3 "$HERE/check-plain-boot.py" "$ROOT/perf-plain-pp1" "$PK" --pilot > "$ROOT/pilot-gate.txt" 2>&1; rc=$?
  sed 's/^/    /' "$ROOT/pilot-gate.txt" >> "$log"
  if [ $rc != 0 ]; then
    echo "$(date +%T) PC STOP: pilot gate exit $rc (1 VOID, 2 NO DATA, 3 STOP counters, 4 HOLD instruments); p1..p6 not run, left disarmed" >> "$log"
    touch "$ROOT/.plain-chain-stopped"; return 1
  fi
  python3 "$HERE/make-plain-stages.py" --arm --out "$STAGES" >> "$log" 2>&1 || { echo "PC STOP: arming failed" >> "$log"; touch "$ROOT/.plain-chain-stopped"; return 1; }
  echo "$(date +%T) PC pilot PASS: main kits armed; chain over the whole list" >> "$log"
  bash "$CHAIN" "$BRD" "$LIST" "$ROOT"; rc=$?
  echo "$(date +%T) PC main chain rc=$rc" >> "$log"
  [ $rc = 0 ] && touch "$ROOT/.plain-chain-done" || touch "$ROOT/.plain-chain-stopped"
  return $rc
}
main "$@"; exit $?
