#!/bin/bash
# after-train.sh <board> <pid>: board <board>'s queue after its current SPEC chain (pid <pid>) ends, in the lead's paper-set
# order (2026-11-01): 1. the ObjLife stages left in either board's queue (objlife-rest.list, out-hfl: existing out dirs
# are skipped, so both boards share it); 2. the Plain stages whose QEMU twin run passed (plain-gate.py over
# census-plain, waiting up to 12 h for that chain), then the deferred MapLife stages (maplife-deferred.list), out-pm.
# Each part needs the previous one's done marker. Log out-pm/after-b<board>.log. AT_* stub every external step.
main() {
  set -u
  local BRD=${1:?board} PID=${2:?pid} H=/mnt/hdd/$USER/capstone S O P L t0 r
  S=$H/spec/board; O=${AT_OUT:-$S/out-hfl}; P=${AT_PM:-$S/out-pm}; mkdir -p "$P"; L=$P/after-b$BRD.log
  local CB=${AT_CHAIN:-bash $H/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh}
  local CEN=${AT_CENSUS:-$H/spec/census-plain} GATE="python3 $S/plain-gate.py"
  echo "AT START $(date '+%F %T') board $BRD waiting for pid $PID" >> "$L"
  t0=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${AT_WAITMAX:-345600} ] && { echo "AT STOP: pid $PID still running" >> "$L"; return 1; }
    sleep ${AT_SLEEP:-60}
  done
  [ -e "$O/.chain-b$BRD-done" ] || { echo "AT STOP: board $BRD's chain ended without .chain-b$BRD-done; nothing launched" >> "$L"; return 1; }
  rm -f "$O/.chain-b$BRD-done" "$O/.chain-b$BRD-stopped"
  echo "AT OBJLIFE BEGIN $(date +%T)" >> "$L"
  $CB "$BRD" "$S/objlife-rest.list" "$O" >> "$L" 2>&1; r=$?
  echo "AT OBJLIFE END rc=$r $(date +%T)" >> "$L"
  [ -e "$O/.chain-b$BRD-done" ] || { echo "AT STOP: the ObjLife part did not finish" >> "$L"; return 1; }
  t0=$(date +%s)
  until [ -e "$CEN/.chain-done" ]; do
    [ $(( $(date +%s) - t0 )) -gt ${AT_CENMAX:-43200} ] && { echo "AT NOTE: the Plain twin chain is not done; gating on what has finished" >> "$L"; break; }
    sleep ${AT_SLEEP:-60}
  done
  { echo "# Plain stages that passed the QEMU twin, $(date '+%F %T'), then the deferred MapLife stages"
    $GATE "$CEN" "$H/spec/stages-train-plain" "$S/plain-candidates.txt" 2>> "$L"
    grep -v '^#' "$S/maplife-deferred.list"; } > "$P/after-b$BRD.list"
  echo "AT PLAIN+MAPLIFE BEGIN $(date +%T): $(grep -c plain-rv64 "$P/after-b$BRD.list") Plain stages" >> "$L"
  $CB "$BRD" "$P/after-b$BRD.list" "$P" >> "$L" 2>&1; r=$?
  echo "AT DONE rc=$r $(date +%T)" >> "$L"
}
main "$@"; exit $?
