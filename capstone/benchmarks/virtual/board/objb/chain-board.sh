#!/bin/bash
# chain-board.sh <board 1|2> <list> <out root>: spec-board.sh over a list of boots on one board, in order.
# A list line is `<stage dir> <step s> <booking min> <twin result dir | ->`; # starts a comment.
# Before each boot the chain waits for the board (its busy check, every 120 s, up to CB_WAIT_S, default 8 h): another
# lane's boot is a wait, never a stop. A busy or 409 refusal from spec-board.sh is waited on the same way. Any other
# SB STOP, or a boot whose known-good control failed (VOID), ends the chain: the next boot would read nothing.
# A benchmark that fails is a result, and the chain goes on. Marker: <out root>/.chain-b<board>-done (or -stopped).
# CB_BUSY and CB_POLL_S stub the busy check and its period for the dry run; spec-board.sh's own stubs pass through the environment.
main() {
  set -u
  local BRD=${1:?board} LIST=${2:?list} ROOT=${3:?out root} HERE line stage step book twin name out waited rc log
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local BUSY
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "CB STOP: board $BRD"; return 2;;
  esac
  BUSY=${CB_BUSY:-$BUSY}
  log=$ROOT/chain-b$BRD.log
  mkdir -p "$ROOT"
  while read -r stage step book twin; do
    case $stage in ''|'#'*) continue;; esac
    name=$(basename "$stage"); out=$ROOT/$name
    [ -e "$out" ] && { echo "$(date +%T) $name exists, skipped" >> "$log"; continue; }
    waited=0
    while :; do
      until $BUSY > /dev/null 2>&1; do
        [ $waited -ge "${CB_WAIT_S:-28800}" ] && { echo "$(date +%T) CB STOP: board $BRD busy for $waited s" >> "$log"; touch "$ROOT/.chain-b$BRD-stopped"; return 1; }
        sleep "${CB_POLL_S:-120}"; waited=$((waited + ${CB_POLL_S:-120}))
      done
      echo "$(date +%T) BEGIN $name step $step booking $book twin $twin" >> "$log"
      if [ "$twin" = - ]; then
        bash "$HERE/spec-board.sh" "$BRD" "$stage" "$out" "$step" "$book" > "$out.sb" 2>&1; rc=$?
      else
        SB_TWIN=$twin/result.json bash "$HERE/spec-board.sh" "$BRD" "$stage" "$out" "$step" "$book" > "$out.sb" 2>&1; rc=$?
      fi
      if grep -q -E "^SB STOP: (board $BRD is busy|the hub refused the booking)" "$out.sb"; then
        # The busy check can read free while the hub still refuses (a booking without the lock): wait here too, or
        # the chain retries in a tight loop and its wait limit never fires (dry run, 2026-10-28).
        echo "$(date +%T) $name: board taken, waiting" >> "$log"; rm -f "$out.sb"
        [ $waited -ge "${CB_WAIT_S:-28800}" ] && { echo "$(date +%T) CB STOP: board $BRD refused for $waited s" >> "$log"; touch "$ROOT/.chain-b$BRD-stopped"; return 1; }
        sleep "${CB_POLL_S:-120}"; waited=$((waited + ${CB_POLL_S:-120})); continue
      fi
      break
    done
    echo "$(date +%T) END $name rc=$rc" >> "$log"
    grep -E "^(SB |RUN |READ )" "$out.sb" | sed 's/^/    /' >> "$log"
    if grep -q "^SB STOP" "$out.sb" || grep -q "^RUN control FAILED" "$out"*.log "$out.sb" 2>/dev/null; then
      echo "$(date +%T) CB STOP after $name" >> "$log"; touch "$ROOT/.chain-b$BRD-stopped"; return 1
    fi
  done < "$LIST"
  touch "$ROOT/.chain-b$BRD-done"
}
main "$@"; exit $?
