#!/bin/bash
# SB_RUN for the Plain chain dry run: the REAL run-stage.py with BOARDVM_STUB_TRANSCRIPT set to the stage's stub, so
# boardvm never calls the board recipe. PLAIN_PILOT_STUB replaces pp1's stub (the pilot-gate negatives). Refuses
# (rc 97) when no stub exists: never a live board call.
st=; prev=; for x in "$@"; do [ "$prev" = --stage ] && st=$x; prev=$x; done
k=$(basename "$st"); k=${k#perf-plain-}
stub=${PLAIN_DRY:?}/../stubs/$k.board.txt
[ "$k" = pp1 ] && [ -n "${PLAIN_PILOT_STUB:-}" ] && stub=$PLAIN_PILOT_STUB
[ -n "$st" ] && [ -f "$stub" ] || { echo "SBRUN-STUB REFUSED: no stub for stage '$st'"; exit 97; }
echo "SBRUN-STUB argv: $* :: stub $(basename "$stub")" >> "$PLAIN_DRY/sbrun-argv.log"
export BOARDVM_STUB_TRANSCRIPT=$stub
exec python3 "${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/wt-spec/capstone/ports/spec-cpu2017/board/run-stage.py" "$@"
