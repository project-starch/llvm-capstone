#!/bin/bash
# SB_RUN for the dry run: the REAL run-stage.py, with BOARDVM_STUB_TRANSCRIPT set to the stage's own stub, so boardvm
# never calls the board recipe. The pilot takes $OB_PILOT_VARIANT's stub when set. Refuses (rc 97) when no stub exists.
st=; out=; prev=; for x in "$@"; do [ "$prev" = --stage ] && st=$x; [ "$prev" = --out ] && out=$x; prev=$x; done
k=$(basename "$st"); k=${k#perf-objb-}
v=; [ "$k" = objbh1 ] && [ -n "${OB_PILOT_VARIANT:-}" ] && v=-$OB_PILOT_VARIANT
stub=$OB_STUB_DIR/$k$v.board.txt
[ -n "$st" ] && [ -f "$stub" ] || { echo "SBRUN-STUB REFUSED: no stub for stage '$st'"; exit 97; }
echo "SBRUN-STUB $k$v argv: $*" >> "$(dirname "$out")/../sbrun-argv.log"
export BOARDVM_STUB_TRANSCRIPT=$stub
exec python3 ${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/wt-spec/capstone/ports/spec-cpu2017/board/run-stage.py "$@"
