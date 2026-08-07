#!/usr/bin/env bash
# CHERI column at REAL-LUA fidelity: build the 13 corpus reproductions as real Lua
# 5.4.7 purecap CheriBSD programs (real userdata/__gc/GC, minimal native-object
# stub -- mirroring the Capstone LUA_CDP_* domains), then run them under the three
# revocation configs via the same cheri-baseline drivers the shim column uses.
#
#   spatial  (rev off)   -> stale access succeeds  -> MISS
#   temporal (async,     -> sweep hasn't run at the -> MISS  (deployed default)
#             default)       contract point
#   eager    (revoke      -> free revokes now, stale -> SWEEP (fault = CAUGHT),
#             every free)     access faults (SIGPROT)    the analog of Capstone revoke
#
# This is the real-Lua counterpart of ./run-lua-cheri.sh (which runs the pure-C
# shims). Same drivers, same rows, same image -- only the workload is upgraded to
# the real interpreter.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$HERE/../../../capstone/tests/cheri-baseline"
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
RUNDIR=${RUNDIR:-$CHERI_ROOT/xlang-run}
STAGE=${STAGE:-$CHERI_ROOT/lua-cdp-cheri}   # basename -> guest /root/lua-cdp-cheri
GUEST_DIR=/root/lua-cdp-cheri
ROWS_FILE="$HERE/rows.tsv"
mkdir -p "$RUNDIR" "$STAGE"

echo "== [1/4] build real-Lua purecap reproduction, copy to the 13 row names =="
OUT="$STAGE" bash "$HERE/real/build-real-lua-cdp.sh" "$HERE/real/cdp_real.c" \
  || { echo "build failed"; exit 1; }
while IFS=$'\t' read -r key shim rest; do
  case "$key" in ''|\#*) continue ;; esac
  cp -f "$STAGE/cdp_real" "$STAGE/$(basename "$shim" .c)"
done < "$ROWS_FILE"
rm -f "$STAGE/cdp_real"
cp -f "$ROWS_FILE" "$STAGE/rows.tsv"
cp -f "$HERE/run-in-guest.sh" "$STAGE/run-in-guest.sh"

echo "== [2/4] stage into the CheriBSD image =="
OVERLAY_SRC="$STAGE" RUNDIR="$RUNDIR" bash "$BASELINE/provision-cheri-vehicle.sh" \
  || { echo "provision failed"; exit 1; }

echo "== [3/4] boot CHERI-QEMU once, run spatial / temporal / eager =="
python3 "$BASELINE/cheri-run.py" "$RUNDIR/qemu-argv.txt" "$RUNDIR/serial.log" "$GUEST_DIR"

echo "== [4/4] classify =="
python3 "$BASELINE/classify.py" "$RUNDIR/serial.log" "$ROWS_FILE" | tee "$RUNDIR/real-lua-table.txt"
echo "table: $RUNDIR/real-lua-table.txt ; serial: $RUNDIR/serial.log"
