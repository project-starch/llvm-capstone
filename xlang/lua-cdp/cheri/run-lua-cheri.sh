#!/usr/bin/env bash
# End-to-end CHERI baseline for the Lua-CDP corpus:
#   1. build — compile the rows purecap against the CheriBSD sysroot
#   2. image — stage the binaries into the CheriBSD image (fast re-stage)
#   3. run   — boot CHERI-QEMU once, run spatial / temporal / eager
#   4. class — MISS / BLOCKED-SYNC / BLOCKED-SWEEP table
#
# Baseline measurement only: this is the CHERI column, not our system. It reuses
# the corpus-agnostic drivers in capstone/tests/cheri-baseline/ (provision, boot,
# classify) — the same ones the mruby CHERI column uses.
#
#   ./run-lua-cheri.sh                 # all 13 rows
#   ONLY_ROWS="1" ./run-lua-cheri.sh   # just key 1 (lua-openssl) — one boot, fast
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$HERE/../../../capstone/tests/cheri-baseline"

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
RUNDIR=${RUNDIR:-$CHERI_ROOT/xlang-run}
STAGE=${STAGE:-$CHERI_ROOT/lua-cdp-cheri}   # basename must match run-in-guest.sh DIR
GUEST_DIR=/root/lua-cdp-cheri
mkdir -p "$RUNDIR" "$STAGE"

# Optional ONLY_ROWS filter: BOTH the guest (iterates rows.tsv) and classify.py
# must see the same filtered list, or classify reports the others as missing.
ROWS_FILE="$HERE/rows.tsv"
if [ -n "${ONLY_ROWS:-}" ]; then
  ROWS_FILE="$RUNDIR/lua-rows-filtered.tsv"
  awk -v want=" ${ONLY_ROWS} " '
    /^#/ { print; next }
    { split($0, f, "\t"); if (index(want, " " f[1] " ")) print }
  ' "$HERE/rows.tsv" > "$ROWS_FILE"
  n=$(grep -vcE '^#|^$' "$ROWS_FILE")
  [ "$n" -gt 0 ] || { echo "ONLY_ROWS='$ONLY_ROWS' matched no rows" >&2; exit 2; }
  echo "== ONLY_ROWS='$ONLY_ROWS' -> $n row(s); NOT a full reproduction =="
fi

echo "== [1/4] compile the corpus purecap =="
OUT="$STAGE" ROWS_FILE="$ROWS_FILE" bash "$HERE/compile-lua-purecap.sh" \
  || { echo "compile failed"; exit 1; }
cp "$ROWS_FILE" "$STAGE/rows.tsv"

echo "== [2/4] stage the corpus into the image =="
OVERLAY_SRC="$STAGE" bash "$BASELINE/provision-cheri-vehicle.sh" \
  || { echo "vehicle provisioning failed"; exit 1; }

echo "== [3/4] boot + run all three configs =="
python3 "$BASELINE/cheri-run.py" "$RUNDIR/qemu-argv.txt" "$RUNDIR/serial.log" "$GUEST_DIR"
echo "cheri-run rc=$?; serial log: $RUNDIR/serial.log"

echo "== [4/4] classify =="
python3 "$BASELINE/classify.py" "$RUNDIR/serial.log" "$ROWS_FILE" | tee "$RUNDIR/lua-table.txt"
echo
echo "table: $RUNDIR/lua-table.txt"
