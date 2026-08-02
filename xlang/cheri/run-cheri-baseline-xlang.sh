#!/usr/bin/env bash
# End-to-end CHERI baseline measurement for the xlang corpus:
#   0. gate  — every shim must reproduce its row's defect natively
#   1. build — compile the 11 rows purecap against the CheriBSD sysroot
#   2. image — provision the vehicle and bake the shims into it
#   3. run   — boot CHERI-QEMU, run spatial / temporal / eager
#   4. class — BLOCKED-SYNC / BLOCKED-SWEEP / MISS table
#
# Baseline measurement only: this is the CHERI column, not our system.
#
# Provisioning lives in capstone/tests/cheri-baseline/provision-cheri-vehicle.sh (build
# CHERI-QEMU + bbl firmware, download CheriBSD purecap distribution sets,
# assemble the image). This script is the measurement half and calls it.
#
#   ./run-cheri-baseline-xlang.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$SCRIPT_DIR/../../capstone/tests/cheri-baseline"   # reused, corpus-agnostic drivers

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
# The extracted distribution rootfs doubles as the purecap SYSROOT: it carries
# /usr/include and /usr/lib, so no separately built sysroot is needed.
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
RUNDIR=${RUNDIR:-$CHERI_ROOT/xlang-run}
STAGE=${STAGE:-$CHERI_ROOT/cheri-baseline-xlang}
GUEST_DIR=/root/cheri-baseline-xlang
mkdir -p "$RUNDIR" "$STAGE"

echo "== [0/4] native validation gate =="
# A shim that does not reproduce its row's defect natively measures nothing,
# so it must not cost a boot.
python3 "$SCRIPT_DIR/check_shim_fidelity.py" \
  || { echo "native gate FAILED — fix the shims before booting"; exit 1; }

echo "== [1/4] compile the corpus purecap =="
[ -d "$ROOTFS/usr/include" ] || {
  echo "no purecap sysroot at $ROOTFS — run provision-cheri-vehicle.sh first"; exit 1; }
CHERI_SDK="$SDK" SYSROOT="$ROOTFS" CC="$SDK/bin/clang" OUT="$STAGE" \
  bash "$SCRIPT_DIR/compile-purecap.sh" || { echo "compile failed"; exit 1; }
# Everything the guest script reads must travel with the binaries.
# ONLY_ROWS="7 12" runs just those rows. The guest iterates rows.tsv and
# classify.py parses against it, so BOTH must see the same filtered list --
# filtering only the staged copy would make classify report every other row as
# missing. One boot costs ~10 min and each shim run ~20s x 3 configs, so this
# turns a ~70 min full pass into a few minutes when re-measuring one row.
# It does NOT re-verify the others: a filtered run is not a reproduction.
ROWS_FILE="$SCRIPT_DIR/rows.tsv"
if [ -n "${ONLY_ROWS:-}" ]; then
  ROWS_FILE="$RUNDIR/rows-filtered.tsv"
  awk -v want=" ${ONLY_ROWS} " '
    /^#/ { print; next }
    { split($0, f, "\t"); if (index(want, " " f[1] " ")) print }
  ' "$SCRIPT_DIR/rows.tsv" > "$ROWS_FILE"
  n=$(grep -vcE '^#|^$' "$ROWS_FILE")
  [ "$n" -gt 0 ] || { echo "ONLY_ROWS='$ONLY_ROWS' matched no rows" >&2; exit 2; }
  echo "== ONLY_ROWS='$ONLY_ROWS' -> $n row(s); this is NOT a full reproduction =="
fi
cp "$ROWS_FILE" "$STAGE/rows.tsv"
cp "$SCRIPT_DIR/run-in-guest.sh" "$STAGE/"
# cheri_status reports the revocation policy each config ACTUALLY applied, so
# config reality is recorded rather than assumed.
"$SDK/bin/clang" --target=riscv64-unknown-freebsd -march=rv64gcxcheri \
  -mabi=l64pc128d --sysroot="$ROOTFS" -mno-relax \
  "$BASELINE/cheri_status.c" -o "$STAGE/cheri_status" 2>/dev/null \
  || echo "  [warn] cheri_status did not build; config reality will be unrecorded"
rm -f "$STAGE"/*.build.log

echo "== [2/4] provision the vehicle + bake in the corpus =="
OVERLAY_SRC="$STAGE" bash "$BASELINE/provision-cheri-vehicle.sh" \
  || { echo "vehicle provisioning failed"; exit 1; }
# provision-cheri-vehicle.sh stages OVERLAY_SRC as /root/<basename>, so name
# the staged directory to match GUEST_DIR.

echo "== [3/4] boot + run all three configs =="
python3 "$BASELINE/cheri-run.py" "$RUNDIR/qemu-argv.txt" "$RUNDIR/serial.log" "$GUEST_DIR"
echo "cheri-run rc=$?; serial log: $RUNDIR/serial.log"

echo "== [4/4] classify =="
python3 "$BASELINE/classify.py" "$RUNDIR/serial.log" "$ROWS_FILE" \
  | tee "$RUNDIR/table.txt"
echo
echo "table: $RUNDIR/table.txt"
echo "Check BEFORE trusting any row: sanity_mock must be rc=0 (else the harness"
echo "itself is faulting), and REVOKE_ENABLED must read 0/1/1 across the three"
echo "passes (else a config did not take)."
