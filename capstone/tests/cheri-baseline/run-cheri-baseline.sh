#!/usr/bin/env bash
# End-to-end CHERI baseline run (agentB-015):
#   1. compile the 15 in-scope corpus rows purecap into a rootfs overlay
#   2. bake the overlay into the CheriBSD purecap disk image
#   3. boot CHERI-QEMU, run each row under config A (spatial only) and B (+sweep)
#   4. classify BLOCKED-SYNC / BLOCKED-SWEEP / MISS and print the table
#
# Baseline measurement only: this is the CHERI column, not our system.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERIBUILD=${CHERIBUILD:-/home/alexey/cheri-ws/cheribuild/cheribuild.py}
OVERLAY=${OVERLAY:-/home/alexey/cheri-ws/rootfs-overlay}
OUT=${OUT:-$OVERLAY/root/cheri-baseline}
WORK=${WORK:-/home/alexey/cheri-ws/baseline-run}
mkdir -p "$WORK" "$OUT"

LOCAL=/home/alexey/cheri-ws/local-deps
export CPATH="$LOCAL/usr/include:${CPATH:-}"
export LIBRARY_PATH="$LOCAL/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$LOCAL/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "== [1/4] compile purecap corpus =="
OUT="$OUT" bash "$SCRIPT_DIR/compile-purecap.sh" || { echo "compile failed"; exit 1; }

echo "== [2/4] bake overlay into disk image =="
python3 "$CHERIBUILD" disk-image-riscv64-purecap --skip-update \
  --disk-image/extra-files "$OVERLAY" \
  >"$WORK/disk-image.log" 2>&1 || { echo "disk image build failed (see $WORK/disk-image.log)"; tail -20 "$WORK/disk-image.log"; exit 1; }

echo "== [3/4] extract qemu run command =="
python3 "$CHERIBUILD" run-riscv64-purecap --pretend --skip-update 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' > "$WORK/pretend.txt"
python3 - "$WORK/pretend.txt" "$WORK/qemu-argv.txt" <<'PY'
import sys, shlex
pretend, outp = sys.argv[1], sys.argv[2]
line = None
for ln in open(pretend):
    if "qemu-system-riscv64xcheri" in ln and "-M" in ln:
        line = ln.strip()
if not line:
    sys.exit("could not find qemu command in pretend output")
# drop any 'cd DIR &&' prefix
if "&&" in line:
    line = line.split("&&", 1)[1].strip()
argv = shlex.split(line)
# start at the qemu binary token
for i, t in enumerate(argv):
    if t.endswith("qemu-system-riscv64xcheri"):
        argv = argv[i:]; break
with open(outp, "w") as f:
    f.write("\n".join(argv) + "\n")
print("qemu argv tokens:", len(argv))
PY

echo "== [4/4] boot + run both configs =="
python3 "$SCRIPT_DIR/cheri-run.py" "$WORK/qemu-argv.txt" "$WORK/serial.log"
rc=$?
echo "cheri-run rc=$rc; serial log: $WORK/serial.log"

echo "== classify =="
python3 "$SCRIPT_DIR/classify.py" "$WORK/serial.log" "$SCRIPT_DIR/rows.tsv"
