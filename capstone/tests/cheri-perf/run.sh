#!/usr/bin/env bash
# End-to-end CHERI-side revoke-cost run -- the CHERI arm of the CHERI-vs-Capstone
# temporal-safety perf comparison (plans/perf-cheri-vs-capstone-qemu.md, PI
# 2026-07-14). Mirrors the Capstone arm (tests/runtime-qemu/run-revoke-cost-probe.sh).
#
#   1. compile the microbench purecap into the rootfs overlay
#   2. bake the overlay into the CheriBSD purecap disk image (cheribuild)
#   3. boot CHERI-QEMU (optionally under -icount for a deterministic count) and
#      run the malloc/touch/free loop under spatial / temporal / eager revocation
#   4. parse the RCPERF lines -> per-op instruction count + overhead vs baseline
#
# Env: ICOUNT=1 injects `-icount shift=0,sleep=off` (deterministic but slower
# boot). RC_ITERS / RC_BLOCK / RC_TRIALS tune the workload (passed to the guest).
#
# CHERI uses its OWN disk image (cheribsd-riscv64-purecap.img), NOT the shared
# capstone rootfs.ext2 -- so no rootfs-lock contention with the Capstone arm.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERIBUILD=${CHERIBUILD:-/home/alexey/cheri-ws/cheribuild/cheribuild.py}
OVERLAY=${OVERLAY:-/home/alexey/cheri-ws/rootfs-overlay}
OUT=${OUT:-$OVERLAY/root/cheri-perf}
WORK=${WORK:-/home/alexey/cheri-ws/perf-run}
ICOUNT=${ICOUNT:-0}
mkdir -p "$WORK" "$OUT"

LOCAL=/home/alexey/cheri-ws/local-deps
export CPATH="$LOCAL/usr/include:${CPATH:-}"
export LIBRARY_PATH="$LOCAL/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$LOCAL/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

# cheribuild's disk-image target always runs a pre-build existence check for
# `makeinfo` (a dependency of the ALREADY-BUILT gdb-riscv64-purecap). If texinfo
# is not installed the bake aborts even though gdb is not rebuilt. A dummy shim
# satisfies the check so gdb is skipped; no sudo/apt needed. (apt install texinfo
# is the alternative.)
if ! command -v makeinfo >/dev/null 2>&1; then
  mkdir -p "$WORK/bin"
  cat > "$WORK/bin/makeinfo" <<'SHIM'
#!/bin/sh
case "$1" in --version) echo "makeinfo (GNU texinfo) 6.8"; exit 0 ;; esac
exit 0
SHIM
  chmod +x "$WORK/bin/makeinfo"
  export PATH="$WORK/bin:$PATH"
  echo "[*] using dummy makeinfo shim ($WORK/bin) to skip gdb rebuild"
fi

echo "== [1/4] compile purecap microbench =="
OUT="$OUT" bash "$SCRIPT_DIR/compile.sh" || { echo "compile failed"; exit 1; }

echo "== [2/4] bake overlay into disk image =="
python3 "$CHERIBUILD" disk-image-riscv64-purecap --skip-update \
  --disk-image/extra-files "$OVERLAY" \
  >"$WORK/disk-image.log" 2>&1 || { echo "disk image build failed (see $WORK/disk-image.log)"; tail -20 "$WORK/disk-image.log"; exit 1; }

echo "== [3/4] extract qemu run command =="
python3 "$CHERIBUILD" run-riscv64-purecap --pretend --skip-update 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' > "$WORK/pretend.txt"
ICOUNT="$ICOUNT" python3 - "$WORK/pretend.txt" "$WORK/qemu-argv.txt" <<'PY'
import sys, shlex, os
pretend, outp = sys.argv[1], sys.argv[2]
line = None
for ln in open(pretend):
    if "qemu-system-riscv64xcheri" in ln and "-M" in ln:
        line = ln.strip()
if not line:
    sys.exit("could not find qemu command in pretend output")
if "&&" in line:
    line = line.split("&&", 1)[1].strip()
argv = shlex.split(line)
for i, t in enumerate(argv):
    if t.endswith("qemu-system-riscv64xcheri"):
        argv = argv[i:]; break
if os.environ.get("ICOUNT", "0") not in ("0", "", "no"):
    argv += ["-icount", "shift=0,sleep=off"]
with open(outp, "w") as f:
    f.write("\n".join(argv) + "\n")
print(f"qemu argv tokens: {len(argv)} (icount={os.environ.get('ICOUNT','0')})")
PY

echo "== [4/4] boot + run all three configs =="
python3 "$SCRIPT_DIR/perf-run.py" "$WORK/qemu-argv.txt" "$WORK/serial.log"
rc=$?
echo "perf-run rc=$rc; serial log: $WORK/serial.log"

echo
echo "=== CHERI revoke-cost: temporal-safety overhead ==="
python3 "$SCRIPT_DIR/parse.py" "$WORK/serial.log"
