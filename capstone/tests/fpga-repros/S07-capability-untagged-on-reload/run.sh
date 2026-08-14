#!/usr/bin/env bash
# Rebuild and stage the four exclusion rungs, and say what each must return.
#
# These are NEGATIVE results: each asks whether one simple property of a capability round trip is
# broken, and each says no. They are here so the hardware side can see what has already been
# excluded, and re-run it after any RTL change rather than taking this README's word for it.
#
# EVERY RUNG HAS A POSITIVE CONTROL, and it is not optional: a rung returning 0xFFFF is
# indistinguishable from a rung whose check cannot fail. Build with the matching *_SELFTEST define
# and require the failing value before believing a clean one.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
LADDER="$ROOT/capstone/tests/runtime-qemu/silicon-ladder"

cat <<'TABLE'
rung       asks                                                  clean   selftest must give
---------  ----------------------------------------------------  ------  ------------------
s06spill   does a spilled capability come back TAGGED?            65535   (see its header)
s06bnds    ... with its BOUNDS intact?                            65535   0
s06wr      ... surviving byte stores written THROUGH it?          65535   0
s06pld     ... surviving a scalar load of its own granule?        65535   0
TABLE

for r in s06spill s06bnds s06wr s06pld; do
  echo "== $r"
  bash "$LADDER/../silicon-ladder/run-ladder-qemu.sh" "$r" 2>&1 | tail -2 || true
done

echo
echo "To run them on the board, stage each and rebuild the initramfs (linux-rebuild BEFORE"
echo "opensbi-rebuild), then:  BAKED_RUNGS=\"s06spill s06bnds s06wr s06pld\" \\"
echo "  python3 -m fpga_driver.run_baked_rungs_fpga"
