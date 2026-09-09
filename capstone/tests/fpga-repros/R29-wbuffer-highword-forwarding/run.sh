#!/usr/bin/env bash
# R-29: a plain 8-byte store adjacent to a 128-bit ldc of the same granule loses the high half.
#
#   ./run.sh verify     check the frozen artifacts AND that the rung still contains the construct
#   ./run.sh rung       the 10 KB reproducer on the board
#
# Split out of ../S06-untagged-ldc-stc-high-half/ on 2026-09-10: the aggregate-copy rung is R-29's
# reproducer, not S-06's. S-06 keeps the memcpy half (s06copy), which is fixed and passing.
#
# The rung image is FROZEN and checksummed on purpose. This platform has a per-image entry stall
# (see ../R16-entry-stall/): a rebuilt image is a fresh draw and may simply never run, so the
# shipped .dom is a draw that has been observed to enter. The board mode runs the known-good
# control k800 FIRST -- a boot whose control fails carries no verdict about anything.
#
# The RTL simulation arm is sim/s06agg-shape.S; its six runs across three revisions are recorded in
# sim/s06agg-shape-RECORDS.md. Run it with the rtl-sim procedure, not from here.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/R29-wbuffer-highword-forwarding
MODE=${1:-verify}

verify() {
  echo "== verifying frozen artifacts =="
  ( cd "$D" && sha256sum -c SHA256SUMS ) || { echo "ARTIFACTS DO NOT MATCH -- do not trust any run"; exit 1; }
}

case "$MODE" in
verify)
  verify
  echo
  echo "the board reproducer must still CONTAIN the construct under test:"
  python3 - <<'PY'
import re, subprocess, sys
dom = 'capstone/tests/fpga-repros/R29-wbuffer-highword-forwarding/src/s06agg.dom'
od = 'llvm/cmake-build-debug/bin/llvm-objdump'
# split with maxsplit=1: objdump lines are "addr: bytes \t mnemonic \t operands", so a plain
# split('\t')[-1] yields the OPERANDS and every mnemonic match silently fails.
LDC = re.compile(r'^ldc\s+(\S+),\s*(-?0x[0-9a-f]+|-?\d+)\(')
STC = re.compile(r'^stc\s+(\S+),\s*(-?0x[0-9a-f]+|-?\d+)\(')
out = subprocess.run([od, '-d', '--triple=capstone64-unknown-elf',
                      '--disassemble-symbols=s06agg_compute', dom],
                     capture_output=True, text=True).stdout
ins = [l.split('\t', 1)[-1].strip() for l in out.split('\n') if '\t' in l]

# The AGGREGATE copy is an ldc/stc pair at a NON-ZERO offset -- the plain-data tail of the struct.
# Offset 0 is the pointer chunk and is safe, so a check that accepted any ldc/stc would pass on a
# build whose data copy had been optimised away.
ev = None
for i, l in enumerate(ins):
    m = LDC.match(l)
    if not m or int(m.group(2), 0) == 0:
        continue
    for x in ins[i+1:i+4]:
        n = STC.match(x)
        if n and n.group(1) == m.group(1) and int(n.group(2), 0) != 0:
            ev = f'{l}  ->  {x}'
            break
    if ev:
        break
print('  s06agg_compute   aggregate copy      :', ('YES  ' + ev) if ev else 'NO')
if not ev:
    print('  the struct copy was not emitted capability-grained -- this rung proves nothing')

# R-29's TRIGGER is adjacency: the plain sd of the high half must be the instruction immediately
# before the ldc of that granule. A build that separated them would pass on silicon and the rung
# would read 64 while testing nothing -- which is exactly how the first sim arm read PASS on three
# revisions and had to be retracted. So this is a gate, not a note.
adj = None
for i, l in enumerate(ins[:-1]):
    m = re.match(r'^sd\s+\S+,\s*(-?0x[0-9a-f]+|-?\d+)\((\S+)\)', l)
    if not m:
        continue
    n = LDC.match(ins[i+1])
    if n and int(n.group(2), 0) != 0 and int(m.group(1), 0) > int(n.group(2), 0):
        adj = f'{l}  ->  {ins[i+1]}'
        break
print('  adjacency (sd high, then ldc):', ('YES  ' + adj) if adj else 'NO')
if not adj:
    print('  the store is NOT adjacent to the load -- this image cannot create R-29 and must not')
    print('  be read as a clean result. Rebuild in the rung order or use the frozen .dom.')
raise SystemExit(0 if (ev and adj) else 1)
PY
  ;;

rung)
  verify
  : "${FPGA_URL:?set FPGA_URL to the board console URL (never commit or echo it)}"
  : "${FPGA_FW:?set FPGA_FW to the fw_payload.bin that has this .dom baked in}"
  export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r25r26r27_66c4e7517.bit}
  echo "== the .dom must be baked into the initramfs; stage it first if this is a fresh tree:"
  echo "     bash capstone/tests/stage-board-domains.sh --apply k800 s06agg.dom lpc"
  echo "   then rebuild: linux-rebuild, then opensbi-rebuild."
  echo
  echo "== control k800 runs FIRST (oracle 4), then the reproducer:"
  echo "     s06agg   64 = clean, 66 = R-29 (the HIGH half lost, the low half intact)"
  echo "     65 would mean something OTHER than R-29 -- do not record it as this defect."
  export SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,\
/test-domains/lpc|s06agg:/test-domains/s06agg.dom"
  bash capstone/tests/preflight-board-run.sh || { echo "preflight BLOCKED -- not spending a boot"; exit 1; }
  cd capstone/tests/rtl-smoke && exec python3 -m fpga_driver.run_sqlite_stages_fpga
  ;;

*)
  sed -n '2,15p' "$0"; exit 2
  ;;
esac
