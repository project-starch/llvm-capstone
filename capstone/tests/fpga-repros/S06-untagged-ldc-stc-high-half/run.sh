#!/usr/bin/env bash
# S-06: a 16-byte ldc/stc round trip over PLAIN data keeps only the low 8 bytes of each chunk.
#
#   ./run.sh sim        RTL simulation, ~14 s, carries its own control   NO BOARD  <- start here
#   ./run.sh verify     check the frozen artifacts                       NO BOARD
#   ./run.sh rung       the 10 KB reproducer on the board
#
# The rung image is FROZEN and checksummed on purpose. This platform has a per-image entry stall
# (see ../R16-entry-stall/): a rebuilt image is a fresh draw and may simply never run, so the
# shipped .dom is a draw that has been observed to enter. The board mode runs the known-good
# control k800 FIRST -- a boot whose control fails carries no verdict about anything.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half
MODE=${1:-verify}

verify() {
  echo "== verifying frozen artifacts =="
  ( cd "$D" && sha256sum -c SHA256SUMS ) || { echo "ARTIFACTS DO NOT MATCH -- do not trust any run"; exit 1; }
}

case "$MODE" in
sim)
  verify
  exec bash "$D/sim/run-sim.sh"
  ;;

verify)
  verify
  echo
  echo "the board reproducer must still CONTAIN the construct under test:"
  python3 - <<'PY'
import subprocess, re, pathlib
dom = 'capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half/src/s06copy.dom'
od  = 'llvm/cmake-build-debug/bin/llvm-objdump'
if not pathlib.Path(od).exists():
    print('  llvm-objdump not built here -- skipping the construct check'); raise SystemExit(0)
out = subprocess.run([od, '-d', '--triple=capstone64-unknown-elf',
                      '--disassemble-symbols=s06copy_compute', dom],
                     capture_output=True, text=True).stdout
# split with maxsplit=1: objdump lines are "addr: bytes \t mnemonic \t operands", so a plain
# split('\t')[-1] yields the OPERANDS and every mnemonic match silently fails.
ins = [l.split('\t', 1)[-1].strip() for l in out.split('\n') if '\t' in l]
LDC = re.compile(r'ldc\s+(\w+),\s*(-?0x[0-9a-f]+|\d+)\((\w+)\)')
STC = re.compile(r'stc\s+(\w+),\s*(-?0x[0-9a-f]+|\d+)\((\w+)\)')
# Match the COPY, not any ldc/stc. Same data register, same base register, DIFFERENT offsets --
# a move between two offsets of one buffer. A looser "is there an ldc near an stc" test matches
# ordinary prologue spill/restore and answers YES for rungs that contain no copy at all; it was
# written that way first and k800 exposed it.
ev = None
for i, l in enumerate(ins):
    m = LDC.match(l)
    if not m:
        continue
    for x in ins[i+1:i+4]:
        n = STC.match(x)
        if n and n.group(1) == m.group(1) and n.group(3) == m.group(3) and n.group(2) != m.group(2):
            ev = f'{l}  ->  {x}'
            break
    if ev:
        break
print('  s06copy_compute  ldc/stc block copy :', ('YES  ' + ev) if ev else 'NO')
if not ev:
    print('  the rung would return 32 while testing NOTHING -- do not run it')

# s06agg must contain the AGGREGATE copy: an ldc/stc pair at a NON-ZERO offset, which is the
# plain-data tail of the struct. Offset 0 is the pointer chunk and is safe, so a check that
# accepted any ldc/stc would pass on a build whose data copy had been optimised away.
out2 = subprocess.run([od, '-d', '--triple=capstone64-unknown-elf',
                       '--disassemble-symbols=s06agg_compute',
                       dom.replace('s06copy.dom', 's06agg.dom')],
                      capture_output=True, text=True).stdout
ins2 = [l.split('\t', 1)[-1].strip() for l in out2.split('\n') if '\t' in l]
ev2 = None
for i, l in enumerate(ins2):
    m = LDC.match(l)
    if not m or int(m.group(2), 0) == 0:
        continue
    for x in ins2[i+1:i+4]:
        n = STC.match(x)
        if n and n.group(1) == m.group(1) and int(n.group(2), 0) != 0:
            ev2 = f'{l}  ->  {x}'
            break
    if ev2:
        break
print('  s06agg_compute   aggregate copy     :', ('YES  ' + ev2) if ev2 else 'NO')
if not ev2:
    print('  the struct copy was not emitted capability-grained -- this rung proves nothing')
raise SystemExit(0 if (ev and ev2) else 1)
PY
  ;;

rung)
  verify
  : "${FPGA_URL:?set FPGA_URL to the board console URL (never commit or echo it)}"
  : "${FPGA_FW:?set FPGA_FW to the fw_payload.bin that has this .dom baked in}"
  export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r20.bit}
  echo "== the .dom must be baked into the initramfs; stage it first if this is a fresh tree:"
  echo "     bash capstone/tests/stage-board-domains.sh --apply k800 s06copy.dom s06agg.dom lpc"
  echo "   then rebuild: linux-rebuild, then opensbi-rebuild TWICE."
  echo
  echo "== control k800 runs FIRST (oracle 4), then both reproducers:"
  echo "     s06copy  32 = clean, 16 = S-06 through memcpy's aligned loop"
  echo "     s06agg   64 = clean, 66 = S-06 through the compiler's struct assignment (no memcpy)"
  export SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,\
/test-domains/lpc|s06copy:/test-domains/s06copy.dom,\
/test-domains/lpc|s06agg:/test-domains/s06agg.dom,\
/test-domains/lpc|s06agg:/test-domains/s06agg.dom"
  bash capstone/tests/preflight-board-run.sh || { echo "preflight BLOCKED -- not spending a boot"; exit 1; }
  cd capstone/tests/rtl-smoke && exec python3 -m fpga_driver.run_sqlite_stages_fpga
  ;;

*)
  sed -n '2,12p' "$0"; exit 2
  ;;
esac
