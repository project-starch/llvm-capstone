#!/usr/bin/env bash
# R-20: after `stc rX,0(a0)`, a `ld a0,0(a0)` is read by the NEXT instruction as the store's base
# address instead of the loaded value. Only on x10/a0; only with a capability store; only when
# both adjacencies hold.
#
#   ./run.sh sim                         RTL simulation, ~14 s             NO BOARD  <- start here
#   ./run.sh verify                      check frozen artifacts            NO BOARD
#   ./run.sh arms <base sqlite_silicon.dom>   regenerate the SQLite arms   NO BOARD
#   ./run.sh rung                        the 13 KB reproducer on the board
#
# The rung images are FROZEN and checksummed on purpose. This platform has a per-image entry stall
# (see ../R16-entry-stall/): a rebuilt image is a fresh draw and may simply never run. Three draws
# of byte-identical code are shipped for that reason -- if one produces no `SQ: G/enter`, use the
# next. The board mode runs the known-good control k800 FIRST; a boot whose control fails carries
# no verdict about anything.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10
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
  echo "sbx_compute must be byte-identical across the three draws (only the link offset differs):"
  python3 - <<'PY'
import re, subprocess, hashlib, pathlib
D = pathlib.Path('capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/src')
OBJDUMP = 'llvm/cmake-build-debug/bin/llvm-objdump'
def fn(p):
    d = subprocess.run([OBJDUMP, '-d', '--triple=capstone64-unknown-elf', str(p)],
                       capture_output=True, text=True).stdout
    st, out = False, []
    for l in d.split('\n'):
        if '<sbx_compute>:' in l: st = True; continue
        if st:
            if re.match(r'^[0-9a-f]+ <', l): break
            m = re.match(r'^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*\t', l)
            if m: out.append(m.group(1).replace(' ', ''))
    return ''.join(out)
h = {p.name: hashlib.md5(fn(p).encode()).hexdigest()[:12] for p in sorted(D.glob('sbx*.dom'))}
i = {p.name: hashlib.md5(p.read_bytes()).hexdigest()[:12] for p in sorted(D.glob('sbx*.dom'))}
for k in h: print(f'  {k:12s} image={i[k]}  sbx_compute={h[k]}')
ok = len(set(h.values())) == 1 and len(set(i.values())) == len(i)
print('  OK -- same code, distinct images' if ok else '  PROBLEM -- draws are not equivalent')
raise SystemExit(0 if ok else 1)
PY
  ;;

arms)
  BASE=${2:-}
  [ -n "$BASE" ] || { echo "usage: ./run.sh arms <base sqlite_silicon.dom>"; exit 2; }
  verify
  OUT=$(mktemp -d)
  python3 "$D/board/make-arms.py" "$BASE" "$OUT" || exit 1
  echo
  echo "== checking regenerated arms against the binaries that were actually run =="
  ( cd "$OUT" && grep -v '^#' "$OLDPWD/$D/board/ARM-SHA256SUMS" | grep -v 'sqlite_silicon' | sha256sum -c - ) \
    || { echo "REGENERATED ARMS DIFFER from those run on the board"; exit 1; }
  echo "arms in $OUT"
  ;;

rung)
  : "${FPGA_URL:?set FPGA_URL (secret; never commit or echo it)}"
  : "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
  export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_65536_r18_fix.bit}
  verify
  O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
  T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
  for f in "$D"/src/sbx*.dom; do cp -f "$f" "$O/" && cp -f "$f" "$T/"; done
  echo "staged $(ls "$D"/src/sbx*.dom | wc -l) draws; rebuild the firmware, then:"
  echo "  cd capstone/caplifive-system/sw/buildroot"
  echo "  make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH=\$(realpath ../../../capstone-c)"
  echo "  make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH=\$(realpath ../../../capstone-c)   # twice"
  echo
  echo "then run, control FIRST:"
  echo '  SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/lpc|sbx8:/test-domains/sbx8.dom" \'
  echo '    python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py'
  echo
  echo "EXPECT  RESULT sbx8 retval=3489660929  (0xD0000001)"
  echo "  bit0 set  = the defect"
  echo "  bit4/bit6 MUST be clear -- they are the instrument validation; if either is set the run is void"
  ;;

*) sed -n '2,14p' "$0"; exit 2 ;;
esac
