#!/usr/bin/env python3
"""OBB_RUN for the dry run: stands in for run_sup_bare.py. No console call. Writes <SUP_OUT>/uart.txt and run.log as
the runner does, with the vector the RTL lane's PREREG tables give for this image under scenario $OBB_SCEN, and exits
0 (SUPTEST END seen) or 2 (no return: the runner's timeout path).
Scenarios: objb-confirmed, objb-stop (objb-stale reads bitstream 3's vector), objb-void (objb-pcc-ctrl reads wrong),
bs3-pass (bitstream-3 vectors, objb-pcc does not return), bs3-finding (bitstream 3, objb-stale off by one word),
bs3-stop (bitstream 3, objb-pcc returns)."""
import os, sys, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('g', str(Path(__file__).resolve().parent.parent / 'objb-bare-grade.py'))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
img = Path(os.environ['SUP_IMG']).stem; out = Path(os.environ['SUP_OUT']); out.mkdir(parents=True, exist_ok=True)
scen = os.environ.get('OBB_SCEN', 'objb-confirmed')
sr, v3, vo, _ = g.T[img]
arm = 'bs3' if scen.startswith('bs3') else 'objb'
vec = list(vo if arm == 'objb' else (v3 or []))
ret = not (arm == 'bs3' and img == 'objb-pcc')
if scen == 'objb-stop' and img == 'objb-stale': vec = list(v3)
if scen == 'objb-void' and img == 'objb-pcc-ctrl': vec = [0xbb5, 0x19, 0]
if scen == 'bs3-finding' and img == 'objb-stale': vec[7] = 0x3333
if scen == 'bs3-stop' and img == 'objb-pcc': ret, vec = True, list(vo)
txt = f'SUPTEST BEGIN {img}\nSB1\n'
if ret:
    txt += f'SR {len(vec):016X}\n' + ''.join(f'SV {v:016X}\n' for v in vec) + 'SUPTEST END\n'
(out / 'uart.txt').write_text(txt)
(out / 'run.log').write_text(f'[sup stub] image {img} scenario {scen} returned={ret} bitstream {os.environ.get("FPGA_BITSTREAM")} rec {os.environ.get("SUP_REC_ADDR")}\n')
assert os.environ.get('FPGA_URL'), 'FPGA_URL not set'
sys.exit(0 if ret else 2)
