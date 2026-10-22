#!/usr/bin/env python3
"""lp-parse.py SERIAL [--json OUT]: the live-peak QEMU's record counters (measure/live-peak), per table instance, per step.

A step starts at '@@STEP-BEGIN <name>' (virtualvm) or 'LP_BEGIN:<name>' (the control's gate). An instance belongs to the
step in which its NEW line appears. Every line carries the instance's running peak and minted count. The staged runners
stop reading QEMU's output at the gate's end marker, so no closing line can be relied on. Per instance, the reading is
bracketed:
  peak_lb   = the largest peak printed (a PEAK line comes within 64 of a peak once live falls from it);
  peak_ub   = just below the next 1,024 step no STEP line reached;
  minted_lb = the largest minted printed; minted_ub = minted_lb + 4,095 (MINT every 4,096), or exact from a RESET/EXIT.
Exit 2 (NO DATA) if the counters were never switched on or no instance appeared."""
import json, re, sys
L = re.compile(r'CAPSTONE_LIVE_PEAK (\w+) inst=(\d+) root=(0x[0-9a-f]+) live=(\d+) peak=(\d+) minted=(\d+) retired=(\d+) next_max=(\d+)')
text = open(sys.argv[1], 'rb').read().decode('utf-8', 'replace').replace('\r', '')
if 'CAPSTONE_LIVE_PEAK ON' not in text:
    print(f"NO DATA: no 'CAPSTONE_LIVE_PEAK ON' line in {sys.argv[1]}"); sys.exit(2)
step, insts, order = '(boot)', {}, []
for line in text.split('\n'):
    m = re.search(r'@@STEP-BEGIN (\S+)|LP_BEGIN:(\S+)', line)
    if m: step = m.group(1) or m.group(2); continue
    m = L.search(line)
    if not m: continue
    kind, inst = m.group(1), int(m.group(2))
    live, peak, minted, retired = map(int, m.groups()[3:7])
    d = insts.setdefault(inst, {'inst': inst, 'root': m.group(3), 'step': None, 'peak_lb': 0, 'step_max': 0,
                                'minted_lb': 0, 'exact': False, 'kinds': {}})
    d['kinds'][kind] = d['kinds'].get(kind, 0) + 1
    if kind == 'NEW': d['step'] = step; order.append(inst)
    if kind == 'STEP': d['step_max'] = max(d['step_max'], peak)
    d['peak_lb'] = max(d['peak_lb'], peak); d['minted_lb'] = max(d['minted_lb'], minted)
    if kind in ('RESET', 'EXIT'): d['exact'] = True; d['peak_ub'] = peak; d['minted_ub'] = minted
for i in order:
    d = insts[i]
    if not d['exact']:
        d['peak_ub'] = (d['step_max'] // 1024 + 1) * 1024 - 1
        d['minted_ub'] = d['minted_lb'] + 4095
    print(f"INST\t{d['step']}\tinst={i}\troot={d['root']}\tpeak=[{d['peak_lb']},{d['peak_ub']}]\tminted=[{d['minted_lb']},{d['minted_ub']}]\t{d['kinds']}")
if not order:
    print("NO DATA: no instance"); sys.exit(2)
if len(sys.argv) > 3 and sys.argv[2] == '--json':
    json.dump([insts[i] for i in order], open(sys.argv[3], 'w'), indent=1)
sys.exit(0)
