#!/usr/bin/env python3
"""Out-of-sample check of the perlbench k-r model on single-context runs: kcyc - rcyc = A*steps + B*(inside cycles/250,000)."""
import re, sys, glob
A, B = float(sys.argv[1]), float(sys.argv[2])
import os
B0 = os.environ.get('SPEC_BOARD_OUT', 'spec/board') + '/'
runs = sys.argv[3:]
if not runs: sys.exit('NO DATA: no runs given')
for r in runs:
    try: t = open(B0 + r + '/spec.txt', 'rb').read().decode('latin-1')
    except OSError: print('NO FILE', r); continue
    b = re.findall(r'capstone_vm: bench cycles=(\d+) instret=(\d+) entries=(\d+)', t)
    e = re.findall(r'capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)', t)
    if not b or not e: print('NO DATA', r); continue
    # the last context is the benchmark; the control's lines come first
    cyc, ins, ent = map(int, b[-1]); st, k, rc, u, o = map(int, e[-1][:5]); ev = e[-1][5]
    pred = A * st + B * cyc / 250000.0
    meas = k - rc
    print(f'{r[:60]:60s} inside {cyc/25e6:8.1f}s steps {st:6d} ev {ev:22s} k-r meas {meas/1e9:6.3f}G pred {pred/1e9:6.3f}G  ratio {meas/pred:5.3f}  ticks-term {B*cyc/250000/25e6:6.1f}s')
