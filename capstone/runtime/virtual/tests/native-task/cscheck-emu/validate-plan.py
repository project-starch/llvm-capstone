#!/usr/bin/env python3
"""validate-plan.py OUT PLAN_DIR: run the plan's host_validation (official specdiff) over OUT/results; print n/m."""
import json, os, subprocess, sys
from pathlib import Path
out, plan = Path(sys.argv[1]), Path(sys.argv[2])
SP = os.environ['SPEC_TREE']  # the SPEC installation holding the references
p = json.loads((plan / 'plan.json').read_text())
ok = 0
for v in p['host_validation']:
    argv = [SP + a[4:] if a.startswith('SPEC/') else a for a in v['argv']]
    r = subprocess.run(argv, cwd=out / 'results', env=dict(os.environ, SPEC=SP), capture_output=True, text=True)
    (out / 'results' / (v['output'] or (v['argv'][-1] + '.cmp'))).write_text(r.stdout + r.stderr)
    ok += r.returncode == 0
    if r.returncode: print('FAIL', v['argv'][-1], r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-200:])
print(f'specdiff {ok}/{len(p["host_validation"])}')
