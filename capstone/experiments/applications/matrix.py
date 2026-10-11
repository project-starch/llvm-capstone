#!/usr/bin/env python3
"""Build a bounded discovery matrix; sizes are records, never allocator events."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--share', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--sizes', type=int, nargs='+', default=[32, 128, 512])
p.add_argument('--batches', type=int, default=8)
p.add_argument('--retained', type=int, nargs='+', default=[0, 256])
a = p.parse_args()
points = []
scripts = dict(perl='records.pl', cpython='objects.py', mruby='records.rb')
for app, arm in [('perl','level0'), ('cpython','level0'), ('mruby','level0'),
                 ('sqlite','level0'), ('mruby','gc-sublet')]:
    for n in a.sizes:
        for retained in a.retained:
            argv = (['-I','/mnt/host/perl-tests/lib'] if app == 'perl' else
                    ['-P','-S'] if app == 'cpython' else [])
            if app in scripts: argv += ['/mnt/host/experiments/'+scripts[app]]
            argv += list(map(str, [n, a.batches, retained]))
            checksum = (a.batches-1)*n*(n-1)//2 + (4*n)*(4*n-1)//2
            points.append(dict(id=f'{app}-{arm}-n{n}-keep{retained}', application=app,
                arm=arm, size=n, batches=a.batches, retained=retained,
                image=str(a.share/f'{app}-{arm}.dom'), argv=argv,
                environment={'PYTHONHOME':'/mnt/host/experiments/pyhome'} if app == 'cpython' else {},
                expected_stdout=f'EXP-OK {app} {checksum}\n',
                expected_phases=['startup','baseline']+[f'{kind}-{epoch}' for epoch in range(a.batches)
                    for kind in ('live','released')]+['exit']))
with a.out.open('x') as f: json.dump(points, f, indent=2); f.write('\n')
