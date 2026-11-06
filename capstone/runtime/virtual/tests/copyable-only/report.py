#!/usr/bin/env python3
"""Attribute six alternating pinned/unpinned windows, using each load's module addresses."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True, help='Contains run/serial.log and capprof/')
p.add_argument('--platform', type=Path, required=True)
p.add_argument('--program', type=Path, required=True)
p.add_argument('--nm', default='nm')
a = p.parse_args()
a.out = a.out.resolve()
s = (a.out / 'run/serial.log').read_text(errors='replace').replace('\r', '')
cases = re.findall(r'COPYABLE_MEASURE_BEGIN:(\d):([01])\n(.*?)COPYABLE_MEASURE_END:\1:\2\n', s, re.S)
if (list((a.out / 'capprof').glob('*.invalid')) or
        {p.name for p in (a.out / 'capprof').glob('window-*.tsv')} !=
        {f'window-{i}.tsv' for i in range(6)}):
    raise SystemExit('missing, extra, or invalid profiling windows')
if len(cases) != 6 or 'COPYABLE_MEASURE_EXIT:0' not in s:
    raise SystemExit('incomplete measurement')
if [(r, arm) for r, arm, _ in cases] != [('1','0'),('1','1'),('2','1'),('2','0'),('3','0'),('3','1')]:
    raise SystemExit('unexpected order')
functions = ['pin_user_pages_remote', '__get_user_pages', 'handle_mm_fault',
             '__handle_mm_fault', 'fixup_user_fault', 'sys_ioctl',
             'sys_mmap', 'sys_munmap']
addresses = {}
for line in (a.platform / 'kernel/System.map').read_text().splitlines():
    pc, _, name = line.split()[:3]
    if name in functions:
        addresses[name] = int(pc, 16)
missing = set(functions) - addresses.keys()
if missing:
    raise SystemExit(f'missing entry symbols: {missing}')
reporter = Path(__file__).resolve().parents[4] / 'ports/spec-cpu2017/tools/prof-report.py'
rows = []
for i, (rep, arm, block) in enumerate(cases):
    if block.count('PASS metadata-break n=16384 sum=5210112') != 1:
        raise SystemExit(f'workload oracle failed in window {i}')
    out = a.out / f'attributed-{i}'
    out.mkdir()
    (out / 'capprof').mkdir()
    (out / 'invocations').mkdir()
    raw = a.out / 'capprof' / f'window-{i}.tsv'
    (out / 'capprof' / raw.name).symlink_to(raw)
    (out / 'module-sections.txt').write_text('\n'.join(l[8:] for l in block.splitlines() if l.startswith('SECTION ')) + '\n')
    (out / 'image-records.txt').write_text('\n'.join(l for l in block.splitlines() if 'domain image entry=' in l) + '\n')
    invocation = dict(route='guest', exe=str(a.program.resolve()), argv=['16384'], detail={'window': i})
    (out / 'invocations/0000.json').write_text(json.dumps(invocation))
    with (out / 'report.txt').open('w') as log:
        subprocess.run(['python3', str(reporter), '--out', str(out), '--platform', str(a.platform),
                        '--kind', 'capstone', '--nm', a.nm], check=True, stdout=log)
    row = json.loads((out / 'profile.json').read_text())['windows'][0]
    row.update(repeat=int(rep), unpinned=bool(int(arm)), entry_attempts={f: 0 for f in functions})
    row['boundary_attempts'] = sum(v for k, v in row['classes'].items() if k != 'program')
    for line in raw.read_text().splitlines():
        if line.startswith('#'):
            continue
        pc, _, _, attempts, _ = line.split('\t')
        for name, address in addresses.items():
            if int(pc, 16) == address:
                row['entry_attempts'][name] += int(attempts)
    rows.append(row)
summary = {'workload': 'two rounds of 16384 allocations; sizes 64+(i%32)*128; touch ends and free',
           'program_sha256': hashlib.sha256(a.program.read_bytes()).hexdigest(),
           'unit': 'QEMU instruction attempts inside workload markers; not cycles or elapsed time',
           'same_binary_both_arms': True, 'rows': rows, 'medians': {}}
for flag, name in [(False, 'pinned'), (True, 'unpinned')]:
    selected = [r for r in rows if r['unpinned'] == flag]
    classes = set().union(*(r['classes'] for r in selected))
    summary['medians'][name] = {
        **{k: statistics.median(r[k] for r in selected) for k in ['total_attempts', 'boundary_attempts']},
        'classes': {c: statistics.median(r['classes'].get(c, 0) for r in selected) for c in sorted(classes)},
        'entry_attempts': {f: statistics.median(r['entry_attempts'][f] for r in selected) for f in functions},
        'total_range': [min(r['total_attempts'] for r in selected), max(r['total_attempts'] for r in selected)]}
x, y = (summary['medians'][k]['total_attempts'] for k in ['pinned', 'unpinned'])
summary['total_change_percent'] = 100 * (y / x - 1)
(a.out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['medians'], indent=2))
print('Total change: %.3f%%' % summary['total_change_percent'])
