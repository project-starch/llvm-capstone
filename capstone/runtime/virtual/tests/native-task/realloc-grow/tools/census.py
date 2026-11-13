#!/usr/bin/env python3
"""census.py SERIAL.LOG: the diag_admit census of one native twin run: per-arena retire lines (size, pages admitted at a
fault, by a window, by the scan, swept at retire), fault-ahead windows, and the @@VM page counters. ERROR if no retire."""
import re, sys, collections
t = open(sys.argv[1], errors='replace').read().replace('\r', '')
R = re.findall(r'retire arena (0x[0-9a-f]+)\+(0x[0-9a-f]+) admit=(\d+) window=(\d+) scan=(\d+) touch=(\d+) eager=(\d+) '
               r'released=(\d+) swept=(\d+) faulted=(\d+)', t)
if not R: sys.exit('ERROR: no retire lines in ' + sys.argv[1])
W = re.findall(r'fault-ahead arena (0x[0-9a-f]+) page (\d+) want (\d+) pinned (\d+)', t)
A = re.findall(r'capstone_vm: admit arena (0x[0-9a-f]+)\+(0x[0-9a-f]+) page', t)
vm = re.findall(r'^@@VM(\d) \d+ vm=(\S+) p=(\S+)', t, re.M)
for k, v, p in vm: print('VM', k, v, p)
if len(vm) == 2:
    d = lambda s: dict((x.split(':')[0], int(x.split(':')[1])) for x in s.strip(',').split(',') if ':' in x and x.split(':')[1])
    a, b = d(vm[0][1]), d(vm[1][1]); print('VM delta', {k: b[k] - a[k] for k in a})
tot = collections.Counter()
by = collections.defaultdict(lambda: collections.Counter())
for addr, size, *n in R:
    n = list(map(int, n)); size = int(size, 16)
    for k, x in zip('admit window scan touch eager released swept faulted'.split(), n): tot[k] += x; by[size][k] += x
    by[size]['arenas'] += 1
print(f'retire lines {len(R)}; admit lines {len(A)}; fault-ahead windows {len(W)} pinned {sum(int(x[3]) for x in W)}')
print('totals', dict(tot))
print(f'{"arena bytes":>14} {"arenas":>7} {"admit":>8} {"window":>8} {"scan":>6} {"swept":>8} {"faulted":>8}  swept MiB')
for size, c in sorted(by.items(), key=lambda x: -x[1]['swept'])[:25]:
    print(f'{size:>14,} {c["arenas"]:>7} {c["admit"]:>8} {c["window"]:>8} {c["scan"]:>6} {c["swept"]:>8} {c["faulted"]:>8}  {c["swept"]*4/1024:8.1f}')
live = collections.Counter(a for a, s in A)
print('admit lines by arena (top, incl. arenas never retired):')
sizes = dict(A)
for a, n in live.most_common(12): print(f'  {a} +{int(sizes[a],16):,}: {n}')
