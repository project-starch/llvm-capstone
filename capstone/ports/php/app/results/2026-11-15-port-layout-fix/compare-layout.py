#!/usr/bin/env python3
"""The dynamic layout check of docs/plans/2026-11-15-php-port-layout-fix.md from two trace runs.

    compare-layout.py STOCK_RUN OTHER_RUN [OUT.json]

Each RUN is a trap_platform.py run bundle of a layout-trace image on bench.php (results.json with one
step). The script reads the ZEND-LAYOUT lines (zend-layout-trace.h) and the NESTED line, prints both
arms' counters per class, whether the sequence hashes match, and every request size whose count
differs below the 16 KiB split, and the bit-length buckets above it.
"""
import json
import re
import sys
from pathlib import Path


def read(run):
    step = json.loads((Path(run) / 'results.json').read_text())[0]
    text = step['output']
    out = {'exit': (step['kind'], step['value']), 'classes': {}, 'hist': {}, 'big': {}}
    for cls, fields in re.findall(r'^ZEND-LAYOUT (small|big) (.*)$', text, re.M):
        out['classes'][cls] = {k: (v if k == 'seq' else int(v)) for k, v in (w.split('=') for w in fields.split())}
    m = re.search(r'^ZEND-LAYOUT frees=(\d+)$', text, re.M)
    out['frees'] = int(m.group(1)) if m else None
    for line in re.findall(r'^ZEND-LAYOUT-HIST(.*)$', text, re.M):
        for w in line.split():
            size, n = w.split(':')
            out['hist'][int(size)] = int(n)
    m = re.search(r'^ZEND-LAYOUT-BIG(.*)$', text, re.M)
    for w in (m.group(1).split() if m else []):
        bits, n = w.split(':')
        out['big'][int(bits)] = int(n)
    m = re.search(r'^PHP_VIRTUAL NESTED (.*)$', text, re.M)
    out['nested'] = m.group(1) if m else None
    return out


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    print(f"exit stock {a['exit']} other {b['exit']}; frees stock {a['frees']} other {b['frees']}")
    for cls in ('small', 'big'):
        x, y = a['classes'].get(cls, {}), b['classes'].get(cls, {})
        print(f'{cls}:')
        for k in ('mallocs', 'reallocs', 'inplace', 'moved', 'failed', 'bytes', 'seq'):
            print(f"  {k:9} stock {x.get(k)!s:>20}  other {y.get(k)!s:>20}  {'equal' if x.get(k) == y.get(k) else 'DIFFERENT'}")
    sizes = sorted(set(a['hist']) | set(b['hist']))
    diff = [(s, a['hist'].get(s, 0), b['hist'].get(s, 0)) for s in sizes if a['hist'].get(s, 0) != b['hist'].get(s, 0)]
    print(f'small histogram: {len(sizes)} sizes, {len(diff)} with different counts')
    for s, x, y in diff[:40]:
        print(f'  size {s}: stock {x} other {y}')
    bits = sorted(set(a['big']) | set(b['big']))
    print('big by bit length: ' + ', '.join(f"{k}: {a['big'].get(k, 0)}/{b['big'].get(k, 0)}" for k in bits))
    print(f"nested line (other): {b['nested']}")
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({'stock': a, 'other': b, 'small_hist_diff': diff}, indent=1) + '\n')


if __name__ == '__main__':
    main()
