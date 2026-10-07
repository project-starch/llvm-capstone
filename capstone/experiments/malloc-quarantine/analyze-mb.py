#!/usr/bin/env python3
"""Summarise unmodified-program runs (run.sh with MQ_PRELOAD=1).

    analyze-mb.py RESULT_DIR

max RSS is the kernel's (time -l); passes = final dequeue epoch / 2 from
mqstat.so.  Wall/user time is printed for orientation only: it is QEMU time.
A run whose exit status is not 0 or that lacks MQ-EXIT-STATS is FAILED.
"""
import re
import sys
from pathlib import Path


def parse(path):
    t = path.read_text(errors='replace')
    arm = re.search(r'^MQ-ARM arm=(\S+).*argv=(.*)$', t, re.M)
    st = re.search(r'^MQ-EXIT-STATS (.*)$', t, re.M)
    rss = re.search(r'^\s*(\d+)\s+maximum resident set size', t, re.M)
    real = re.search(r'^\s*([\d.]+) real', t, re.M)
    stats = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', st.group(1))) if st else {}
    return {'arm': arm.group(1), 'argv': arm.group(2).strip(), 'stats': stats,
            'ok': 'MQ-EXIT rc=0' in t and bool(st), 'rss': int(rss.group(1)) if rss else None,
            'real': float(real.group(1)) if real else None}


def main():
    runs = [parse(p) for p in sorted(Path(sys.argv[1]).glob('*.txt'))]
    if not runs:
        sys.exit('no runs in ' + sys.argv[1])
    base = {r['argv']: r['rss'] for r in runs if r['ok'] and r['arm'] == 'off'}
    print(f"{'program':28} {'arm':5} {'maxRSS MiB':>10} {'RSS/off':>7} {'passes':>6} {'qemu s':>7}")
    for r in sorted(runs, key=lambda r: (r['argv'], ['off', 'on', 'sync'].index(r['arm'])
                                          if r['arm'] in ('off', 'on', 'sync') else 9)):
        if not r['ok']:
            print(f"{r['argv'][:28]:28} {r['arm']:5} FAILED")
            continue
        b = base.get(r['argv'])
        print(f"{r['argv'][:28]:28} {r['arm']:5} {r['rss'] / 1024:10.2f} "
              f"{(r['rss'] / b if b else float('nan')):7.2f} {r['stats']['dequeue'] // 2:6d} "
              f"{r['real'] or 0:7.1f}")


if __name__ == '__main__':
    main()
