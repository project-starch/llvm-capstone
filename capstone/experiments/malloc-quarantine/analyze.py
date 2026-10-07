#!/usr/bin/env python3
"""Summarise traced runs (mqtrace.so sample lines): one row per (arm, argv).

    analyze.py RESULT_DIR...

Peak values are maxima over the sampled points; maxrss comes from the
kernel (time -l), which sees the true peak, not only the sampled ones.
A run without MQ-DONE is reported as FAILED, never as a zero.
Revocation passes = final dequeue epoch / 2 (one pass advances it by 2).
"""
import re
import sys
from pathlib import Path


def setting(argv):
    """The program and its arguments, without env assignments and wrappers."""
    words = [w for w in argv.split() if w != 'env' and not w.startswith('./traced') and not re.match(r'\w+=', w)]
    return ' '.join(words)


def parse(path):
    text = path.read_text(errors='replace')
    arm = re.search(r'^MQ-ARM arm=(\S+).*argv=(.*)$', text, re.M)
    rows = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
            for line in text.splitlines() if line.startswith('MQ op=')]
    rss = re.search(r'^\s*(\d+)\s+maximum resident set size', text, re.M)
    pl = re.search(r'^MQ-DONE .*peak_live=(\d+)', text, re.M)
    return {
        'arm': arm.group(1) if arm else '?', 'argv': arm.group(2).strip() if arm else '?',
        'ok': 'MQ-DONE' in text and 'MQ-EXIT rc=0' in text, 'rows': rows,
        'maxrss_kib': int(rss.group(1)) if rss else None,
        'peak_live': int(pl.group(1)) if pl else None,
    }


def summary(run):
    rows = run['rows']
    live = max(r['live_req'] for r in rows)
    return {
        'live_mib': live / 2**20,
        'peak_alloc_mib': max(r['allocated'] for r in rows) / 2**20,
        'peak_resident_mib': max(r['resident'] for r in rows) / 2**20,
        'maxrss_mib': (run['maxrss_kib'] or 0) / 1024,
        'passes': rows[-1]['dequeue'] // 2,
        'alloc_over_live': max(r['allocated'] for r in rows) / max(live, 1),
    }


def main():
    runs = [parse(p) for d in sys.argv[1:] for p in sorted(Path(d).glob('*.txt'))]
    if not runs:
        sys.exit('no runs found in ' + ' '.join(sys.argv[1:]))
    base = {r['argv']: summary(r) for r in runs if r['ok'] and r['arm'] == 'off'}
    print(f"{'argv':44} {'arm':5} {'live':>8} {'peakAlloc':>9} {'peakRes':>8} {'maxRSS':>8} "
          f"{'RSS/off':>7} {'alloc/live':>10} {'passes':>6}")
    for r in sorted(runs, key=lambda r: (r['argv'], r['arm'])):
        if not r['ok'] or not r['rows']:
            print(f"{r['argv'][:44]:44} {r['arm']:5} FAILED")
            continue
        s = summary(r)
        b = base.get(r['argv'])
        ratio = f"{s['maxrss_mib'] / b['maxrss_mib']:.2f}" if b and b['maxrss_mib'] else '-'
        print(f"{r['argv'][:44]:44} {r['arm']:5} {s['live_mib']:8.2f} {s['peak_alloc_mib']:9.2f} "
              f"{s['peak_resident_mib']:8.2f} {s['maxrss_mib']:8.2f} {ratio:>7} "
              f"{s['alloc_over_live']:10.2f} {s['passes']:6d}")


if __name__ == '__main__':
    main()
