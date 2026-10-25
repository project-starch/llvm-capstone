#!/usr/bin/env python3
"""compare-gate.py A B [tags...]: two-sided per-case comparison of Tier 3 chain records under A/<t3-tag> and B/<t3-tag>.
Canonical verdict = case, verdict, reason, infra, cause, symbol, offset, attribution, reached, completed, image sha,
and every control's (name, observed, cause). Serial signature = multiset of (cause, address==0, sha prefix) over
'domain fault' lines. No data is an ERROR (exit 2), never a match."""
import collections, json, re, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2])
tags = sys.argv[3:] or sorted(p.name[3:] for p in A.glob('t3-*') if p.is_dir())
def canon(r):
    o = r.get('observation') or {}
    f = o.get('fault') or {}
    ctl = tuple(sorted((c.get('name'), c.get('observed'),
                        (re.search(r'cause=(\d+)', c.get('evidence') or '') or [None, None])[1]) for c in o.get('controls') or []))
    return (o.get('case'), r.get('verdict'), r.get('reason'), json.dumps(o.get('infra'), sort_keys=True), f.get('cause'),
            f.get('symbol'), f.get('offset'), o.get('attribution'), o.get('reached'), o.get('completed'), o.get('image_sha256'), ctl)
def verdicts(d):
    vs = sorted(d.rglob('verdicts.jsonl'))
    if not vs: return None
    return sorted(canon(json.loads(l)) for v in vs for l in v.read_text().splitlines() if l.strip())
def signature(d):
    logs = sorted(d.rglob('serial.log'))
    if not logs: return None
    c = collections.Counter()
    for s in logs:
        for m in re.finditer(r'domain fault cause=(\d+) pc=\S+ address=(0x[0-9a-f]+).*?sha256=([0-9a-f]+)', s.read_text(errors='replace')):
            c[(int(m[1]), int(m[2], 16) == 0, m[3])] += 1
    return c
def refused(d):
    """Refusal lines over the arm's guest consoles; None (NO-DATA) unless EVERY console shows the load banner."""
    total = 0
    for s in d.rglob('serial.log'):
        t = s.read_text(errors='replace')
        if 'capstone_vm: loaded, diag' not in t: return None
        total += t.count('wrote no event record')
    return total
bad = nodata = 0
for t in tags:
    a, b = A / ('t3-' + t), B / ('t3-' + t)
    va, vb, sa, sb = verdicts(a), verdicts(b), signature(a), signature(b)
    if None in (va, vb, sa, sb):
        nodata += 1; print(f'{t:6} NO-DATA verdicts {va is not None}/{vb is not None} serial {sa is not None}/{sb is not None}'); continue
    same_v, same_s = va == vb, sa == sb
    ra, rb = refused(a), refused(b)
    if ra is None or rb is None:
        nodata += 1; print(f'{t:6} NO-DATA a guest console lacks the load banner (refusal count void)'); continue
    print(f'{t:6} cases {len(va):3}/{len(vb):3} verdicts {"EQUAL" if same_v else "DIFFER"} faults {sum(sa.values()):3}/{sum(sb.values()):3} '
          f'signature {"EQUAL" if same_s else "DIFFER"} refused-lines {ra}/{rb} :: {dict(collections.Counter(x[1] for x in va))}')
    if not same_v:
        for x in sorted(set(va) ^ set(vb)): print('   ', 'A' if x in va else 'B', x)
    if not same_s:
        print('    A', dict(sa)); print('    B', dict(sb))
    bad += (not same_v) or (not same_s)
print(f'TOTAL arms {len(tags)} differ {bad} no-data {nodata}')
sys.exit(2 if nodata else (1 if bad else 0))
