#!/usr/bin/env python3
"""Tabulate the 19 temporal SQLite cases of the paper's Table 2 for PoisonCap and for Sublet.

Run in this record's directory. Sixteen cases come from ../2026-10-26-poisoncap-heap256 (runs
driver-la and v6-sublet-block, with their repetitions), three (02, 11, 13) from this record.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OLD = HERE.parent / '2026-10-26-poisoncap-heap256'
POP = '02 03 05 06 07 10 11 12 13 14 15 16 17 19 20 21 22 23 24'.split()
NEW = ('02', '11', '13')

# A case that PoisonCap does not reject counts as "in quarantine only" when the stale object's block
# was freed and was still in PoisonCap's quarantine at the stale access. The verdict files do not
# hold that; the diagnostic runs named here do.
IN_QUARANTINE = {
    '11': 'runs/c19-diag11-la (this record)',
    '15': '../2026-10-24-poisoncap/EVIDENCE.md (diag3-la)',
    '16': '../2026-10-26-poisoncap-heap256/README.md (diag-driver16-la)',
    '17': '../2026-10-26-poisoncap-heap256/README.md (diag-driver16-la)',
}


def poisoncap(record, run):
    """{case number: (state, detail)} from score-cases.py's verdicts.md."""
    out = {}
    for line in (record / 'verdicts' / run / 'verdicts.md').read_text().splitlines():
        m = re.match(r'\| (\d\d)_[^|]*\| ([^|]*)\|', line)
        if not m:
            continue
        reading = m.group(2).strip()
        if reading.startswith('fault'):
            out[m.group(1)] = ('at site', reading)
        elif reading == 'returned':
            out[m.group(1)] = ('returned', reading)
        elif reading == 'no-reading':
            out[m.group(1)] = ('timeout', 'neither faults nor returns within the timeout')
        else:
            out[m.group(1)] = ('other', reading)
    return out


def sublet(record, run):
    """{case number: (state, detail)} from run-virtual-staged.py's results.jsonl."""
    out = {}
    for line in (record / 'runs' / run / 'results.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['name'].startswith('control'):
            continue
        fault = r.get('fault') or {}
        if r.get('at_site'):
            out[r['name'][:2]] = ('at site', f"fault {fault.get('function')} cause={fault.get('cause')}")
        elif r.get('returned'):
            out[r['name'][:2]] = ('returned', 'returned')
        else:
            out[r['name'][:2]] = ('other', f"fault {fault.get('function')} cause={fault.get('cause')}")
    return out


def merged(reader, old_run, new_run):
    old, new = reader(OLD, old_run), reader(HERE, new_run)
    return {c: (new if c in NEW else old)[c] for c in POP}


def group(case, state):
    if state == 'at site':
        return 'rejected at site'
    if case in IN_QUARANTINE:
        return 'in quarantine only'
    return 'neither'


def main():
    pc, pc2 = merged(poisoncap, 'driver-la', 'c19-driver-la'), merged(poisoncap, 'driver-la-rep2', 'c19-driver-la-rep2')
    sl, sl2 = merged(sublet, 'v6-sublet-block', 'c19-sublet-block'), merged(sublet, 'v6-sublet-block-rep2', 'c19-sublet-block-rep2')
    print('| case | record | PoisonCap | PoisonCap reading | Sublet, block port | Sublet reading |')
    print('|---|---|---|---|---|---|')
    for c in POP:
        record = '2026-11-18' if c in NEW else '2026-10-26'
        print(f'| {c} | {record} | {group(c, pc[c][0])} | {pc[c][1]} | {group(c, sl[c][0])} | {sl[c][1]} |')
    print()
    for name, table in (('PoisonCap', pc), ('Sublet', sl)):
        counts = {}
        for c in POP:
            counts.setdefault(group(c, table[c][0]), []).append(c)
        for g in ('rejected at site', 'in quarantine only', 'neither'):
            cases = counts.get(g, [])
            print(f'{name:9} {g:19} {len(cases):2} of {len(POP)}  {" ".join(cases)}')
    print()
    print(f'repetitions equal, PoisonCap: {pc == pc2}')
    print(f'repetitions equal, Sublet:    {sl == sl2}')
    for c, source in sorted(IN_QUARANTINE.items()):
        print(f'in quarantine, case {c}: {source}')


if __name__ == '__main__':
    main()
