#!/usr/bin/env python3
"""Every PHP corpus verdict of run-6 (the fixed nested layer) against run-4, case by case.

    compare-corpora.py RUN4_DIR RUN6_CORPORA_DIR [OUT.json]

RUN4_DIR is run-4/ on p13 (php-engine, php-engine-tav, php-plain-temporal, php-zend-alloc under it);
RUN6_CORPORA_DIR is run-6/corpora/. For every verdicts.jsonl under a run-4 corpus directory, the
run-6 file at the same relative path is read, the CAUGHT tallies are printed for both, and every
case whose verdict or reason changed is listed with both evidences.
"""
import json
import sys
from pathlib import Path

CORPORA = ('php-engine', 'php-engine-tav', 'php-plain-temporal', 'php-zend-alloc')


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['observation']['case']] = r
    return rows


def main():
    run4, run6 = Path(sys.argv[1]), Path(sys.argv[2])
    record = {}
    for corpus in CORPORA:
        for old_file in sorted((run4 / corpus).rglob('verdicts.jsonl')):
            rel = old_file.relative_to(run4 / corpus)
            new_file = run6 / corpus / rel
            key = f'{corpus}/{rel.parent}'
            if not new_file.is_file():
                print(f'{key}: NO RUN-6 FILE')
                record[key] = {'missing': True}
                continue
            old, new = load(old_file), load(new_file)
            tally = lambda rows: sum(1 for r in rows.values() if r['verdict'] == 'CAUGHT')
            changed = []
            for case in sorted(set(old) | set(new)):
                a, b = old.get(case), new.get(case)
                va = (a['verdict'], a.get('reason')) if a else None
                vb = (b['verdict'], b.get('reason')) if b else None
                if va != vb:
                    changed.append({'case': case, 'run4': va, 'run6': vb,
                                    'run4_evidence': a and a.get('evidence'),
                                    'run6_evidence': b and b.get('evidence')})
            record[key] = {'caught_run4': tally(old), 'caught_run6': tally(new), 'cases': len(new),
                           'changed': changed}
            print(f'{key}: CAUGHT run-4 {tally(old)} run-6 {tally(new)} of {len(new)}; '
                  f'{len(changed)} changed')
            for c in changed:
                print(f"    {c['case']}: {c['run4']} -> {c['run6']}")
                print(f"      run-4: {(c['run4_evidence'] or '')[:220]}")
                print(f"      run-6: {(c['run6_evidence'] or '')[:220]}")
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(record, indent=1) + '\n')


if __name__ == '__main__':
    main()
