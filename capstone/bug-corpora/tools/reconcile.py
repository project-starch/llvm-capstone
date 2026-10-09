#!/usr/bin/env python3
"""Reconcile this tree's three-arm study with another tree's case set.

WHY THIS EXISTS. The study restructured 22 group-corpora into one corpus per
application while another lane grew and measured several of the same groups. A
trial merge of the two reports seven `modify/delete` conflicts and reads as two
irreconcilable case sets. It is not: keyed on the BUG rather than on the case
DIRECTORY the two reconcile with nothing orphaned, because deleting one case
renumbers every directory after it inside its group.

So the key here is the upstream id plus the slug -- the directory name with its
leading ordinal stripped -- and the ordinal difference is reported separately as
a rename map.

WHAT IT DOES NOT DO. It does not map the other tree's mechanism-level arms onto
the study's three. That mapping is a judgement with one trap in it, and the trap
is recorded in RECONCILE.md rather than encoded here: the other tree's `spatial`
arm is BOUNDS-ONLY -- its own oracles say `free only marks the block free` --
while `capstone-sysalloc` revokes synchronously at free. The two cannot differ
on a case where nothing is freed and they differ systematically on one where
something is, so reading `spatial` across is valid for a spatial crossing and
wrong for a temporal one.

    tools/reconcile.py OTHER_BUG_CORPORA_DIR [--json out.json]

OTHER_BUG_CORPORA_DIR is a bug-corpora/ directory in the older layout, one
`case.json` per case directory under `<program>/<group>/`.
"""
import argparse, collections, glob, json, os, re, sys

ORDINAL = re.compile(r'^\d+_')


def bug_key(case_dir):
    """The bug, not the directory: drop the leading ordinal, keep id and slug."""
    return ORDINAL.sub('', str(case_dir))


def read_other(root):
    out = {}
    for path in glob.glob(os.path.join(root, '*', '*', '*', 'case.json')):
        program, group, case = os.path.relpath(path, root).split(os.sep)[:3]
        with open(path) as handle:
            declared = json.load(handle)
        arms = {name: value for name, value in (declared.get('arms') or {}).items()
                if isinstance(value, dict)}
        out[(program, group, bug_key(case))] = {
            'dir': case,
            'measured': sorted(n for n, v in arms.items() if v.get('status') == 'measured'),
        }
    return out


def read_study(protection_json):
    with open(protection_json) as handle:
        study = json.load(handle)
    out = {}
    for corpus in study['corpora']:
        for case in corpus['cases']:
            out[(corpus['program'], case['group'], bug_key(case['case']))] = {
                'dir': case['case'],
                'ignored': bool(case.get('ignored')),
                'verdicts': {arm: (case['arms'].get(arm) or {}).get('verdict')
                             for arm in study['arms']},
            }
    return study['arms'], out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('other', help="the other tree's bug-corpora directory")
    parser.add_argument('--protection', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'protection.json'))
    parser.add_argument('--json', dest='out')
    args = parser.parse_args()

    other = read_other(args.other)
    if not other:
        print(f"reconcile: no case.json under {args.other}", file=sys.stderr)
        return 2
    arms, study = read_study(args.protection)

    shared = sorted(set(other) & set(study))
    only_other = sorted(set(other) - set(study))
    only_study = sorted(set(study) - set(other))
    renamed = [k for k in shared if other[k]['dir'] != study[k]['dir']]

    report = {
        'arms': arms,
        'other_cases': len(other), 'study_cases': len(study),
        'same_bug': len(shared),
        'only_other': [{'program': p, 'group': g, 'bug': b, 'dir': other[(p, g, b)]['dir'],
                        'other_measured': other[(p, g, b)]['measured']}
                       for p, g, b in only_other],
        'only_study': [{'program': p, 'group': g, 'bug': b} for p, g, b in only_study],
        'renamed': [{'program': p, 'group': g, 'bug': b,
                     'other_dir': other[(p, g, b)]['dir'],
                     'study_dir': study[(p, g, b)]['dir']} for p, g, b in renamed],
    }
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(report, handle, indent=1)
            handle.write('\n')

    print(f"other {len(other)}  study {len(study)}  same bug {len(shared)}  "
          f"only-other {len(only_other)}  only-study {len(only_study)}  "
          f"renumbered {len(renamed)}")
    # A case the study cannot answer yet, grouped by what the other tree HAS
    # measured for it, because that is what decides whether a run is needed.
    buckets = collections.Counter()
    for key in only_other:
        buckets[(key[0] + '/' + key[1], tuple(other[key]['measured']))] += 1
    for (group, measured), count in sorted(buckets.items()):
        print(f"  {count:4}  {group:40} other measured: {', '.join(measured) or 'nothing'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
