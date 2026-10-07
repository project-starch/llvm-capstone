#!/usr/bin/env python3
"""Turn a virtual run's result.json into a corpus result bundle.

    collect-virtual-results.py --run <work dir of run-virtual-corpus.py>
                               --stamp 20261007-virtual [--name <dir name>]

Writes `<corpus>/results/<stamp>/` with a `matrix.tsv`, an `inputs.json` and a
`README.md`, which is what rule 6 of the corpus contract asks for: a summary,
never the capture it came from. Serial logs stay in scratch -- they carry kernel
and driver banners, so scrubbing them is endless and per-log, and twelve result
lines are better evidence than a thousand log lines anyway.

The corpus is taken from the run's own record, so a bundle cannot be filed under
a corpus the run did not measure.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stamp', required=True)
    p.add_argument('--name', help='Result directory name; default <stamp>')
    a = p.parse_args()

    record = json.loads((a.run / 'result.json').read_text())
    corpus = REPO / record['corpus']
    if not corpus.is_dir():
        sys.exit(f"the run names a corpus that is not here: {record['corpus']}")
    out = corpus / 'results' / (a.name or a.stamp)
    out.mkdir(parents=True, exist_ok=True)

    lines = ['\t'.join(('case', 'arm', 'verdict', 'cause', 'attributed', 'evidence'))]
    for row in record['rows']:
        attributed = row.get('attributed')
        lines.append('\t'.join((
            row['tag'], 'control' if row['control'] else record['arm'], row['verdict'],
            f"{row['cause']} ({row['cause_name']})" if 'cause' in row else '--',
            {True: 'yes', False: 'no', None: '--'}[attributed]
            if attributed in (True, False, None) else '--',
            row['detail'].replace('\t', ' '))))
    (out / 'matrix.tsv').write_text('\n'.join(lines) + '\n')

    (out / 'inputs.json').write_text(json.dumps(dict(
        corpus=record['corpus'], arm=record['arm'], address_space='virtual',
        status=record['status'], cases=record['cases'], measured=record['measured'],
        controls=record['controls'], controls_held=record['controls_held'],
        verdicts=record['verdicts'], completed=record['completed'],
        platform_sha256=record['platform_sha256'], gate_sha256=record['gate_sha256'],
        staged_sha256=record.get('staged_sha256', {}),
        image_sha256={row['tag']: row['image_sha256'] for row in record['rows']}),
        indent=2) + '\n')

    counts = ', '.join(f'{n} {v}' for v, n in sorted(record['verdicts'].items()))
    (out / 'README.md').write_text(f"""# {record['corpus']} in the virtual address space

Arm `{record['arm']}`, run {a.stamp}. Status **{record['status']}**:
{record['measured']} of {record['cases']} cases measured, {record['controls_held']} of
{record['controls']} controls held. Verdicts: {counts}.

Every case is one Linux process under the virtual launcher `capstone-vexec`, so
a capability fault ends that process and the rest of the corpus keeps running --
the whole corpus is one boot. `matrix.tsv` has one line per row; `inputs.json`
carries the sha256 of every image, of the gate script, of each staged resource
and of the QEMU binary, launcher, module and kernel that ran them.

A verdict means:

| verdict | what it says |
|---|---|
| `detected` | a Capstone capability fault, causes 24-30, with the cause named |
| `trap` | the process stopped on something else -- an illegal instruction, an access or page fault. A stop, but not the capability mechanism answering |
| `silent` | the case ran its own pre-defect marker and completed |
| `control-failure` | the case refused its own setup; never a verdict about the defect |
| `harness` | the case did not run, or faulted before announcing itself. Not a measurement |
| `timeout` | killed at the per-case limit. Not a measurement |

`attributed` is filled only where the corpus's labelled probe is an external
symbol: the fault's pc is compared against that symbol's extent resolved from
the image that ran, shifted by the load address the launcher published. A `--`
means the corpus's probe is `static`, so no attribution is claimed.

Reproduce with `tools/build-virtual-*.py` and `tools/run-virtual-corpus.py`; the
lane document is `capstone/docs/plans/bug-corpora-virtual-address-space.md`.
""")
    print(f"{record['corpus']:<52} {record['status']:<5} {counts}  -> {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
