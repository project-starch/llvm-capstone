#!/usr/bin/env python3
"""Write a run plan for run-virtual-corpus.py from a corpus and its images.

    plan-virtual-corpus.py --corpus <corpus dir> --images <build dir>
                           --out <plan.json> [--arm-argv fixed --arm-argv buggy]
                           [--case-number] [--expect-symbol NAME]...
                           [--defect-marker TEXT] [--case-timeout S]

WHY A GENERATOR AND NOT A HAND-WRITTEN PLAN. Two conventions cover this tree's
C corpora, and both put the case number on the command line so a fixture told to
run another case refuses instead of silently running:

  * one arm per case, the number alone -- `--case-number` (postgres/c-repros);
  * a control arm and a defect arm over ONE binary, selected at run time --
    `--arm-argv fixed --arm-argv buggy` with `--case-number`, which is the
    plain-heap corpora's `<prog> fixed|buggy <n>`.

A third shape is the REPLAY corpora -- APR's pools and buckets, pymalloc,
memcached's slabs. Their hosted entry reads an event file and writes a report,
`<prog> <trace> <report> <mode>`, and the event file carries the case number so
a fixture handed another case's trace refuses it. `--trace-magic` writes that
file per case, in the same layout the physical runners pack, and points the row
at it. The report goes under /tmp because the staged gate disk is read-only.

The control arm is emitted FIRST for every case AND MARKED AS A CONTROL, because
rule 1 of the corpus contract is paired arms over one binary and a defect arm
whose control did not hold is not a reading. The runner then requires every such
row to complete, and keeps them out of the detection counts so a corpus of N
paired cases does not read as 2N measurements. Each arm becomes its own row,
named `<case>:<arm>`; `--no-control-arm` turns the marking off for a corpus
whose first argv word selects something other than a control.

`--expect-symbol` names the corpus's labelled probe. Give it only where the
probe is EXTERNAL: a `static` probe duplicated per translation unit cannot be
resolved from the image, and the two plain-heap siblings record exactly that
limitation on their CheriBSD arms. Where it is absent the run still records the
fault; it just does not claim attribution.
"""
import argparse
import json
from pathlib import Path
import struct
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm-argv', action='append', default=[],
                   help='One argv word selecting an arm, repeatable, control first')
    p.add_argument('--no-control-arm', action='store_true',
                   help='Do not mark the first --arm-argv row as a control')
    p.add_argument('--case-number', action='store_true',
                   help='Append the case number, which the fixture checks')
    p.add_argument('--expect-symbol', action='append', default=[])
    p.add_argument('--defect-marker')
    p.add_argument('--case-timeout', type=int, default=120)
    p.add_argument('--arm', default='virtual', help='The label this run records')
    p.add_argument('--trace-magic', help="The corpus's trace magic, e.g. 0x314C4F4F50525041")
    p.add_argument('--trace-words', type=int, default=16,
                   help='uint64 words in the trace header plus its one event')
    p.add_argument('--trace-case-index', type=int, default=13,
                   help='Which word carries the case number')
    p.add_argument('--trace-dir', type=Path, help='Where to write the trace files')
    p.add_argument('--trace-mode', default='0', help='The mode argument after the report path')
    p.add_argument('--trace-no-mode', action='store_true',
                   help="Omit the mode argument: the pymalloc entry accepts exactly "
                        "two paths outside its PoisonCap build and returns 2 for a third")
    p.add_argument('--repo', type=Path, help='Repository root, to record a relative corpus path')
    a = p.parse_args()

    if a.trace_magic and not a.trace_dir:
        p.error('--trace-magic needs --trace-dir')
    images = sorted(a.images.glob('[0-9][0-9]_*.dom'))
    if not images:
        sys.exit(f'no case images under {a.images}; build them first')
    corpus = a.corpus.resolve()
    cases = []
    for image in images:
        name = image.name[:-len('.dom')]
        number = str(int(name.split('_')[0]))
        for arm in (a.arm_argv or ['']):
            if a.trace_magic:
                words = [0] * a.trace_words
                words[0] = int(a.trace_magic, 0)
                words[1] = 1   # one event; a wrong count is the negative control
                words[a.trace_case_index] = int(number)
                a.trace_dir.mkdir(parents=True, exist_ok=True)
                trace = a.trace_dir / f'{name}.bin'
                trace.write_bytes(struct.pack(f'<{a.trace_words}Q', *words))
                argv = [f'/mnt/vm/traces/{name}.bin', f'/tmp/{name}.report']
                if not a.trace_no_mode:
                    argv.append(a.trace_mode)
            else:
                argv = ([arm] if arm else []) + ([number] if a.case_number else [])
            case = dict(tag=f'{name}:{arm}' if arm else name,
                        image=str(image), argv=argv)
            if (len(a.arm_argv) > 1 and arm == a.arm_argv[0]
                    and not a.no_control_arm):
                case['control'] = True
            if a.expect_symbol:
                case['expect_symbol'] = a.expect_symbol
            if a.defect_marker:
                case['defect_marker'] = a.defect_marker
            cases.append(case)
    try:
        where = str(corpus.relative_to(a.repo.resolve())) if a.repo else str(corpus)
    except ValueError:
        where = str(corpus)
    plan = dict(corpus=where, arm=a.arm, case_timeout=a.case_timeout, cases=cases)
    if a.trace_magic:
        plan['stage'] = {'traces': str(a.trace_dir)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    print(f'{len(cases)} rows over {len(images)} cases -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
