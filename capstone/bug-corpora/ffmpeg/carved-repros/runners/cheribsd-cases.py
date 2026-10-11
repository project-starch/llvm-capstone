#!/usr/bin/env python3
"""cases.json for the FFmpeg carved corpus on CheriBSD purecap: every case, fixed then buggy, plus
the carve control.

    cheribsd-cases.py <bin dir> <out json> fault|complete

EVERY CASE DIRECTORY MUST HAVE A ROW, and the gate below enforces it (a stale table once measured
four cases of twenty-five and passed).

THE EXPECTATION IS PER ARM, not per case. Every crossing here stays inside its one allocation (the
driver exits 75 otherwise), so on an arm whose bounds are the allocation's -- revocation on or off,
field bounds -- the buggy run COMPLETES with VERDICT DEFECT-REPRODUCED, and on the carve-bounds arm,
where ffc_carve() narrows each region, it FAULTS at the labelled probe. The caller passes which, and
a row that reads otherwise is data for the record, not a runner failure.
"""
import json
import pathlib
import sys

BIN, OUT, EXPECT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
CORPUS = pathlib.Path(__file__).resolve().parent.parent
SIGPROT = 34
assert EXPECT in ("fault", "complete"), EXPECT

# case -> the probe its first crossing access goes through (each case.c names it)
PROBES = {0: 'ffc_write_probe_u8', 1: 'ffc_write_probe_u32', 2: 'ffc_read_probe_u32',
          3: 'ffc_read_probe_u32', 4: 'ffc_write_probe_u32', 5: 'ffc_write_probe_u32',
          6: 'ffc_write_probe_u8', 7: 'ffc_write_probe_u8', 8: 'ffc_write_probe_u32',
          9: 'ffc_write_probe_u8', 10: 'ffc_write_probe_u8', 11: 'ffc_read_probe_u32',
          12: 'ffc_read_probe_u32'}

present = {int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir()}
if present != set(PROBES):
    sys.exit(f'CONTROL-FAILED cheribsd-cases.py: corpus has {sorted(present)}, the table '
             f'{sorted(PROBES)}')
for n, d in ((n, next(CORPUS.glob(f'{n:02d}_*'))) for n in sorted(present)):
    if PROBES[n].replace('ffc_', '') not in (d / 'case.c').read_text():
        sys.exit(f'CONTROL-FAILED cheribsd-cases.py: case {n} does not call {PROBES[n]}')
for sym in set(PROBES.values()) | {'ffc_write_probe_u8'}:
    if not (BIN / f'supervise-{sym}').exists():
        sys.exit(f'CONTROL-FAILED cheribsd-cases.py: supervise-{sym} was not built')


def buggy(name, prog, sym, n):
    want = (f'SUPERVISE exit signalled={SIGPROT}', 128 + SIGPROT) if EXPECT == 'fault' \
        else ('SUPERVISE exit status=0', 0)
    return dict(name=name, program=str(BIN / f'supervise-{sym}'), args=['./target', 'buggy', str(n)],
                inputs={'target': str(prog)}, timeout=300, expect=want[0], exit=want[1])


cases = [dict(name='carve-control-fixed', program=str(BIN / 'carve-control'), args=['fixed', '99'],
              timeout=120, expect_regex=r'VERDICT FIXED .*', exit=0),
         buggy('carve-control-buggy', BIN / 'carve-control', 'ffc_write_probe_u8', 99)]
for n in sorted(present):
    prog = BIN / f'ffc-{n:02d}'
    cases.append(dict(name=f'ffc-{n:02d}-fixed', program=str(prog), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    cases.append(buggy(f'ffc-{n:02d}-buggy', prog, PROBES[n], n))
OUT.write_text(json.dumps(cases, indent=2) + '\n')
print(f'  cases.json: {len(cases)} arms, all {len(present)} case directories plus the carve control; '
      f'buggy arms predicted to {EXPECT.upper()}')
