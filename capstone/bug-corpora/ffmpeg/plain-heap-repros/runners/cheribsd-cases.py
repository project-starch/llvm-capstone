#!/usr/bin/env python3
"""cases.json for the FFmpeg plain-heap corpus on CheriBSD, revocation ON.

THE TABLE BELOW MUST COVER EVERY CASE DIRECTORY, AND A GATE AT THE BOTTOM ENFORCES
THAT. It is the reason this file was rewritten on 2026-10-08: `CASES` listed cases
0-3 while the corpus had grown to 25, so a run measured four cases and reported
nothing at all about the other twenty-one. The suite PASSED, the controls fired,
and the missing rows were invisible -- the shape this tree calls a gate that goes
stale silently. Adding a case without adding a row here now fails the run.

EACH ROW CARRIES ITS OWN EXPECTATION, because a crossing shorter than the gap to
the next size class CANNOT fault: malloc bounds a capability to the allocator's
USABLE size, so the access stays inside the same allocation. Declaring FAULT for
such a row makes a correct reading look like a failure, which is how a measured
absorption gets mistaken for a bug. Cases 1 and 4 are those rows, and both are
declared COMPLETE here from `tools/size-class-audit.py`, which reproduces four
in-guest usable sizes recorded in memcached/plain-heap-repros/00's case.json.

A BELOW-BASE crossing is immune to that mechanism entirely -- there is no slack
before the base -- so case 2 and case 13 are FAULT regardless of their request.

ATTRIBUTION is available here, unlike the sibling plain-heap corpora: the probes
are a single external definition in shared/driver.c, so `supervise` can resolve
the symbol from the image and SCHEMA rule 2 -- the fault AT the labelled probe --
can be met. One supervise per probe symbol, because PROBE_SYMBOL is compile-time.
"""
import json
import pathlib
import sys

BIN = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
CORPUS = pathlib.Path(__file__).resolve().parent.parent
SIGPROT = 34

# case -> (probe symbol, bytes the buggy arm requests, what it crosses, expectation)
# Request sizes and crossings are MEASURED by tools/size-class-audit.py, which
# interposes on malloc/calloc at run time rather than reading them off the source.
CASES = {
    0:  ('ffh_read_probe',     16,  '4 bytes past a 16-byte request',                     'FAULT'),
    1:  ('ffh_read_probe_u8',  24,  '4 bytes past a 24-byte request; usable 32, so 8 '
                                    'bytes of slack ABSORB it',                           'COMPLETE'),
    2:  ('ffh_read_probe',     16,  '4 bytes BELOW the base -- immune to the usable-size '
                                    'mechanism, there is no slack before the base',       'FAULT'),
    3:  ('ffh_write_probe_u8', 512, '33 bytes past a 512-byte request',                   'FAULT'),
    4:  ('ffh_write_probe_u8', 17,  '1 byte past a 17-byte request; usable 32, so 15 '
                                    'bytes of slack ABSORB it',                           'COMPLETE'),
    5:  ('ffh_read_probe_u8',  24,  'the probe lands 4 bytes past a 24-byte request; usable 32, '
                                    'so 8 bytes of slack ABSORB it. The UNREDUCED defect runs 64 '
                                    'bytes past, but the reduction touches only the first element '
                                    'and that is what the capability check sees -- MEASURED as a '
                                    'COMPLETION on 2026-10-08, refuting this row s earlier FAULT',   'COMPLETE'),
    6:  ('ffh_read_probe_u8',  48,  '20 bytes past a 48-byte request',                    'FAULT'),
    7:  ('ffh_write_probe_u8', 32,  '16 bytes past a 32-byte request',                    'FAULT'),
    8:  ('ffh_write_probe_u8', 160, '64 bytes past a 160-byte request',                   'FAULT'),
    9:  ('ffh_write_probe_u8', 64,  '16 bytes past a 64-byte request',                    'FAULT'),
    10: ('ffh_write_probe_u8', 64,  '1026 bytes past a 64-byte request',                  'FAULT'),
    11: ('ffh_read_probe_u8',  16,  '1 byte past a 16-byte request',                      'FAULT'),
    12: ('ffh_write_probe_u8', 16,  '1 byte past a 16-byte request',                      'FAULT'),
    13: ('ffh_read_probe',     16,  '16 bytes BELOW the base -- the single-reflection '
                                    'mirror returns a negative index',                    'FAULT'),
    14: ('ffh_read_probe_u8',  16,  '586 bytes past a 16-byte request',                   'FAULT'),
    15: ('ffh_read_probe_u8',  48,  '48 bytes past a 48-byte request',                    'FAULT'),
    16: ('ffh_read_probe_u8',  32,  '16 bytes past a 32-byte request',                    'FAULT'),
    17: ('ffh_write_probe_u8', 16,  '16 bytes past a 16-byte request',                    'FAULT'),
    18: ('ffh_read_probe_u8',  32,  '8 bytes past a 32-byte request',                     'FAULT'),
    19: ('ffh_write_probe_u8', 32,  '8 bytes past a 32-byte request',                     'FAULT'),
    20: ('ffh_write_probe_u8', 16,  '48 bytes past a 16-byte request',                    'FAULT'),
    21: ('ffh_read_probe',     16,  '8 bytes past a 16-byte request',                     'FAULT'),
    22: ('ffh_write_probe_u8', 512, '372 bytes past a 512-byte request',                   'FAULT'),
    23: ('ffh_write_probe_u8', 16,  '24 bytes past a 16-byte request',                    'FAULT'),
    24: ('ffh_read_probe_u8',  16,  '48 bytes past a 16-byte request',                    'FAULT'),
}

# --- THE GATE. A case directory with no row here is a case this run would have
# been silent about, so it is a hard failure and not a warning.
present = {int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir()}
missing, extra = sorted(present - set(CASES)), sorted(set(CASES) - present)
if missing or extra:
    msg = []
    if missing:
        msg.append(f'cases {missing} exist in the corpus but have no row in CASES -- '
                   f'the run would measure nothing about them')
    if extra:
        msg.append(f'CASES names {extra}, which is not a case directory')
    sys.exit('CONTROL-FAILED cheribsd-cases.py: ' + '; '.join(msg))

# Every probe symbol a row names must have been built, or supervise cannot resolve it.
for n, (sym, _, _, _) in sorted(CASES.items()):
    if not (BIN / f'supervise-{sym}').exists():
        sys.exit(f'CONTROL-FAILED cheribsd-cases.py: case {n} names probe {sym}, but '
                 f'{BIN}/supervise-{sym} was not built -- add it to run-cheribsd.sh')

cases = []

# The request sizes this corpus actually uses, measured in the SAME boot. The
# memcached/wireshark calloc table is not transferable across size classes.
cases.append(dict(
    name='cap-bounds',
    program=str(BIN / 'cap-bounds'),
    args=['16', '17', '24', '32', '48', '64', '160', '512'],
    timeout=300,
    expect='CAPBOUNDS DONE',
    exit=0,
))

for n, (sym, request, what, expect) in sorted(CASES.items()):
    prog = BIN / f'ffh-{n:02d}'
    # Control arm first: the upstream fix is in, nothing crosses, exit 0.
    cases.append(dict(
        name=f'ffh-{n:02d}-fixed',
        program=str(prog),
        args=['fixed', str(n)],
        timeout=300,
        expect_regex=r'VERDICT FIXED .*',
        exit=0,
    ))
    # Buggy arm, supervised so the fault is seen from outside the process.
    if expect == 'FAULT':
        cases.append(dict(
            name=f'ffh-{n:02d}-buggy',
            program=str(BIN / f'supervise-{sym}'),
            args=['./target', 'buggy', str(n)],
            inputs={'target': str(prog)},
            timeout=300,
            expect=f'SUPERVISE exit signalled={SIGPROT}',
            exit=128 + SIGPROT,
        ))
    else:
        # The crossing cannot leave the usable allocation, so a COMPLETION is the
        # correct reading and a fault here would be the surprise.
        cases.append(dict(
            name=f'ffh-{n:02d}-buggy',
            program=str(BIN / f'supervise-{sym}'),
            args=['./target', 'buggy', str(n)],
            inputs={'target': str(prog)},
            timeout=300,
            expect='SUPERVISE exit status=0',
            exit=0,
        ))

OUT.write_text(json.dumps(cases, indent=2) + '\n')
print(f'  wrote {OUT}: {len(cases)} cases, covering all {len(present)} case directories')
print('\n  PREDICTIONS, written before the run:')
for n, (sym, request, what, expect) in sorted(CASES.items()):
    verb = 'CATCH   ' if expect == 'FAULT' else 'COMPLETE'
    print(f'    case {n:>2}: {verb} -- {what}  (probe {sym})')
