#!/usr/bin/env python3
"""cases.json for the FFmpeg plain-heap corpus on CheriBSD, revocation ON.

All four rows predict a CATCH, so the buggy arms are declared to FAULT and a
COMPLETION is a visible refutation rather than a quiet pass. Case 1 is the one
most likely to be refuted and that is written down BEFORE the run: its request
is 24 bytes, which is not a size-class boundary, so if malloc rounds it to 32 the
crossing at offset 24 stays inside the usable allocation and nothing can fault --
exactly the mechanism that refuted memcached/plain-heap-repros/00.

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
SIGPROT = 34

# case -> (probe symbol, bytes requested by the buggy arm, what it crosses)
CASES = {
    0: ('ffh_read_probe',     16,  '4 bytes past a 16-byte request'),
    1: ('ffh_read_probe_u8',  24,  '1 byte past a 24-byte request -- NOT a size class, may be absorbed'),
    2: ('ffh_read_probe',     16,  '4 bytes BELOW the base -- immune to the usable-size mechanism'),
    3: ('ffh_write_probe_u8', 512, '33 bytes past a 512-byte request'),
}

cases = []

# The request sizes this corpus actually uses, measured in the SAME boot. The
# memcached/wireshark calloc table is not transferable across size classes.
cases.append(dict(
    name='cap-bounds',
    program=str(BIN / 'cap-bounds'),
    args=['16', '24', '512', '2048'],
    timeout=300,
    expect='CAPBOUNDS DONE',
    exit=0,
))

for n, (sym, request, what) in CASES.items():
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
    # PREDICTED: SIGPROT. A completion REFUTES this row's prediction.
    cases.append(dict(
        name=f'ffh-{n:02d}-buggy',
        program=str(BIN / f'supervise-{sym}'),
        args=['./target', 'buggy', str(n)],
        inputs={'target': str(prog)},
        timeout=300,
        expect=f'SUPERVISE exit signalled={SIGPROT}',
        exit=128 + SIGPROT,
    ))

out = OUT
out.write_text(json.dumps(cases, indent=2) + '\n')
print(f'  wrote {out}: {len(cases)} cases')
for c in cases:
    exp = c.get('expect') or c.get('expect_regex')
    print(f"    {c['name']:20s} exit={c.get('exit'):<4} {exp}")
print('\n  PREDICTIONS, written before the run:')
for n, (sym, request, what) in CASES.items():
    print(f'    case {n}: CATCH -- {what}  (probe {sym})')
