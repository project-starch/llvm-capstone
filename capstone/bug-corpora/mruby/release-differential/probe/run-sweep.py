#!/usr/bin/env python3
"""Run every extracted case against an arm and record the verdict.

  run-sweep.py <arm-binary> <out.json> [--timeout N] [--only sha,sha]

A case is the harness plus the test lines its commit added. The harness prints
["PASS"] when nothing failed and one line per failure otherwise, so a verdict is
read off the output rather than the exit status, which mruby uses for other
things too.
"""
import json, subprocess, sys, os, re, concurrent.futures

W = '/tmp/capstone/mruby-corpus'
binary, out_path = sys.argv[1], sys.argv[2]
timeout = int(sys.argv[sys.argv.index('--timeout')+1]) if '--timeout' in sys.argv else 60
only = sys.argv[sys.argv.index('--only')+1].split(',') if '--only' in sys.argv else None

index = json.load(open(f'{W}/cases/index.json'))
if only:
    index = [e for e in index if e['name'] in only or e['sha'][:9] in only]

# The exception classes a test raises when the PIN simply lacks the feature the
# commit added, as opposed to answering wrongly or faulting.
FEATURE_GAP = ('NoMethodError', 'NameError', 'NotImplementedError', 'LocalJumpError')

def run(entry):
    case = f"{W}/cases/{entry['name']}.rb"
    try:
        p = subprocess.run([binary, case], capture_output=True, text=True, timeout=timeout)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return dict(entry, verdict='TIMEOUT', detail='')
    asan = re.search(r'AddressSanitizer: ([a-z-]+)', err + out)
    combined = err + out
    if rc < 0 or rc >= 128:
        sig = -rc if rc < 0 else rc - 128
        return dict(entry, verdict='CRASH', detail=f'signal {sig}',
                    asan=asan.group(1) if asan else None,
                    assertion=(re.search(r'Assertion [^\n]*', combined) or [None])[0]
                              if 'Assertion' in combined else None)
    if asan:
        return dict(entry, verdict='ASAN', detail=asan.group(1), asan=asan.group(1))
    if '["PASS"]' in out:
        return dict(entry, verdict='PASS', detail='')
    fails = re.findall(r'^\[(.*)\]$', out, re.M)
    if not fails:
        # Nothing parsed: the file did not even run (new syntax, or a parse error).
        first = (err.strip().splitlines() or [''])[0]
        return dict(entry, verdict='NORUN', detail=first[:200])
    kinds = re.findall(r'"(EXCEPTION|NOT_EQUAL|EQUAL|NOT_TRUE|NOT_FALSE|NOT_NIL|NIL|'
                       r'NOT_INCLUDE|NOT_KIND_OF|NO_RAISE|WRONG_RAISE|RAISED|FLUNK|'
                       r'NOT_PREDICATE|PREDICATE|NOT_OPERATOR)"', out)
    excs = re.findall(r'"EXCEPTION", "([A-Za-z:]+)"', out) + re.findall(r'"RAISED", "([A-Za-z:]+)"', out)
    gap = bool(excs) and all(e in FEATURE_GAP for e in excs) and \
          all(k in ('EXCEPTION', 'RAISED') for k in kinds)
    return dict(entry, verdict='FEATURE_GAP' if gap else 'FAIL',
                detail=f'{len(fails)} failures: ' + ','.join(sorted(set(kinds))),
                exceptions=sorted(set(excs)))

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(run, index))
json.dump(results, open(out_path, 'w'), indent=1)
from collections import Counter
print(binary)
for k, v in Counter(r['verdict'] for r in results).most_common():
    print(f'  {k:12} {v}')
