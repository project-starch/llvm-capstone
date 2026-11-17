#!/usr/bin/env python3
"""p1-check.py TRANSCRIPT STEPS.tsv [--ref RUNS.tsv.gz ...]: the P1 embedded-suite kit check, one transcript.

Exit 0 PASS, 1 FAIL (every problem listed), 2 NO DATA (no @@PERF-BEGIN at all, or an unreadable input).
Per step of STEPS.tsv, in order:
  - @@PERF-BEGIN and @@PERF-END with the step's tag, rc=0;
  - `BENCH-INFO <id> source=context-csr rdcycle_rdinstret=ok` and `counters=present`;
  - exactly W + I BENCH lines labelled w0.. and 0.., each with cycles > 0, instret > 0 and ns > 0, none VERIFY-FAILED;
  - `BENCH-DONE <id> failed=0`;
  - @@NATIVE-CNT <n> begin and end, both with run_cycles= and runs=, run_cycles(end) > run_cycles(begin).
Across steps: check= is one value per (program, iteration label) over both arms. With --ref (bitstream-3 runs-*.tsv.gz),
that value must equal the reference's for every (program, label) the reference has; a program the reference lacks is
listed as unreferenced. Console lines are taken after stripping CR; a heartbeat (@@HB...) cut into a line is removed,
and so is a kernel console line that starts mid-line (unsplice_printk; the count is printed first).
Test: test-p1-check.sh."""
import argparse, gzip, re, sys
from collections import defaultdict


# A kernel console line can land inside a user line on the board console (p1k1-3 on bitstream 3, 2026-11-17: 38/24/24
# times, mostly capstone_vm's release line inside the BENCH-DONE line of the process that just exited):
# "BENCH-DONE beebs/sglib-ha[  143.107178] capstone_vm: release retires ...\nshtable-libc failed=0". Each such fragment,
# a timestamped line starting mid-line, is removed with its newline, which rejoins the user line. This runs to a fixed
# point, because a second fragment right behind the first only becomes mid-line once the first is gone. A line whose
# text before the fragment is a known quote tag (native-smoke.sh's `@@NS-DMESG `, dmesg quoted on purpose) is kept.
# The count is printed: a parsing repair, never silent. The gate's checks then run on the rejoined text unchanged.
KSPLICE = re.compile(r'(?<=[^\n])\[\s*\d+\.\d{6}\] [^\n]*\n')
QUOTE_TAGS = ('@@NS-DMESG ',)


def unsplice_printk(t):
    n = 0
    while True:
        out, last, k = [], 0, 0
        for m in KSPLICE.finditer(t):
            ls = t.rfind('\n', 0, m.start()) + 1
            if t[ls:m.start()] in QUOTE_TAGS:
                continue
            out.append(t[last:m.start()]); last = m.end(); k += 1
        if not k:
            return t, n
        out.append(t[last:]); t = ''.join(out); n += k

ap = argparse.ArgumentParser()
ap.add_argument('transcript'); ap.add_argument('steps'); ap.add_argument('--ref', action='append', default=[])
a = ap.parse_args()
try:
    text = open(a.transcript, 'rb').read().decode('utf-8', 'replace')
    steps = [l.rstrip('\n').split('\t') for l in open(a.steps)][1:]
except OSError as e:
    print(f'NO DATA: {e}'); sys.exit(2)
text = re.sub(r'@@HB[^\n]*\n', '', text.replace('\r', ''))
text, nsplice = unsplice_printk(text)
print(f'kernel fragments removed from mid-line: {nsplice}')
lines = text.split('\n')
if not any(l.startswith('@@PERF-BEGIN ') for l in lines):
    print(f'NO DATA: no @@PERF-BEGIN in {a.transcript} ({len(lines)} lines)'); sys.exit(2)

BENCH = re.compile(r'BENCH (\S+) it=(\S+) cycles=(\d+) instret=(\d+) entries=(\d+) ns=(-?\d+) check=(0x[0-9a-f]+)(.*)$')
probs, checks, nbench = [], defaultdict(set), 0
pos = 0
for st in steps:
    n, prog, arm, proc, k, args = st[:6]
    w, i = (int(x) for x in args.split())
    tag = f'{n} {prog} {arm} {proc} K{k}'
    try:
        b = lines.index(f'@@PERF-BEGIN {tag}', pos)
    except ValueError:
        probs.append(f'step {n}: no @@PERF-BEGIN {tag} (after line {pos})'); continue
    e = next((j for j in range(b + 1, len(lines)) if lines[j].startswith('@@PERF-END ') or lines[j].startswith('@@PERF-BEGIN ')), None)
    if e is None or not lines[e].startswith(f'@@PERF-END {tag} '):
        probs.append(f'step {n}: no @@PERF-END {tag} before the next marker'); pos = b + 1; continue
    m = re.search(r' rc=(\d+)', lines[e])
    if not m or m.group(1) != '0':
        probs.append(f'step {n}: {lines[e]}')
    body = lines[b + 1:e]; pos = e + 1
    if not any(re.match(r'BENCH-INFO \S+ source=context-csr rdcycle_rdinstret=ok$', l) for l in body):
        probs.append(f'step {n}: no "source=context-csr rdcycle_rdinstret=ok"')
    if not any(re.match(r'BENCH-INFO \S+ counters=present ', l) for l in body):
        probs.append(f'step {n}: counters not present')
    rows = [BENCH.match(l) for l in body if l.startswith('BENCH ')]
    if None in rows:
        probs.append(f'step {n}: unparsable BENCH line')
    rows = [r for r in rows if r]
    labels = [r.group(2) for r in rows]
    want = [f'w{x}' for x in range(w)] + [str(x) for x in range(i)]
    if labels != want:
        probs.append(f'step {n}: BENCH labels {labels}, want {want}')
    for r in rows:
        nbench += 1
        cy, ins, ns = int(r.group(3)), int(r.group(4)), int(r.group(6))
        if cy <= 0 or ins <= 0 or ns <= 0:
            probs.append(f'step {n} it={r.group(2)}: cycles={cy} instret={ins} ns={ns} (each must be > 0)')
        if 'VERIFY-FAILED' in r.group(8):
            probs.append(f'step {n} it={r.group(2)}: VERIFY-FAILED')
        checks[(prog, r.group(2))].add((arm, r.group(7)))
    if not any(re.match(r'BENCH-DONE \S+ failed=0$', l) for l in body):
        probs.append(f'step {n}: no BENCH-DONE failed=0')
    cnt = {}
    for l in body:
        m = re.match(rf'@@NATIVE-CNT {n} (begin|end) (.*)$', l)
        if m:
            kv = dict(x.split('=', 1) for x in m.group(2).split() if '=' in x)
            if 'run_cycles' in kv and 'runs' in kv:
                cnt[m.group(1)] = int(kv['run_cycles'])
    if set(cnt) != {'begin', 'end'}:
        probs.append(f'step {n}: @@NATIVE-CNT begin/end missing or without run_cycles/runs')
    elif cnt['end'] <= cnt['begin']:
        probs.append(f'step {n}: run_cycles did not advance ({cnt["begin"]} -> {cnt["end"]})')

ref = defaultdict(set)
for path in a.ref:
    with gzip.open(path, 'rt') as f:
        hdr = f.readline().rstrip('\n').split('\t'); P, I, C = hdr.index('program'), hdr.index('it'), hdr.index('check')
        for l in f:
            c = l.rstrip('\n').split('\t'); ref[(c[P], c[I])].add(c[C])
unref = set()
for key, vals in sorted(checks.items()):
    v = {c for _, c in vals}
    if len(v) != 1:
        probs.append(f'{key[0]} it={key[1]}: check= differs across arms: {sorted(vals)}')
    elif a.ref:
        if key in ref:
            if v != ref[key]:
                probs.append(f'{key[0]} it={key[1]}: check={v.pop()} but bitstream 3 read {sorted(ref[key])}')
        else:
            unref.add(key[0])
progs = sorted({k[0] for k in checks})
print(f'steps {len(steps)}, BENCH lines {nbench}, programs {len(progs)}, '
      f'reference: {"none" if not a.ref else f"{len(progs) - len(unref)} matched, unreferenced {sorted(unref)}"}')
for p in probs:
    print('  FAIL', p)
print('PASS' if not probs else f'FAIL {len(probs)} problems')
sys.exit(1 if probs else 0)
