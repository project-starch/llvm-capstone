#!/usr/bin/env python3
"""Board transcript of a perf run (run-*.sh steps) -> the per-iteration TSV perf-analyze.py reads.

A process is the text between `@@PERF-BEGIN <n> <program> <arm> <proc> K<k>` and the matching `@@PERF-END <n> ...
rc=<rc> <bench line>`. Inside it, the harness prints `BENCH-INFO <suite>/<name> counters=present warmup=W
iterations=I`, then one `BENCH <suite>/<name> it=<label> cycles= instret= entries= ns= check=0x..[ VERIFY-FAILED]`
per iteration, then `BENCH-DONE <id> failed=<f>`.

The trailing dmesg quote on @@PERF-END (the module's per-process release line) is NOT used for the data: a process
that died before release would quote the previous process's line. It is only checked: a quote identical to the
previous process's is flagged.

A Plain-arm process (a plain RISC-V executable, user-mode counters) also prints one `BENCH-KERN <id> it=<label>
timer= irq= minflt= majflt= nvcsw= nivcsw=` line per iteration; those go to <out>.kern.tsv (one row per iteration,
keyed like the main rows), and a Plain process without one per BENCH line is a problem. Its END line has no dmesg
quote.

Exit 1 when any process is missing its END, has the wrong number of BENCH lines, a VERIFY-FAILED, counters other
than present, a nonzero rc, or failed > 0 -- listed, and still written with rc carried, so perf-analyze refuses it.
"""
import argparse
import re
import sys
from pathlib import Path

ARM = r'(T[123]|Plain)'
BEGIN = re.compile(r'@@PERF-BEGIN (\d+) (\S+) ' + ARM + r' (\d+) K(\d+)')
END = re.compile(r'@@PERF-END (\d+) (\S+) ' + ARM + r' (\d+) K(\d+) rc=(\d+)(.*)')
KERN = re.compile(r'BENCH-KERN (\S+)/(\S+) it=(w?\d+) timer=(\d+) irq=(\d+) minflt=(-?\d+) majflt=(-?\d+) '
                  r'nvcsw=(-?\d+) nivcsw=(-?\d+)')
INFO = re.compile(r'BENCH-INFO (\S+)/(\S+) counters=(\S+) warmup=(\d+) iterations=(\d+)')
BENCH = re.compile(r'BENCH (\S+)/(\S+) it=(w?\d+) cycles=(\d+) instret=(\d+) entries=(\d+) ns=(-?\d+) '
                   r'check=(0x[0-9a-f]+)( VERIFY-FAILED)?')
DONE = re.compile(r'BENCH-DONE (\S+) failed=(\d+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('transcript', type=Path)
    ap.add_argument('--board', required=True)
    ap.add_argument('--boot', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    text = a.transcript.read_text(errors='replace')
    # The driver's heartbeat (`@@HB <i> out=..B/..L tail=<base64>`) is its own line, but it can land in the middle of
    # a guest line ("ns=16960066@@HB 129 ...\n40 check=..."). Remove every heartbeat with its newline, which rejoins
    # the guest line, and count the mid-line ones: a repair, never silent (as checks-diff.py and perfmain-table.py).
    hb = re.compile(r'@@HB \d+ out=\S* tail=[A-Za-z0-9+/=]*\n')
    mid = sum(1 for m in hb.finditer(text) if m.start() > 0 and text[m.start() - 1] != '\n')
    text = hb.sub('', text)
    if mid:
        print(f'parse-board: rejoined {mid} guest line(s) split by a mid-line @@HB', file=sys.stderr)
    lines = text.splitlines()
    procs, cur, problems, last_quote = [], None, [], None
    for ln in lines:
        m = BEGIN.search(ln)
        if m:
            if cur:
                problems.append(f'step {cur["n"]}: no @@PERF-END before the next BEGIN')
                procs.append(cur)
            cur = dict(n=int(m[1]), program=m[2], arm=m[3], proc=m[4], K=m[5], info=None, bench=[], done=None,
                       rc=None, kern=[])
            continue
        if cur is None:
            continue
        m = END.search(ln)
        if m and int(m[1]) == cur['n']:
            cur['rc'] = int(m[6])
            quote = m[7].strip()
            if cur['arm'] != 'Plain':   # a Plain step has no module release line to quote
                if quote and quote == last_quote:
                    problems.append(f'step {cur["n"]}: release quote identical to the previous step ({quote[:60]})')
                last_quote = quote
            procs.append(cur)
            cur = None
            continue
        m = INFO.search(ln)
        if m:
            cur['info'] = m
        m = KERN.search(ln)
        if m:
            cur['kern'].append(m)
            continue
        m = BENCH.search(ln)
        if m:
            cur['bench'].append(m)
        m = DONE.search(ln)
        if m:
            cur['done'] = int(m[2])
    if cur:
        problems.append(f'step {cur["n"]}: no @@PERF-END (transcript ends)')
        procs.append(cur)
    if not procs:
        print(f'parse-board: no @@PERF steps in {a.transcript}', file=sys.stderr)
        sys.exit(1)
    rows = []
    for p in procs:
        rc = p['rc'] if p['rc'] is not None else 255
        info = p['info']
        if info is None:
            problems.append(f'step {p["n"]} {p["program"]} {p["arm"]}: no BENCH-INFO')
            want = None
        else:
            if info[3] != 'present':
                problems.append(f'step {p["n"]}: counters={info[3]}')
            want = int(info[4]) + int(info[5])
        if want is not None and len(p['bench']) != want:
            problems.append(f'step {p["n"]} {p["program"]} {p["arm"]}: {len(p["bench"])} BENCH lines, want {want}')
        if p['arm'] == 'Plain' and [k[3] for k in p['kern']] != [b[3] for b in p['bench']]:
            problems.append(f'step {p["n"]} {p["program"]} Plain: {len(p["kern"])} BENCH-KERN lines for '
                            f'{len(p["bench"])} BENCH lines')
        if any(b[9] for b in p['bench']):
            problems.append(f'step {p["n"]} {p["program"]} {p["arm"]}: VERIFY-FAILED')
        if p['done'] is None or p['done'] != 0:
            problems.append(f'step {p["n"]} {p["program"]} {p["arm"]}: BENCH-DONE failed={p["done"]}')
        if rc != 0:
            problems.append(f'step {p["n"]} {p["program"]} {p["arm"]}: rc={rc}')
        bad = 0 if rc == 0 and p['done'] == 0 and not any(b[9] for b in p['bench']) else (rc or 1)
        for b in p['bench']:
            rows.append([a.board, a.boot, b[1], p['program'], p['arm'], f'{p["proc"]}', b[3], b[4], b[5], b[7], str(bad),
                         b[6], p['K'], b[8], str(p['n'])])
    with a.out.open('w') as f:
        f.write('board\tboot\tsuite\tprogram\tarm\tproc\tit\tcycles\tinstret\tns\trc\tentries\tK\tcheck\tstep\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    kern = [[a.board, a.boot, k[1], p['program'], p['arm'], p['proc'], k[3], *k.groups()[3:], str(p['n'])]
            for p in procs for k in p['kern']]
    if kern:
        kout = a.out.with_name(a.out.name + '.kern.tsv')
        with kout.open('w') as f:
            f.write('board\tboot\tsuite\tprogram\tarm\tproc\tit\ttimer\tirq\tminflt\tmajflt\tnvcsw\tnivcsw\tstep\n')
            for r in kern:
                f.write('\t'.join(map(str, r)) + '\n')
        print(f'parse-board: {len(kern)} BENCH-KERN rows -> {kout}')
    print(f'parse-board: {len(procs)} processes, {len(rows)} iteration rows -> {a.out}')
    if problems:
        print('parse-board: PROBLEMS:\n  ' + '\n  '.join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
