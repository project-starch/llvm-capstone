#!/usr/bin/env python3
"""Board transcript of a perf run (run-*.sh steps) -> the per-iteration TSV perf-analyze.py reads.

A process is the text between `@@PERF-BEGIN <n> <program> <arm> <proc> K<k>` and the matching `@@PERF-END <n> ...
rc=<rc> <bench line>`. Inside it, the harness prints `BENCH-INFO <suite>/<name> counters=present warmup=W
iterations=I`, then one `BENCH <suite>/<name> it=<label> cycles= instret= entries= ns= check=0x..[ VERIFY-FAILED]`
per iteration, then `BENCH-DONE <id> failed=<f>`.

The trailing dmesg quote on @@PERF-END (the module's per-process release line) is NOT used for the data: a process
that died before release would quote the previous process's line. It is only checked: a quote identical to the
previous process's is flagged.

Exit 1 when any process is missing its END, has the wrong number of BENCH lines, a VERIFY-FAILED, counters other
than present, a nonzero rc, or failed > 0 -- listed, and still written with rc carried, so perf-analyze refuses it.
"""
import argparse
import re
import sys
from pathlib import Path

BEGIN = re.compile(r'@@PERF-BEGIN (\d+) (\S+) (T[123]) (\d+) K(\d+)')
END = re.compile(r'@@PERF-END (\d+) (\S+) (T[123]) (\d+) K(\d+) rc=(\d+)(.*)')
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
                       rc=None)
            continue
        if cur is None:
            continue
        m = END.search(ln)
        if m and int(m[1]) == cur['n']:
            cur['rc'] = int(m[6])
            quote = m[7].strip()
            if quote and quote == last_quote:
                problems.append(f'step {cur["n"]}: release quote identical to the previous step ({quote[:60]})')
            last_quote = quote
            procs.append(cur)
            cur = None
            continue
        m = INFO.search(ln)
        if m:
            cur['info'] = m
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
    print(f'parse-board: {len(procs)} processes, {len(rows)} iteration rows -> {a.out}')
    if problems:
        print('parse-board: PROBLEMS:\n  ' + '\n  '.join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
