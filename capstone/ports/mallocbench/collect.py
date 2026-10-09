#!/usr/bin/env python3
"""Collect and check the mimalloc-bench runs of all three systems; write one results.json.

    collect.py --capstone DIR... --native DIR --out results.json

--capstone  run directories of remote-probe.sh (each holds work-<name>/serial.log); untraced
            runs of the same program in several directories are repetitions
--native    the results directory of native/run-native.sh (<name>.txt stdout, <name>.err stderr)

Every run is checked as the README requires before anything is plotted:
  exit      exit status 0
  stream    the console stream is complete: the lines between MB_BEGIN and MB_END equal
            MB_OUT_LINES (Capstone), and a final record exists (MQ-DONE, CAPSTONE_VM_STATS)
  output    the program's own output equals the native run's (sh6bench and barnes: without
            the lines that report times)
  ops       a traced run counted the native run's allocations
A failed check is printed and recorded; nothing is silently dropped. A run without MB_END is
reported as still running and not checked. Exit status 1 if a finished run fails a check, 2 if
an expected input is missing or a run is still running.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROGRAMS = ['barnes', 'mleak5', 'mleak50', 'mstress', 'espresso', 'cfrac', 'glibc-simple', 'sh6bench']
RESULT = re.compile(r'^(MB_RUSAGE|CAPSTONE_VM_STATS|CAPSTONE_VM_SAMPLE|MQ)')
TIMES = {'sh6bench': re.compile(r'elapsed time|CPU|seconds', re.I),
         'barnes': re.compile(r'^(COMPUTESTART|COMPUTEEND|COMPUTETIME|TRACKTIME|PARTITIONTIME|'
                              r'TREEBUILDTIME|FORCECALCTIME|RESTTIME)\b')}


def kv(line):
    return {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}


def hist(line):
    return {int(b): int(c) for b, c in re.findall(r'(\d+):(\d+)', line.split(None, 1)[1])} if ' ' in line else {}


def tracer(lines):
    """MQ records of one run: per-sample rows, the last histograms, the final record."""
    out = {'rows': [], 'hist': None, 'life': None, 'sizes': None, 'stride': None, 'done': None}
    for l in lines:
        if l.startswith('MQ op='):
            out['rows'].append(kv(l))
        elif l.startswith('MQ-HIST'):
            out['hist'] = hist(l)
        elif l.startswith('MQ-LIFE'):
            out['life'] = hist(l)
        elif l.startswith('MQ-SIZES'):
            out['sizes'] = hist(l)
        elif l.startswith('MQ-STRIDE'):
            out['stride'] = hist(l)
        elif l.startswith('MQ-DONE'):
            out['done'] = kv(l)
    return out


def program_output(lines, prog):
    keep = [l for l in lines if not RESULT.match(l) and l.strip()]
    if prog in TIMES:
        keep = [l for l in keep if not TIMES[prog].search(l)]
    return keep


def capstone_run(log):
    text = log.read_text(errors='replace').replace('\r', '')
    name = log.parent.name[len('work-'):]
    lines = text.split('\n')
    try:
        b = next(i for i, l in enumerate(lines) if l.startswith('MB_BEGIN:'))
    except StopIteration:
        return {'name': name, 'path': str(log), 'error': 'no MB_BEGIN'}
    e = next((i for i, l in enumerate(lines) if l.startswith('MB_END:')), None)
    body = lines[b + 1:e] if e else lines[b + 1:]
    # The gate prints one empty line after the stream, before MB_END.
    while body and body[-1] == '':
        body.pop()
    run = {'name': name, 'path': str(log), 'rc': int(lines[e].split(':')[-1]) if e else None}
    m = re.search(r'^MB_OUT_LINES:\S+:(\d+)', text, re.M)
    run['out_lines'] = int(m.group(1)) if m else None
    run['streamed_lines'] = len(body)
    m = re.search(r'^MB_OUT_SHA256:\S+:(\w+)', text, re.M)
    run['out_sha256'] = m.group(1) if m else None
    run['samples'] = [kv(l) for l in body if l.startswith('CAPSTONE_VM_SAMPLE')]
    stats = [kv(l) for l in body if l.startswith('CAPSTONE_VM_STATS')]
    run['stats'] = stats[-1] if stats else None
    rus = [kv(l) for l in body if l.startswith('MB_RUSAGE')]
    run['rusage'] = rus[-1] if rus else None
    run['tracer'] = tracer(body)
    run['output'] = body
    return run


def native_run(d, name):
    out, err = d / f'{name}.txt', d / f'{name}.err'
    if not out.exists():
        return None
    errl = err.read_text(errors='replace').split('\n') if err.exists() else []
    rc = next((int(l[3:]) for l in reversed(errl) if l.startswith('rc=')), None)
    m = next((re.search(r'(\d+)', l) for l in errl if 'Maximum resident set size' in l), None)
    return {'name': name, 'path': str(out), 'rc': rc, 'maxrss_kib': int(m.group(1)) if m else None,
            'tracer': tracer(errl), 'output': out.read_text(errors='replace').split('\n')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capstone', nargs='+', type=Path, required=True)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    data = {'programs': PROGRAMS, 'capstone': {}, 'native': {}, 'checks': []}
    missing = 0
    running = []

    def check(system, prog, run, what, ok, detail=''):
        data['checks'].append({'system': system, 'program': prog, 'run': run, 'check': what,
                               'ok': bool(ok), 'detail': detail})
        if not ok:
            print(f'FAIL {system:8} {prog:13} {run:45} {what:7} {detail}')

    for prog in PROGRAMS:
        nat = {v: native_run(a.native, prog + v) for v in ('', '-traced')}
        data['native'][prog] = nat
        ref_out = program_output(nat[''] ['output'], prog) if nat[''] else None
        ref_ops = (nat['-traced'] or {}).get('tracer', {}).get('done', {}) or {}
        ref_ops = ref_ops.get('ops')
        for v, r in nat.items():
            if not r:
                missing += 1
                print(f'MISSING native {prog}{v}')
                continue
            check('native', prog, r['path'], 'exit', r['rc'] == 0, f'rc={r["rc"]}')
        runs = {'plain': [], 'traced': []}
        for d in a.capstone:
            for v, kind in (('', 'plain'), ('-traced', 'traced')):
                log = d / f'work-{prog}{v}' / 'serial.log'
                if log.exists():
                    runs[kind].append(capstone_run(log))
        data['capstone'][prog] = runs
        for kind, rs in runs.items():
            if not rs:
                missing += 1
                print(f'MISSING capstone {prog} {kind}')
            for r in rs:
                rid = str(Path(r['path']).parent.parent.name)
                if 'error' in r:
                    check('capstone', prog, rid, 'stream', False, r['error'])
                    continue
                if r['rc'] is None:
                    print(f'RUNNING  capstone {prog:13} {rid}')
                    running.append(f'{prog} {rid}')
                    continue
                check('capstone', prog, rid, 'exit', r['rc'] == 0, f'rc={r["rc"]}')
                final = r['tracer']['done'] if kind == 'traced' else r['stats']
                check('capstone', prog, rid, 'stream',
                      r['out_lines'] is not None and r['out_lines'] == r['streamed_lines'] and final,
                      f'streamed={r["streamed_lines"]} written={r["out_lines"]} final={bool(final)}')
                if ref_out is not None:
                    got = program_output(r['output'], prog)
                    first = next((i for i, (x, y) in enumerate(zip(ref_out, got)) if x != y),
                                 min(len(ref_out), len(got)))
                    check('capstone', prog, rid, 'output', got == ref_out,
                          '' if got == ref_out else
                          f'native {len(ref_out)} lines, capstone {len(got)}; first difference at '
                          f'line {first}: {(got[first] if first < len(got) else "<end>")[:60]!r}')
                if kind == 'traced' and ref_ops is not None:
                    ops = (r['tracer']['done'] or {}).get('ops')
                    check('capstone', prog, rid, 'ops', ops == ref_ops, f'capstone {ops} native {ref_ops}')
    for runs in data['capstone'].values():
        for rs in runs.values():
            for r in rs:
                r.pop('output', None)
    for nat in data['native'].values():
        for r in nat.values():
            if r:
                r.pop('output', None)
    a.out.write_text(json.dumps(data))
    bad = [c for c in data['checks'] if not c['ok']]
    data['running'] = running
    a.out.write_text(json.dumps(data))
    print(f'{len(data["checks"])} checks, {len(bad)} failed, {len(running)} runs still running, '
          f'{missing} inputs missing -> {a.out}')
    sys.exit(2 if missing or running else 1 if bad else 0)


if __name__ == '__main__':
    main()
