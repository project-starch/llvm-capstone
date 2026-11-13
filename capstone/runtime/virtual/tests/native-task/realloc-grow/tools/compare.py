#!/usr/bin/env python3
"""compare.py BENCH...: arm A (nat-fair-cache) against arm B (patch 0011) for out/c-{A,B}-<bench>: per run, summed over
its command windows, kernel (incl. the 0x80 bucket: kernel text at its physical address, plus OpenSBI ~0.01 M) and
user instructions (capmix3 shift 24), pages allocated (pgalloc_dma32) and faults over the commands, escapes (k1, k3,
k4), validation, and whether the two arms' result files are byte-identical. Missing data is an ERROR line, never 0."""
import sys, re, hashlib
from pathlib import Path
O = Path(__file__).resolve().parent / 'out'
NC = 14
def windows(path):
    snaps, cur = [], None
    for line in open(path):
        w = line.split()
        if not w: continue
        if w[0] in ('SNAP', 'FINAL'): cur = {'tag': w[0], 'label': int(w[2]), 'b': {}}
        elif w[0] == 'B': cur['b'][int(w[1], 16)] = sum(map(int, w[2:2 + NC]))
        elif w[0] == 'END': snaps.append(cur)
    s = [x for x in snaps if x['tag'] == 'SNAP']
    assert len(s) % 2 == 0 and all((s[i]['label'], s[i + 1]['label']) == (1, 2) for i in range(0, len(s), 2)), path
    k = u = 0
    for i in range(0, len(s), 2):
        a, b = s[i], s[i + 1]
        for key, v in b['b'].items():
            d = v - a['b'].get(key, 0); va = (key << 24) & ((1 << 64) - 1)
            if va >= 0xffff000000000000 or 0x80000000 <= va < 0x81000000: k += d
            else: u += d
    return k, u, len(s) // 2
def run(arm, b):
    d = O / f'c-{arm}-{b}'
    t = (d / 'vm/serial.log').read_text(errors='replace').replace('\r', '')
    k, u, nw = windows(d / 'capmix.txt')
    vm = {}
    for tag, n, v in re.findall(r'^@@VM(\d) (\d+) vm=(\S+)', t, re.M):
        vm[(tag, n)] = {x.split(':')[0]: int(x.split(':')[1]) for x in v.strip(',').split(',')}
    pages = sum(vm[('1', n)]['pgalloc_dma32'] - vm[('0', n)]['pgalloc_dma32'] for (tg, n) in vm if tg == '0')
    faults = sum(vm[('1', n)]['pgfault'] - vm[('0', n)]['pgfault'] for (tg, n) in vm if tg == '0')
    cnt = {}
    for n, w, v in re.findall(r'^@@NATIVE-CNT (\d+) (begin|end) (.*)$', t, re.M):
        cnt[(n, w)] = dict(x.split('=') for x in v.split() if '=' in x)
    esc = {kk: sum(int(cnt[(n, 'end')][kk]) - int(cnt[(n, 'begin')][kk]) for (n, w) in cnt if w == 'begin') for kk in ('k1', 'k3', 'k4')}
    val = (O / f'c-{arm}-{b}.validate').read_text().strip().splitlines()[-1] if (O / f'c-{arm}-{b}.validate').exists() else 'ERROR no validate'
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (d / 'results').iterdir() if p.is_file() and p.name != 'control'} if (d / 'results').is_dir() else {}
    return dict(k=k, u=u, nw=nw, pages=pages, faults=faults, esc=esc, val=val, files=files, nvm=len(vm) // 2)
for b in sys.argv[1:]:
    try: A, B = run('A', b), run('B', b)
    except Exception as e: print(f'ERROR {b}: {e!r}'); continue
    same = A['files'] == B['files'] and A['files']
    diff = sorted(f for f in set(A['files']) | set(B['files']) if A['files'].get(f) != B['files'].get(f))
    print(f'{b}: windows {A["nw"]}/{B["nw"]}  outputs {"IDENTICAL (%d files)" % len(A["files"]) if same else "DIFFER " + str(diff[:6])}')
    for key, lab in (('k', 'kernel'), ('u', 'user'), ('pages', 'pages'), ('faults', 'faults')):
        a, c = A[key], B[key]; print(f'   {lab:>7} A {a:>15,}  B {c:>15,}  B-A {c - a:>+15,} ({100 * (c - a) / a if a else float("nan"):+.2f}%)')
    print(f'   escapes A {A["esc"]}  B {B["esc"]}')
    print(f'   validate A: {A["val"][:90]}\n   validate B: {B["val"][:90]}')
