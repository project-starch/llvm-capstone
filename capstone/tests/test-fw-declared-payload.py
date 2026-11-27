#!/usr/bin/env python3
"""test-fw-declared-payload.py [CHECKER] : synthetic controls for fw-declared-payload.py (default: the one beside this file).

Each case writes a stand-in firmware (a few junk bytes, then one gzip stream of concatenated newc archives, as the board's
payloads carry the rootfs and the overlay) and a declared dir, runs the checker, and compares its exit code with the
expected one. Exit 0 only if every case reads as expected. The cases (2026-11-27, the board lane and the runtime lane):
  1 one archive, every file in base                          -> 0 (GO), as before the multi-archive change
  2 two archives, a new file in archive 2 not declared       -> 1 (the first-archive-only checker passed this)
  3 two archives, a declared file absent                     -> 1
  4 archive 2 overrides an archive-1 path, not in replaced   -> 1
  5 the same override declared in replaced.txt               -> 0
  6 non-zero bytes between the archives                      -> 2 (malformed)
  7 two archives, archive-2 file declared in manifest        -> 0"""
import gzip, hashlib, os, subprocess, sys, tempfile
from pathlib import Path

CHK = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name('fw-declared-payload.py'))

def entry(name, data, mode=0o100644):
    nb = name.encode() + b'\0'
    h = b'070701' + b''.join(b'%08X' % v for v in (0, mode, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(nb), 0))
    out = h + nb; out += b'\0' * ((4 - len(out) % 4) % 4); out += data; out += b'\0' * ((4 - len(out) % 4) % 4)
    return out

def archive(files):
    return b''.join(entry(n, d) for n, d in files) + entry('TRAILER!!!', b'', 0)

def sha(b): return hashlib.sha256(b).hexdigest()

def run(case, archives, base, manifest, replaced=None, gap=b''):
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); cpio = gap.join(archives) if gap else b''.join(a + b'\0' * 512 for a in archives)
        (t / 'fw.bin').write_bytes(b'\x7fJUNK' + gzip.compress(cpio) + b'TAIL')
        dd = t / 'decl'; dd.mkdir()
        (dd / 'base.txt').write_text(''.join(f'{sha(d)}  ./{n}\n' for n, d in base))
        (dd / 'manifest.txt').write_text(''.join(f'{sha(d)}  ./{n}\n' for n, d in manifest))
        if replaced is not None: (dd / 'replaced.txt').write_text(''.join(f'{sha(d)}  ./{n}\n' for n, d in replaced))
        r = subprocess.run([sys.executable, CHK, str(t / 'fw.bin'), str(dd)], capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]

B = [('init', b'#!/bin/sh base\n'), ('bin/sh', b'busybox'), ('etc/x', b'x')]
O = [('init', b'#!/bin/sh overlay\n'), ('stages/a/run.sh', b'run a'), ('smoke.dom', b'dom')]
cases = [
    (1, 0, dict(archives=[archive(B)], base=B, manifest=[('bin/sh', b'busybox')])),
    (2, 1, dict(archives=[archive(B), archive([('stages/stale.dom', b'stale')])], base=B, manifest=[('bin/sh', b'busybox')])),
    (3, 1, dict(archives=[archive(B), archive(O[1:])], base=B, manifest=O[1:] + [('stages/b/run.sh', b'never built')])),
    (4, 1, dict(archives=[archive(B), archive(O)], base=B, manifest=O)),
    (5, 0, dict(archives=[archive(B), archive(O)], base=B, manifest=O[1:], replaced=[O[0]])),
    (6, 2, dict(archives=[archive(B), archive(O[1:])], base=B, manifest=O[1:], gap=b'\0\0\0\0GARBAGE')),
    (7, 0, dict(archives=[archive(B), archive(O[1:])], base=B, manifest=O[1:])),
]
bad = 0
for n, want, kw in cases:
    got, last = run(n, **kw)
    ok = got == want; bad += not ok
    print(f"case {n}: want {want} got {got} {'OK' if ok else 'WRONG'} | {last[0][:150] if last else ''}")
print(f"TEST-FW-DECLARED {'PASS' if not bad else 'FAIL'} ({len(cases) - bad}/{len(cases)}) on {CHK}")
sys.exit(1 if bad else 0)
