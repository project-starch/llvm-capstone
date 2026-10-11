#!/usr/bin/env python3
"""Codegen gate and pointer-cast census for a capstone64 build of a native CMake/ninja tree.

    codegen-gate.py --compdb compile_commands.json --sources linked.txt --out DIR
                    [--inc DIR]... [--define NAME]... [--jobs N] [--timeout S]
                    [--replace-inc NATIVE=CAPSTONE]...

Needs CAPSTONE_CLANG (from capstone/tests/capstone-test-env.sh); CAPSTONE_MUSL_SRC defaults to
/tmp/capstone/musl-src/musl-1.2.5.

For every source in --sources (one path per line; the translation units the program LINKS),
take its native compile command, keep its -I/-D/-include arguments, and compile it for
capstone64 with `-S -o /dev/null`: CODEGEN, not -fsyntax-only, because a -fsyntax-only census
missed a backend failure (__thread, ISSUES C-47) on 2026-09-23.

The same pass counts two cast classes with clang's own warnings, which on capstone64 fire on
EVERY pointer->integer cast (a 16-byte pointer is wider than any integer the code casts to):
  lossy    -Wvoid-pointer-to-int-cast, -Wpointer-to-int-cast   pointer -> integer
  rebuilt  -Wint-to-void-pointer-cast, -Wint-to-pointer-cast   integer -> pointer
A lossy site loses the capability tag; it is a provenance bug only if the integer later becomes
a pointer again, which the classification pass (by hand) decides. -Wcapstone-pointer-roundtrip
is NOT used: it sees only `uintptr_t`-spelled values and misses GLib's GPOINTER_TO_* macros.

Outputs in DIR: result.txt (one line per file: ok|error|timeout, lossy, rebuilt),
lossy-sites.txt, rebuilt-sites.txt, errors.txt. Exit 1 if any file failed codegen or timed out;
exit 2 (ERROR) if no source was compiled at all, or CAPSTONE_CLANG is not set.
"""
import argparse, concurrent.futures as cf, json, os, re, shlex, subprocess, sys, time

CL = os.environ.get('CAPSTONE_CLANG', '')      # set by capstone/tests/capstone-test-env.sh
MUSL = os.environ.get('CAPSTONE_MUSL_SRC', '/tmp/capstone/musl-src/musl-1.2.5')
if not os.access(CL, os.X_OK):
    print(f'ERROR: CAPSTONE_CLANG={CL!r} is not an executable; source capstone-test-env.sh',
          file=sys.stderr)
    sys.exit(2)
LOSSY = ('-Wvoid-pointer-to-int-cast', '-Wpointer-to-int-cast')
REBUILT = ('-Wint-to-void-pointer-cast', '-Wint-to-pointer-cast')


def base_flags(incs, defines):
    res = subprocess.check_output([CL, '-print-resource-dir'], text=True).strip()
    f = ['-target', 'capstone64-unknown-elf', '-Xclang', '-target-feature', '-Xclang', '+m',
         '-Xclang', '-target-feature', '-Xclang', '+a', '-ffreestanding', '-O1', '-S',
         '-o', '/dev/null', '-fno-crash-diagnostics', '-D_GNU_SOURCE', '-nostdinc',
         '-isystem', f'{MUSL}/arch/capstone64', '-isystem', f'{MUSL}/arch/generic',
         '-isystem', f'{MUSL}/obj/include', '-isystem', f'{MUSL}/include',
         '-isystem', f'{res}/include', '-Wno-everything', *LOSSY, *REBUILT]
    for d in defines:
        f.append(f'-D{d}')
    for i in incs:          # ahead of the native paths: capstone64 config.h/glibconfig.h win
        f += ['-I', i]
    return f


def native_args(cmd, replace):
    args, keep, i = shlex.split(cmd), [], 0
    while i < len(args):
        a = args[i]
        if a in ('-I', '-isystem', '-D', '-include') and i + 1 < len(args):
            v = replace.get(args[i + 1], args[i + 1])
            if v:
                keep += ['-I' if a == '-isystem' else a, v]
            i += 2
            continue
        if a.startswith('-I') or a.startswith('-D'):
            v = replace.get(a[2:], a[2:])
            if v:
                keep.append(a[:2] + v)
        i += 1
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compdb', required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--inc', action='append', default=[])
    p.add_argument('--define', action='append', default=[])
    p.add_argument('--replace-inc', action='append', default=[],
                   help='NATIVE=CAPSTONE; an empty CAPSTONE drops the native path')
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    replace = dict(r.split('=', 1) for r in a.replace_inc)
    want = [l.strip() for l in open(a.sources) if l.strip() and '/tools/lemon/' not in l]
    cmds = {}
    for e in json.load(open(a.compdb)):
        f = os.path.normpath(os.path.join(e['directory'], e['file']))
        c = e.get('command') or ' '.join(map(shlex.quote, e.get('arguments', [])))
        if not c:                       # entries with no command at all count as missing below
            continue
        # A source can have several entries (other rules, other targets). Prefer the one that
        # PRODUCES AN OBJECT (-o ...): an entry without it compiled this gate's first run of the
        # tshark census with the wrong flags for 13 files (no -DNDEBUG).
        makes_obj = ' -o ' in f' {c} '
        if f not in cmds or (makes_obj and not cmds[f][2]):
            cmds[f] = (e['directory'], c, makes_obj)
    base = base_flags(a.inc, a.define)
    jobs, missing = [], []
    for f in want:
        if f not in cmds:
            missing.append(f)
            continue
        d, c, _ = cmds[f]
        jobs.append((f, d, base + native_args(c, replace) + [f]))
    if not jobs:
        print('ERROR: no source from --sources has a compile command in --compdb', file=sys.stderr)
        sys.exit(2)

    def run(j):
        f, d, cmd = j
        t0 = time.time()
        try:
            r = subprocess.run([CL] + cmd, cwd=d, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            return f, 'timeout', '', time.time() - t0
        state = 'ok' if r.returncode == 0 else 'error'
        return f, state, r.stderr, time.time() - t0

    os.makedirs(a.out, exist_ok=True)
    rows, lossy, rebuilt, errors = [], [], [], []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for f, state, err, dt in ex.map(run, jobs):
            lo = [l for l in err.splitlines() if 'warning:' in l and any(w in l for w in LOSSY)]
            rb = [l for l in err.splitlines() if 'warning:' in l and any(w in l for w in REBUILT)]
            lossy += lo
            rebuilt += rb
            if state != 'ok':
                first = next((l for l in err.splitlines() if 'error' in l), state)
                errors.append(f'{f}: {first}')
            rows.append(f'{state:<8} {len(lo):5d} {len(rb):5d} {dt:7.1f}s {f}')
    open(f'{a.out}/result.txt', 'w').write('\n'.join(sorted(rows)) + '\n')
    open(f'{a.out}/lossy-sites.txt', 'w').write('\n'.join(sorted(set(lossy))) + '\n')
    open(f'{a.out}/rebuilt-sites.txt', 'w').write('\n'.join(sorted(set(rebuilt))) + '\n')
    open(f'{a.out}/errors.txt', 'w').write('\n'.join(errors) + '\n')
    n_bad = sum(1 for r in rows if not r.startswith('ok'))
    print(f'compiled {len(jobs)} ({len(missing)} listed sources had no compile command); '
          f'codegen failures {n_bad}; lossy casts {len(set(lossy))} sites; '
          f'rebuilt casts {len(set(rebuilt))} sites')
    for m in missing[:5]:
        print(f'  no compile command: {m}')
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
