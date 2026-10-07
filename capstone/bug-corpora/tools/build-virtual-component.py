#!/usr/bin/env python3
"""Build an allocator-component corpus's cases through the port's own CMake seam,
for the VIRTUAL address space.

    build-virtual-component.py --port <ports/...> --corpus <corpus dir>
                               --corpus-var APRP_CORPUS_SRC --out <new dir>
                               [--define APRP_BUCKETS=ON]... [--only 00,03]
                               [--jobs 4]

WHY THE PORT'S SEAM AND NOT A DIRECT COMPILE. These corpora replay a defect
against the APPLICATION'S OWN allocator -- APR's pools, memcached's slabs,
pymalloc's arenas, wmem -- so the case is only half the program; the other half
is the upstream allocator the port prepares, patches and compiles. The port
already exposes exactly one way in, a cache variable naming one corpus source,
and both the native and the physical-domain arms go through it. This arm goes
through the same one, with the `capstone-application` preset, so the virtual
images differ from the native ones in the toolchain and in nothing else.

The preset builds the port's HOSTED sources: an ordinary `main()`, the
platform's malloc under the allocator, and a libc. On the virtual platform that
platform allocator is the virtual SDK's heap -- per-object bounds and a revoke
on free -- so what this arm measures is a nested allocator sitting on a
protected system allocator, which is the question the corpora were built to ask.
It is NOT the port's own protected nested allocator; that is a separate arm and
a separate image.

TWO SEAM SHAPES, because the ports have two. Most take ONE corpus source and
build one `defects` target, so this runs one configure and one build per case
(`--corpus-var`); the port's library is recompiled each time, which is slower
than it could be and is what keeps the arms honest about building the same way.
The wmem port instead takes the corpus ROOT and declares one target per case in
a single configure (`--corpus-dir-var`), naming them `NN-slug` as the contract
names run artifacts. Both end the same way: one image per case, named after its
case directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--corpus-var', help="The port's cache variable naming ONE case source")
    p.add_argument('--corpus-dir-var', help="The port's cache variable naming the corpus ROOT")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--define', action='append', default=[], help='NAME=VALUE, repeatable')
    p.add_argument('--only', help='comma-separated case number prefixes')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--target', default='defects', help='The seam\'s executable target')
    a = p.parse_args()
    if bool(a.corpus_var) == bool(a.corpus_dir_var):
        p.error('give exactly one of --corpus-var and --corpus-dir-var')

    if not (a.port / 'CMakePresets.json').is_file():
        sys.exit(f'{a.port} has no CMakePresets.json')
    presets = json.loads((a.port / 'CMakePresets.json').read_text())
    if 'capstone-application' not in [c['name'] for c in presets['configurePresets']]:
        sys.exit(f'{a.port} has no capstone-application preset; add one before '
                 f'claiming this component builds for the virtual platform')
    cases = sorted(d for d in a.corpus.glob('[0-9][0-9]_*') if (d / 'case.c').is_file())
    if a.only:
        keep = {w.zfill(2) for w in a.only.split(',')}
        cases = [d for d in cases if d.name[:2] in keep]
    if not cases:
        sys.exit(f'no case directories with a case.c under {a.corpus}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'work').mkdir(exist_ok=True)
    built, failed = [], []
    if a.corpus_dir_var:
        return whole_corpus(a, cases, built, failed)
    for case in cases:
        work = a.out / 'work' / case.name[:2]
        log = a.out / f'{case.name}.log'
        with log.open('w') as stream:
            configure = ['cmake', '--preset', 'capstone-application',
                         '-S', str(a.port.resolve()), '-B', str(work.resolve()),
                         f'-D{a.corpus_var}={(case / "case.c").resolve()}',
                         *(f'-D{d}' for d in a.define)]
            stream.write(' '.join(configure) + '\n')
            stream.flush()
            ok = not subprocess.run(configure, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
            if ok:
                build = ['cmake', '--build', str(work.resolve()),
                         '--target', a.target, '-j', str(a.jobs)]
                stream.write('\n' + ' '.join(build) + '\n')
                stream.flush()
                ok = not subprocess.run(build, stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
        image = work / 'bin' / a.target
        if ok and image.is_file():
            shutil.copy2(image, a.out / f'{case.name}.dom')
            print(f'  {case.name:<56} OK  {image.stat().st_size} bytes')
            built.append(case.name)
        else:
            print(f'  {case.name:<56} FAIL ({log})')
            sys.stdout.write(''.join(
                f'      {line}' for line in log.read_text().splitlines(True)
                if 'error' in line.lower())[:800])
            failed.append(case.name)
    (a.out / 'build.json').write_text(json.dumps(
        dict(port=str(a.port), corpus=str(a.corpus), preset='capstone-application',
             corpus_var=a.corpus_var, defines=a.define, target=a.target,
             built=built, failed=failed), indent=2) + '\n')
    print(f'built={len(built)} failed={len(failed)}  ({a.out})')
    return 1 if failed else 0


def stem(name):
    """`00_fix_a_b` -> `00-a-b`, the target name the wmem port declares."""
    number, _, rest = name.partition('_')
    return number + '-' + rest.partition('_')[2].replace('_', '-')


def whole_corpus(a, cases, built, failed):
    """One configure, one target per case: the wmem port's seam."""
    work = a.out / 'work' / 'all'
    log = a.out / 'configure.log'
    stems = {case: stem(case.name) for case in cases}
    with log.open('w') as stream:
        configure = ['cmake', '--preset', 'capstone-application',
                     '-S', str(a.port.resolve()), '-B', str(work.resolve()),
                     f'-D{a.corpus_dir_var}={a.corpus.resolve()}',
                     *(f'-D{d}' for d in a.define)]
        stream.write(' '.join(configure) + '\n')
        stream.flush()
        if subprocess.run(configure, stdout=stream, stderr=subprocess.STDOUT).returncode:
            print(f'configure FAILED ({log})')
            return 1
        build = ['cmake', '--build', str(work.resolve()), '-j', str(a.jobs),
                 '--target', *sorted(stems.values())]
        stream.write('\n' + ' '.join(build) + '\n')
        stream.flush()
        subprocess.run(build, stdout=stream, stderr=subprocess.STDOUT)
    for case, name in sorted(stems.items()):
        image = work / 'bin' / name
        if image.is_file():
            shutil.copy2(image, a.out / f'{case.name}.dom')
            print(f'  {case.name:<56} OK  {image.stat().st_size} bytes')
            built.append(case.name)
        else:
            print(f'  {case.name:<56} FAIL (no {image})')
            failed.append(case.name)
    (a.out / 'build.json').write_text(json.dumps(
        dict(port=str(a.port), corpus=str(a.corpus), preset='capstone-application',
             corpus_dir_var=a.corpus_dir_var, defines=a.define,
             targets=sorted(stems.values()), built=built, failed=failed), indent=2) + '\n')
    print(f'built={len(built)} failed={len(failed)}  ({a.out})')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
