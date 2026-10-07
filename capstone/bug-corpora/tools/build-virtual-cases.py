#!/usr/bin/env python3
"""Build a corpus's cases as VIRTUAL Capstone applications, one per case.

    build-virtual-cases.py --corpus <corpus dir> --sdk <virtual SDK dir>
                           --out <new dir> [--source FILE]... [--include DIR]...
                           [--opt -O0] [--define NAME=VALUE]... [--only 00,03]

The virtual SDK is an APPLICATION SDK: `capstone-cc` compiles against the
Capstone musl headers and links the virtual application runtime, so a case that
calls the platform's own `malloc`/`calloc` -- which is what the plain-heap and
client corpora do -- needs no port library and no entry adapter. That is why
these corpora can be measured on Capstone at all: their physical `spatial` and
`sublet` arms were declared PREDICTED because no capstone-domain runner existed,
and the bare-metal domain build they would have needed has no libc.

`shared/driver.c` is compiled in by default because every corpus in this tree
keeps `main()` there; pass `--source` for anything else a corpus needs and
`--no-driver` for one that has none.

-O0 IS THE DEFAULT, deliberately. These cases turn on one specific read or
write running off one specific object. An optimiser that hoists, merges or
discards it moves the fault away from the labelled probe, and the corpora's
CheriBSD arms were re-measured at -O0 for exactly that reason. A build at -O1
measures the optimiser as much as the mechanism.

Each image is named after its case directory, so the run's rows, the plan and
the archived result tree all carry the same name.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, action='append', default=[],
                   help='An extra translation unit, repeatable')
    p.add_argument('--include', type=Path, action='append', default=[],
                   help='An extra include directory, repeatable')
    p.add_argument('--link', type=Path, action='append', default=[],
                   help='An archive to link after the sources, repeatable: the '
                        'corpora whose cases replay a PORTED allocator need its '
                        'library, built for this platform')
    p.add_argument('--define', action='append', default=[], help='NAME or NAME=VALUE')
    p.add_argument('--opt', default='-O0')
    p.add_argument('--no-driver', action='store_true')
    p.add_argument('--only', help='comma-separated case number prefixes')
    a = p.parse_args()

    cc = a.sdk / 'capstone-cc'
    if not cc.is_file():
        sys.exit(f'no capstone-cc in {a.sdk}')
    cases = sorted(d for d in a.corpus.glob('[0-9][0-9]_*') if (d / 'case.c').is_file())
    if a.only:
        keep = {w.zfill(2) for w in a.only.split(',')}
        cases = [d for d in cases if d.name[:2] in keep]
    if not cases:
        sys.exit(f'no case directories with a case.c under {a.corpus}')

    shared = a.corpus / 'shared'
    sources = list(a.source)
    if not a.no_driver:
        driver = shared / 'driver.c'
        if not driver.is_file():
            sys.exit(f'{driver} does not exist; pass --no-driver or --source')
        sources.insert(0, driver)
    includes = [shared, *a.include] if shared.is_dir() else list(a.include)

    a.out.mkdir(parents=True, exist_ok=True)
    built, failed = [], []
    for case in cases:
        image = a.out / f'{case.name}.dom'
        command = [str(cc), a.opt, *(f'-D{d}' for d in a.define),
                   *(f'-I{i}' for i in includes), str(case / 'case.c'),
                   *(str(s) for s in sources), *(str(l) for l in a.link),
                   '-o', str(image)]
        log = a.out / f'{case.name}.err'
        with log.open('w') as stream:
            stream.write(' '.join(command) + '\n\n')
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'  {case.name:<56} FAIL')
            sys.stdout.write(''.join(f'      {l}' for l in
                                     log.read_text().splitlines(True)[2:10]))
            failed.append(case.name)
        else:
            print(f'  {case.name:<56} OK  {image.stat().st_size} bytes')
            built.append(case.name)
    (a.out / 'build.json').write_text(json.dumps(
        dict(corpus=str(a.corpus), sdk=str(a.sdk), opt=a.opt,
             sources=[str(s) for s in sources], includes=[str(i) for i in includes],
             link=[str(l) for l in a.link], defines=a.define,
             built=built, failed=failed), indent=2) + '\n')
    print(f'built={len(built)} failed={len(failed)}  ({a.out})')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
