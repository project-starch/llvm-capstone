#!/usr/bin/env python3
"""Lift the corpus trigger sources out of the physical engine's #ifdef ladder into files.

WHY. The physical port selects a trigger with -DRUNG_E_CRASHnnn, so the two arms of a case are
two different IMAGES: they differ in the allocator mode under test AND in the compiled-in
source text, the string table, and every address downstream of it. On the virtual profile the
trigger arrives on stdin, so one image serves every case and the arms differ in the allocator
mode alone. That is the whole reason this script exists -- not tidiness.

WHY THE C PREPROCESSOR AND NOT A PARSER. The ladder's sources are C string literals: escaped
quotes, \\n, and line continuations that concatenate adjacent literals. Re-implementing that is
a transcription risk on the one piece of data the whole experiment rests on -- a trigger that
differs from the physical arm's by one character is a different program. So the real cpp does
the work, keyed on the same -D the physical build uses, and this script only unquotes the
result.

POSITIVE CONTROL, because a silent default is the failure mode this guards. rung_E_domain.c's
`# else` arm is CRASH-110's source, so an id with no branch of its own does not fail -- it
quietly yields parse_url("file:///"). Every extracted source is therefore checked against that
default and a match is an ERROR for every id but CRASH-110, which owns it.

  extract-triggers.py            write triggers/ and triggers/manifest.json
  extract-triggers.py --check    regenerate into a temp dir and diff; non-zero on drift
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER_SRC = HERE.parent / 'engine' / 'rung_E_domain.c'
OUT = HERE / 'triggers'


def ladder_text(path):
    """The `#ifndef RUNG_E_SOURCE` block, found by matching #if nesting rather than by line
    number -- the file is edited often and a hardcoded range would silently truncate."""
    lines = path.read_text().splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith('#ifndef RUNG_E_SOURCE'))
    depth = 0
    for i in range(start, len(lines)):
        s = lines[i].lstrip('#').strip()
        directive = s.split()[0] if s.split() else ''
        if directive in ('if', 'ifdef', 'ifndef'):
            depth += 1
        elif directive == 'endif':
            depth -= 1
            if depth == 0:
                return '\n'.join(lines[start:i + 1]) + '\n'
    raise SystemExit(f'{path}: unterminated #ifndef RUNG_E_SOURCE block')


def unquote(text):
    """Join adjacent C string literals and resolve the escapes the ladder actually uses."""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    if not parts:
        raise SystemExit(f'not a string literal: {text!r}')
    escapes = {'n': '\n', 't': '\t', '"': '"', '\\': '\\', "'": "'", '0': '\0'}
    out = []
    for part in parts:
        i = 0
        while i < len(part):
            if part[i] == '\\':
                nxt = part[i + 1]
                if nxt not in escapes:
                    raise SystemExit(f'unhandled escape \\{nxt} in {part!r}')
                out.append(escapes[nxt])
                i += 2
            else:
                out.append(part[i])
                i += 1
    return ''.join(out)


def extract(ladder, case_id):
    """Run the preprocessor with this case's -D and read back its source and statement flag."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'ladder.h').write_text(ladder)
        (d / 'probe.c').write_text('#include "ladder.h"\n'
                                   'SOURCE RUNG_E_SOURCE\n'
                                   '#ifdef RUNG_E_STMT\n'
                                   'MODE stmt\n'
                                   '#else\n'
                                   'MODE expr\n'
                                   '#endif\n')
        define = [] if case_id is None else [f'-DRUNG_E_{case_id.replace("-", "")}']
        cpp = subprocess.run(['cc', '-E', '-P', *define, str(d / 'probe.c')],
                             capture_output=True, text=True)
        if cpp.returncode != 0:
            raise SystemExit(f'{case_id}: cpp failed: {cpp.stderr.strip()}')
    source = mode = None
    for line in cpp.stdout.splitlines():
        if line.startswith('SOURCE '):
            source = unquote(line[len('SOURCE '):])
        elif line.startswith('MODE '):
            mode = line.split()[1]
    if source is None or mode is None:
        raise SystemExit(f'{case_id}: cpp output carried no SOURCE/MODE')
    return source, mode


def main():
    check = '--check' in sys.argv[1:]
    ladder = ladder_text(LADDER_SRC)
    ids = sorted({f'CRASH-{m[1]}' for m in re.finditer(r'RUNG_E_CRASH(\d+)', ladder)})
    if len(ids) < 20:
        raise SystemExit(f'only {len(ids)} cases found in the ladder -- refusing to write a '
                         f'partial trigger set')
    default, _ = extract(ladder, None)

    manifest, written = {}, {}
    for case_id in ids:
        source, mode = extract(ladder, case_id)
        if source == default and case_id != 'CRASH-110':
            raise SystemExit(f'{case_id}: extracted source is the ladder DEFAULT '
                             f'({default!r}) -- that id has no branch of its own, so the '
                             f'physical build has been compiling the wrong trigger for it')
        if not source.strip():
            raise SystemExit(f'{case_id}: empty trigger source')
        written[f'{case_id}.php'] = source
        manifest[case_id] = {'file': f'{case_id}.php', 'mode': mode}

    target = Path(tempfile.mkdtemp()) if check else OUT
    target.mkdir(parents=True, exist_ok=True)
    for name, text in written.items():
        (target / name).write_text(text)
    (target / 'manifest.json').write_text(json.dumps(
        {'source': str(LADDER_SRC.relative_to(LADDER_SRC.parents[3])),
         'cases': manifest}, indent=2, sort_keys=True) + '\n')

    if not check:
        print(f'{len(written)} triggers -> {target}')
        for case_id in ids:
            print(f'  {case_id:<10} {manifest[case_id]["mode"]:<4} '
                  f'{written[case_id + ".php"]!r:.70}')
        return 0

    drift = []
    for name in sorted(list(written) + ['manifest.json']):
        have = OUT / name
        want = (target / name).read_text()
        if not have.exists():
            drift.append(f'  MISSING  {name}')
        elif have.read_text() != want:
            drift.append(f'  DIFFERS  {name}')
    for stale in sorted(p.name for p in OUT.glob('*')):
        if stale not in written and stale != 'manifest.json':
            drift.append(f'  STALE    {stale}')
    if drift:
        print(f'triggers/ is out of date with {LADDER_SRC.name}:')
        print('\n'.join(drift))
        print('re-run extract-triggers.py and commit the result')
        return 1
    print(f'triggers/ matches {LADDER_SRC.name}: {len(written)} cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
