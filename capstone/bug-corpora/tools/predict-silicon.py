#!/usr/bin/env python3
"""Predict the plain corpora's silicon verdicts (docs/plans/2026-10-15-bug-corpus-on-fpga.md, section 5).

  predict-silicon.py --bounds-model BOUNDS_MODEL_PY (--geometry PROGRAM=RAWDIR... | --geometry-tsv TSV)
                     [--write-geometry TSV] [--align 16] [--json OUT] [--md OUT]

--geometry reads a -DCORPUS_GEOMETRY run's raw directory (the guest serial log); --write-geometry saves just the
measured lines as a TSV, which is what gets committed, and --geometry-tsv reads that back instead of raw logs.

Three plain kinds, each predicted from what silicon does differently from the virtual profile:
- plain-heap (spatial). The virtual profile has exact bounds; silicon rounds every capability OUTWARD. A virtual
  CATCH stays a catch unless the probe's access lands inside the rounding slack. The slack comes from the RTL's own
  encoder (capstone-ariane capstone-sublet verif/tests/custom/capstone/bounds-model.py, cderive_child), evaluated
  at every placement of the object's base that the allocator's alignment allows within one granule. The geometry
  is measured, not assumed: each case's BUGGY image built with -DCORPUS_GEOMETRY prints its probe's capability
  just before the access (tools/corpus-geometry.h).
- plain-temporal. The verdict is the allocator's mechanism: an allocator that revokes on free makes the stale access
  trap with cause 25 on silicon (stale-access arm, caplifive_sublet_d93a8d66b, 2026-10-16), where QEMU reads 24
  (ISSUES Q-11). level0 does not revoke, so the access completes. Both columns are given; decision (b) picks.
- subobject. The crossing stays inside one allocation, and outward rounding only enlarges bounds, so it is MISSED
  on silicon as on the virtual profile.

A positive control runs first: a 65535-byte object read one byte past its end must NOT be predicted CAUGHT on
every placement (the RTL lane's own example: 113 bytes of slack above at a +0x10 cursor). If it is, the model or
the tool is not doing what this file says, and nothing is printed. A case with no geometry is an error, not a row.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEOM = re.compile(r'^GEOM probe=(\S+) type=(\d+) base=(\S+) end=(\S+) cursor=(\S+) width=(\d+)')
PLAIN = {'memcached': ['plain-heap-repros', 'plain-temporal-repros'],
         'ffmpeg': ['plain-heap-repros', 'plain-temporal-repros', 'subobject-repros'],
         'wireshark': ['plain-heap-repros', 'plain-temporal-repros']}


def load_model(path):
    spec = importlib.util.spec_from_file_location('bounds_model', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spatial(model, size, off, width, align, cursor0=0x80010000):
    """Over every allowed placement of the object's base within one granule: CAUGHT if the access [off, off+width)
    leaves the silicon bounds [-slack_below, size+slack_above), relative to the object's start."""
    granule = max(model.cderive_child(cursor0, size)['granule'], align)
    outcomes, worst = [], None
    for k in range(0, granule, align):
        r = model.cderive_child(cursor0 + k, size)
        caught = off < -r['slack_below'] or off + width > size + r['slack_above']
        outcomes.append(caught)
        if worst is None or r['slack_above'] + r['slack_below'] > worst['slack_above'] + worst['slack_below']:
            worst = r
    n = sum(outcomes)
    verdict = 'CAUGHT' if n == len(outcomes) else 'MISSED' if n == 0 else f'DEPENDS ({n}/{len(outcomes)} placements catch)'
    return verdict, worst


def control(model, align):
    v, worst = spatial(model, 65535, 65535, 1, align)
    if v == 'CAUGHT':
        raise SystemExit(f'positive control FAILED: a 65535-byte object read one byte past its end is predicted '
                         f'CAUGHT on every placement (slack {worst}); the model is not rounding outward')
    return v, worst


def geometry(rawdir):
    """case number -> the LAST GEOM line of its BUGGY run (the access that crossed). The runner boots once per
    corpus, so the guest's serial log holds every run, each opened by the driver's `case=N arm=fixed|buggy` line."""
    out, current = {}, None
    logs = sorted(pathlib.Path(rawdir).rglob('serial.log'))
    if not logs:
        raise SystemExit(f'no serial.log under {rawdir}')
    for log in logs:
        for line in log.read_text(errors='replace').replace('\r', '').splitlines():
            m = re.match(r'^case=(\d+) arm=(fixed|buggy)$', line)
            if m:
                current = (int(m[1]), m[2])
                continue
            g = GEOM.match(line)
            if g and current and current[1] == 'buggy':
                out[current[0]] = g
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bounds-model', type=pathlib.Path, required=True)
    p.add_argument('--geometry', action='append', default=[], help='PROGRAM=RAWDIR of a -DCORPUS_GEOMETRY run')
    p.add_argument('--geometry-tsv', type=pathlib.Path, help='program, case number and GEOM line, tab-separated')
    p.add_argument('--write-geometry', type=pathlib.Path)
    p.add_argument('--align', type=int, default=16, help="the silicon allocator's base alignment, in bytes")
    p.add_argument('--json', type=pathlib.Path)
    p.add_argument('--md', type=pathlib.Path)
    a = p.parse_args()
    model = load_model(a.bounds_model)
    ctl = control(model, a.align)
    geo = {}
    for g in a.geometry:
        prog, raw = g.split('=', 1)
        geo[prog] = geometry(raw)
    if a.geometry_tsv:
        for line in a.geometry_tsv.read_text().splitlines():
            if line and not line.startswith('#'):
                prog, num, geom = line.split('\t')
                geo.setdefault(prog, {})[int(num)] = GEOM.match(geom)
    if not geo:
        raise SystemExit('no geometry given (--geometry or --geometry-tsv)')
    if a.write_geometry:
        a.write_geometry.write_text('# program\tcase\tGEOM line (tools/corpus-geometry.h), last one of the BUGGY run\n'
                                    + ''.join(f'{prog}\t{num}\t{m.group(0)}\n' for prog in sorted(geo)
                                              for num, m in sorted(geo[prog].items())))
    rows, missing = [], []
    for prog, corpora in PLAIN.items():
        for corpus in corpora:
            for cj in sorted((ROOT / prog / corpus).glob('[0-9][0-9]_*/case.json')):
                case = json.loads(cj.read_text())
                if case.get('duplicate_of'):
                    continue
                name = cj.parent.name
                vm = (case.get('arms') or {}).get('virtual-malloc') or {}
                virtual = vm.get('verdict') if isinstance(vm, dict) else None
                row = {'program': prog, 'corpus': corpus, 'case': name, 'virtual_malloc': virtual}
                if corpus == 'plain-heap-repros':
                    m = geo.get(prog, {}).get(case.get('case'))
                    if not m:
                        missing.append(f'{prog}/{corpus}/{name}')
                        continue
                    probe, typ, base, end, cur, width = m.groups()
                    base, end, cur, width = int(base, 16), int(end, 16), int(cur, 16), int(width)
                    if int(typ) == 7:
                        missing.append(f'{prog}/{corpus}/{name} (probe pointer untagged)')
                        continue
                    size, off = end - base, cur - base
                    verdict, worst = spatial(model, size, off, width, a.align)
                    # The same object from an allocator that derives the child at the size ROUNDED UP to its
                    # alignment: the access is then measured against the rounded child, slack included.
                    rounded = -(-size // a.align) * a.align
                    verdict_r, _ = spatial(model, rounded, off, width, a.align)
                    row.update(kind='spatial', probe=probe, size=size, offset=off, width=width,
                               exact=worst['exact'], slack_below=worst['slack_below'],
                               slack_above=worst['slack_above'], silicon=verdict,
                               changes=(virtual == 'CAUGHT' and verdict != 'CAUGHT'),
                               rounded_size=rounded, silicon_rounded=verdict_r,
                               changes_rounded=(virtual == 'CAUGHT' and verdict_r != 'CAUGHT'))
                elif corpus == 'plain-temporal-repros':
                    row.update(kind='temporal', silicon_revoking='CAUGHT (cause 25)', silicon_level0='MISSED',
                               changes=None)
                else:
                    row.update(kind='subobject', silicon='MISSED', changes=(virtual == 'CAUGHT'))
                rows.append(row)
    if missing:
        raise SystemExit('no geometry for: ' + ', '.join(missing))
    sha = hashlib.sha256(a.bounds_model.read_bytes()).hexdigest()[:16]
    summary = {'bounds_model_sha256': sha, 'align': a.align, 'control': {'verdict': ctl[0], 'worst': ctl[1]},
               'rows': rows}
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1) + '\n')
    lines = [f'bounds model sha256 {sha}; allocator alignment {a.align}; positive control (65535 B, read at +65535): '
             f'{ctl[0]}', '',
             '| program | corpus | case | virtual-malloc | size | offset | width | slack below/above (worst) | silicon, child at the request size | silicon, child at the size rounded to the alignment |',
             '|---|---|---|---|---:|---:|---:|---|---|---|']
    for r in rows:
        if r['kind'] == 'spatial':
            lines.append(f"| {r['program']} | {r['corpus']} | {r['case']} | {r['virtual_malloc']} | {r['size']} | "
                         f"{r['offset']} | {r['width']} | {r['slack_below']}/{r['slack_above']} | {r['silicon']}"
                         f"{' (CHANGES)' if r['changes'] else ''} | {r['silicon_rounded']} (size {r['rounded_size']})"
                         f"{' (CHANGES)' if r['changes_rounded'] else ''} |")
        elif r['kind'] == 'temporal':
            lines.append(f"| {r['program']} | {r['corpus']} | {r['case']} | {r['virtual_malloc']} | | | | | "
                         f"revoking: {r['silicon_revoking']}; level0: {r['silicon_level0']} | same |")
        else:
            lines.append(f"| {r['program']} | {r['corpus']} | {r['case']} | {r['virtual_malloc']} | | | | | "
                         f"{r['silicon']} | {r['silicon']} |")
    text = '\n'.join(lines) + '\n'
    if a.md:
        a.md.write_text(text)
    print(text)
    kinds = {k: sum(1 for r in rows if r['kind'] == k) for k in ('spatial', 'temporal', 'subobject')}
    changed = sum(1 for r in rows if r.get('changes'))
    changed_r = sum(1 for r in rows if r.get('changes_rounded'))
    print(f'rows {len(rows)} {kinds}; spatial verdicts that change from virtual: {changed} with the child at the '
          f'request size, {changed_r} with it rounded to {a.align}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
