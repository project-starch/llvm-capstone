#!/usr/bin/env python3
"""Score one lookaside row of PoisonCap's published arm.

    score-cases.py --published RUN [--spatial RUN] --nm PATH/llvm-nm --rootfs DIR --out DIR

RUN is a run-cases.py output directory (run.json, results.jsonl); each names the build it ran,
whose unstripped binaries give the symbol table a fault's PC offset is looked up in. Run where
PoisonCap's llvm-nm runs (platform/run-in-container.sh).

Per case, the plan's tiers (bug-corpora/sqlite/engine-repros/results/2026-10-24-poisoncap/PLAN.md, rule 5):
  qualified         the case declares fault_sites (case.json); otherwise "no qualified trigger"
  fault at site     the process faulted and the PC's function is one of the fault_sites
  mechanism-signed  fault at site, and the fault is the poison report: SIGPROT si_code 3
                    (PROT_CHERI_SEALED, which their QEMU raises for poison) with a poison line on
                    the console during that process; no other path produces both
  attributable      only with --spatial: fault at site on published and none on spatial (the arms
                    differ only in NESTED_TEMPORAL)
NO-READING: no BEGIN printed, a CONTROL-FAILED line, a timeout, or a non-signal error exit.
RETURNED-WITH-ERROR: the process returned, but printed a failing `... rc=N` line (N != 0) first;
REPRO322_MAIN ignores run_case()'s result, so a case that stopped early (heap exhausted, say) would
otherwise read as a clean return. Reported as its own outcome, never as a miss or a catch.
A fault's function is the sized symbol (llvm-nm --print-size) that contains the PC; a PC outside
every sized symbol, or an object that cannot be read, is "unresolved", never at a site. Signed
requires SIGPROT as well as code 3.
"""
import argparse
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[5]
CORPUS = REPO / 'capstone/bug-corpora/sqlite/engine-repros'
DEMOS = REPO / 'capstone/ports/sqlite/poisoncap/backport'  # demonstrations, not corpus cases (build-virtual.py)


def load(run):
    meta = json.loads((run / 'run.json').read_text())
    rows = [json.loads(l) for l in (run / 'results.jsonl').read_text().splitlines() if l.strip()]
    return meta, {r['name']: r for r in rows if r['kind'] == 'case'}


def symbols(nm, binary):
    out = subprocess.run([nm, '-n', '--defined-only', '--print-size', str(binary)],
                         capture_output=True, text=True)
    if out.returncode:
        return None
    syms = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in 'tTwW':
            syms.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
    return syms


def function_at(syms, pc):
    if syms is None:
        return None
    for address, size, name in syms:
        if address <= pc < address + size:
            return name
    return None


def outcome(row):
    if row is None:
        return 'not-run'
    if not row['begin'] or row['control_failed']:
        return 'no-reading'
    if row['exit'] == 124:
        return 'no-reading'
    if row['signal']:
        return 'fault'
    if row['exit'] == 0 and row['returned']:
        failing = [l for l in row.get('rc_lines', []) if not l.startswith('repro ')]
        return 'returned-with-error' if failing else 'returned'
    return 'no-reading'


def score(row, build, nm, sites, rootfs):
    o = outcome(row)
    rec = {'outcome': o}
    if o == 'fault' and row['fault']:
        rec['code'] = row['fault']['code']
        # The PC is an offset into the object dladdr named: the case binary, or a library of the
        # guest (libc's memcpy, say), which is read from the platform's rootfs.
        obj = row['fault']['obj']
        rec['object'] = obj
        if obj.endswith('/' + row['name']) or obj == row['name']:
            image = build / row['name']
        elif obj.startswith('/'):
            image = rootfs / obj.lstrip('/')
        else:
            image = None
        rec['function'] = function_at(symbols(nm, image), int(row['fault']['pc_off'], 16)) if image else None
        rec['signal'] = row['signal']
        rec['at_site'] = rec['function'] is not None and rec['function'] in sites
        rec['poison_console'] = bool(row['console_poison'])
    elif o == 'fault':
        rec['at_site'] = False
        rec['note'] = 'signal without a REPRO_FAULT line'
    else:
        rec['at_site'] = False
    if o == 'no-reading':
        rec['tail'] = row['tail'] if row else []
    return rec


def gates_passed(published_meta, spatial_meta):
    """Both runs passed their gates (without a spatial run, the published one did)."""
    passed = lambda meta: bool(meta.get('gate')) and all(meta['gate'].values())
    return passed(published_meta) and (spatial_meta is None or passed(spatial_meta))


def judge(case, published, spatial, nm, rootfs):
    """One case's row: its tiers on the published run and, given a spatial run, on that one too.
    PUBLISHED and SPATIAL are (build dir, rows by name); SPATIAL is None without a spatial run."""
    root = next(r for r in (CORPUS, DEMOS) if (r / case / 'case.json').exists())
    sites = json.loads((root / case / 'case.json').read_text()).get('fault_sites') or []
    p_rec = score(published[1].get(case), published[0], nm, sites, rootfs)
    s_rec = score(spatial[1].get(case), spatial[0], nm, sites, rootfs) if spatial else None
    row = {'case': case, 'qualified': bool(sites), 'fault_sites': sites, 'published': p_rec, 'spatial': s_rec}
    row['mechanism_signed'] = (row['qualified'] and p_rec['at_site'] and p_rec.get('signal') == 'SIGPROT'
                               and p_rec.get('code') == '3' and p_rec.get('poison_console', False))
    # A control that did not read (no-reading, unresolved, not run) cannot support attribution.
    usable = s_rec is not None and s_rec['outcome'] in ('returned', 'fault') and \
        not (s_rec['outcome'] == 'fault' and s_rec.get('function') is None)
    row['attributable'] = (row['qualified'] and p_rec['at_site'] and usable and not s_rec['at_site']) if s_rec else None
    return row


def count(table):
    counts = {'qualified': 0, 'fault_at_site': 0, 'attributable': 0, 'mechanism_signed': 0}
    for row in table:
        if row['qualified']:
            counts['qualified'] += 1
            counts['fault_at_site'] += row['published']['at_site']
            counts['attributable'] += bool(row['attributable'])
            counts['mechanism_signed'] += row['mechanism_signed']
    return counts


def render(table, counts, with_spatial):
    """verdicts.md: one table row per case, then the counts."""
    def cell(x):
        if x is None:
            return 'not run'
        return x['outcome'] + (f" {x.get('function')} code={x.get('code')}" if x['outcome'] == 'fault' else '')
    lines = ['| case | published | spatial | signed | attributable |', '|---|---|---|---|---|']
    for r in table:
        name = r['case'] if r['qualified'] else r['case'] + ' (no qualified trigger)'
        lines.append(f"| {name} | {cell(r['published'])} | {cell(r['spatial'])} | "
                     f"{'yes' if r['mechanism_signed'] else ''} | {'yes' if r['attributable'] else ''} |")
    lines.append('')
    lines.append(f"qualified {counts['qualified']}: fault at site {counts['fault_at_site']}, "
                 f"mechanism-signed {counts['mechanism_signed']}"
                 + (f", attributable {counts['attributable']}" if with_spatial else ''))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--published', type=Path, required=True)
    p.add_argument('--spatial', type=Path)
    p.add_argument('--nm', required=True)
    p.add_argument('--rootfs', type=Path, required=True, help='the guest rootfs, for faults in its libraries')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pub_meta, pub = load(a.published)
    spa_meta, spa = load(a.spatial) if a.spatial else (None, None)
    if not gates_passed(pub_meta, spa_meta):
        raise SystemExit('a run did not pass its gate; nothing to score')
    published = (Path(pub_meta['build']), pub)
    spatial = (Path(spa_meta['build']), spa) if a.spatial else None
    cases = sorted(set(pub_meta['cases']) | set(spa_meta['cases'] if spa_meta else []))
    table = [judge(case, published, spatial, a.nm, a.rootfs) for case in cases]
    counts = count(table)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'verdicts.json').write_text(json.dumps({'published': str(a.published), 'spatial': str(a.spatial),
                                                     'counts': counts, 'cases': table}, indent=1) + '\n')
    lines = render(table, counts, bool(a.spatial))
    (a.out / 'verdicts.md').write_text('\n'.join(lines) + '\n')
    print(lines[-1])


if __name__ == '__main__':
    main()
