#!/usr/bin/env python3
"""Completeness and validity gate for ONE ObjBounds kit boot run through the chain (run-stage.py's out dir).

  check-objb-boot.py OUT_DIR KIT_DIR [--runs runs.tsv] [--board 2]

OUT_DIR is the chain's <root>/perf-objb-<kit>; KIT_DIR is perf/objb/kits/<group>/<kit> (steps.tsv, SHA256SUMS).
parse-board.py alone cannot see a boot that ended early: it reads the processes that are there and exits 0. This gate
is what makes such a boot VOID. The same rules as the Caps and Plain gates.

VOID unless every one holds:
  C1 run-stage's log says `RUN control PASSED` (the known-good crc.K0 T1 control exited 0, no fault);
  C2 the step text has `@@OBJB-SUMS OK`, and its SPEC-HASH block equals the kit's SHA256SUMS line for line;
  C3 the kit's own DONE line (@@PERF-<KIT>-DONE) and @@SPEC-DONE are present;
  C4 parse-board.py exits 0 on the step text (every process: END, rc 0, W+I BENCH lines, no VERIFY-FAILED, failed=0);
  C5 the parsed processes are EXACTLY the kit's steps.tsv: same (program, arm, proc) set, same count;
  C6 no @@STEP-BOUND, kernel oops/OOM, or [runner] TIMEOUT in the step text.
Exit 0 = VALID, 1 = VOID (each failed check listed), 2 = no data (the out dir or its spec.txt is missing).
"""
import argparse, csv, re, subprocess, sys
from pathlib import Path

PB = Path(__file__).resolve().parents[2] / 'parse-board.py'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path); ap.add_argument('kit', type=Path); ap.add_argument('--runs', type=Path)
    ap.add_argument('--board', default='2')
    a = ap.parse_args()
    k = a.kit.name
    spec, log = a.out / 'spec.txt', Path(str(a.out) + '.log')
    if not spec.exists():
        print(f'NO DATA: {spec} missing'); return 2
    text = spec.read_text(errors='replace')
    bad = []
    lg = log.read_text(errors='replace') if log.exists() else ''
    if 'RUN control PASSED' not in lg:
        bad.append(f'C1 control not PASSED ({log})')
    sums = (a.kit / 'SHA256SUMS').read_text().splitlines()
    m = re.search(r'^@@SPEC-HASH-BEGIN perfkit\n(.*?)^@@SPEC-HASH-END perfkit$', text, re.M | re.S)
    if '@@OBJB-SUMS OK' not in text or not m or m.group(1).splitlines() != sums:
        bad.append('C2 board-side kit hashes absent or not equal to SHA256SUMS')
    if not re.search(rf'^@@PERF-{k.upper()}-DONE$', text, re.M) or not re.search(rf'^@@SPEC-DONE objb-{k} ', text, re.M):
        bad.append(f'C3 @@PERF-{k.upper()}-DONE or @@SPEC-DONE absent')
    runs = a.runs or (a.out / f'runs-{k}.tsv')
    p = subprocess.run([sys.executable, str(PB), str(spec), '--board', a.board, '--boot', a.out.name, '--out', str(runs)],
                       capture_output=True, text=True)
    if p.returncode:
        bad.append(f'C4 parse-board rc={p.returncode}: {(p.stdout + p.stderr).strip()[-300:]}')
    want = [(r['program'], r['arm'], r['proc']) for r in csv.DictReader(open(a.kit / 'steps.tsv'), delimiter='\t')]
    got = set()
    if runs.exists():
        got = {(r['program'], r['arm'], r['proc']) for r in csv.DictReader(open(runs), delimiter='\t')}
    if got != set(want) or len(want) != len(set(want)):
        bad.append(f'C5 parsed processes {len(got)} != plan {len(want)} '
                   f'(missing {len(set(want) - got)}, extra {len(got - set(want))})')
    if re.search(r'@@STEP-BOUND|\[runner\] TIMEOUT|Oops|Unable to handle|BUG:|Out of memory|oom-kill', text):
        bad.append('C6 a bound, timeout or kernel oops/OOM in the step text')
    print(f'{k}: {"VALID" if not bad else "VOID"}: {len(got)} of {len(want)} processes parsed')
    for b in bad:
        print('  ' + b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
