#!/usr/bin/env python3
"""Stub board transcripts (BOARDVM_STUB_TRANSCRIPT) for the ObjBounds dry run, one per kit, built from REAL boards.
The Plain lane's build-plain-stubs.py, adapted; every stub carries a @@DRYRUN-STUB line and is never a result.

Frame. A real SPEC-chain boot on bitstream 3 (spec/board/out/519.lbm_r-test/guest/board.txt): its rungs, its control
step (crc.K0 T1, the control spec-board.sh runs) and everything after its spec step.

Spec step. The wrapper's own lines (@@SPEC-STAGE, @@OBJB-SUMS OK, @@SPEC-ARM-BEGIN, @@SPEC-BEGIN), then one block per
line of the kit's steps.tsv: the CAMPAIGN's bitstream-3 block of the same program, arm and process (main5-t3a*,
main6-t3k*), with its @@PERF-BEGIN/END renumbered to this kit's step and K. Then the kit's DONE line, @@SPEC-END, the
hash block (the kit's SHA256SUMS) and @@SPEC-DONE. The data are bitstream 3's: right for a parser, gate and chain dry
run, meaningless as an ObjBounds reading.

Pilot negatives (objbh1 only):
- neg-void: the step text is cut after the 20th process (no DONE, no hash block, no @@SPEC-END): the boot ended early.
- neg-control: the known-good control's step ends with rc 1.
"""
import hashlib, os, re, sys
from pathlib import Path

WORK = Path(os.environ.get('CAPSTONE_WORK', '/mnt/hdd/' + os.environ.get('USER', '') + '/capstone'))
OBJB = WORK / 'perf/objb'
OUT = Path(os.environ.get('OB_STUBS', str(OBJB / 'board/dryrun/stubs')))
FRAME = WORK / 'spec/board/out/519.lbm_r-test/guest/board.txt'
CAMP = [WORK / f't3-run-b3-e7af48457-b{b}/{r}/guest/board.txt' for b, r in
        [(1, 'main5-t3a1'), (2, 'main5-t3a2'), (1, 'main5-t3a3'), (2, 'main5-t3a4'),
         (1, 'main6-t3k1'), (2, 'main6-t3k2'), (1, 'main6-t3k3')]]
KITS = [(f'objbh{i}', OBJB / f'kits/heap15/objbh{i}') for i in (1, 2, 3)] + \
       [(f'objba{i}', OBJB / f'kits/other88/objba{i}') for i in (1, 2, 3, 4)]


def campaign_blocks():
    blocks = {}
    for f in CAMP:
        b = f.read_text(errors='replace').replace('\r', '')
        body = b[b.index('@@STEP-BEGIN pilot\n'):b.index('@@STEP-END pilot ')]
        body = re.sub(r'@@HB \d+ out=\S* tail=[A-Za-z0-9+/=]*\n', '', body)
        for m in re.finditer(r'^@@PERF-BEGIN \d+ (\S+) (T[123]) (\d+) K\d+\n(.*?)^@@PERF-END \d+ \S+ \S+ \d+ K\d+ (rc=[^\n]*)$',
                             body, re.M | re.S):
            blocks.setdefault((m[1], m[2], m[3]), (m[4], m[5]))
    return blocks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = FRAME.read_text(errors='replace').replace('\r', '')
    i0 = frame.index('@@STEP-BEGIN spec\n')
    i1 = frame.index('@@STEP-END spec ')
    head, tail = frame[:i0], frame[frame.index('\n', i1) + 1:]
    assert head.count('@@STEP-END control 0\n') == 1
    blocks = campaign_blocks()
    bad = 0
    variants = [(k, kit, None) for k, kit in KITS] + [('objbh1', KITS[0][1], v) for v in ('neg-void', 'neg-control')]
    for k, kit, neg in variants:
        steps = [l.split('\t') for l in (kit / 'steps.tsv').read_text().splitlines()[1:]]
        sums = (kit / 'SHA256SUMS').read_text()
        run_sha = hashlib.sha256((kit / 'stage' / f'run-{k}.sh').read_bytes()).hexdigest()[:16]
        body, missing = [], []
        for n, prog, arm, proc, K, args in steps:
            src = blocks.get((prog, arm, proc))
            if src is None:
                missing.append((prog, arm, proc)); continue
            inner, end = src
            body.append(f'@@PERF-BEGIN {n} {prog} {arm} {proc} K{K}')
            body.append(inner.rstrip('\n'))
            body.append(f'@@PERF-END {n} {prog} {arm} {proc} K{K} {end}')
        pre = (f'@@SPEC-STAGE objb-{k} board arms=perfkit commands=1 validations=0\n'
               f'@@OBJB-SUMS OK {len(sums.splitlines())} files verified on the board; run-{k}.sh {run_sha}\n'
               '@@SPEC-ARM-BEGIN perfkit\n@@SPEC-BEGIN 0 run perfkit 0\n')
        post = ('@@SPEC-END 0 run perfkit 0 rc=0 up0=130.00 up1=1500.00 sys=1,1,1 nbench=1 :: bench cycles=1 instret=1 entries=1\n'
                '@@SPEC-HASH-BEGIN perfkit\n' + sums + '@@SPEC-HASH-END perfkit\n@@SPEC-ARM-END perfkit\n'
                f'@@SPEC-DONE objb-{k} board\n')
        mark = f'@@DRYRUN-STUB {k}{" " + neg if neg else ""}: frame 519.lbm_r-test; blocks are the bitstream-3 campaign\'s (not a result)\n'
        h = head
        if neg == 'neg-void':
            text = '\n'.join(body[:60]) + '\n'          # 20 processes, then the boot ended
            spec = '@@STEP-BEGIN spec\n' + mark + pre + text + '@@STEP-END spec 0\n'
        else:
            text = '\n'.join(body) + f'\n@@PERF-{k.upper()}-DONE\n'
            spec = '@@STEP-BEGIN spec\n' + mark + pre + text + post + '@@STEP-END spec 0\n'
            if neg == 'neg-control':
                h = head.replace('@@STEP-END control 0\n', '@@STEP-END control 1\n')
        name = f'{k}.board.txt' if not neg else f'{k}-{neg}.board.txt'
        (OUT / name).write_text(h + spec + tail)
        got = re.findall(r'^@@PERF-BEGIN (\d+) (\S+) (\S+) (\d+) K(\d+)$', text, re.M)
        want = [tuple(s[:5]) for s in steps]
        ok = got == want and not missing
        bad += not ok and neg is None
        print(f'{name}: {len(got)} steps, plan {"EQUALS" if got == want else "DIFFERS FROM"} steps.tsv ({len(want)}), '
              f'missing campaign blocks {len(missing)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
