#!/usr/bin/env python3
"""Stub board transcripts (BOARDVM_STUB_TRANSCRIPT) for the Caps chain dry run, one per kit, built from REAL boards.

Frame. A real SPEC-chain boot on board 1, bitstream 3 (spec/board/out/519.lbm_r-test/guest/board.txt):
- its rungs b0-stats and b0-hello;
- its vbatch rung start and @@MODTEXT;
- its control step (crc.K0 T1, the same control spec-board.sh runs);
- everything after its spec step: VIRTUAL_STAGED_DONE, RESULT, and the rung's END.

Spec step. What the wrapper prints, as the host test of run.sh printed it (@@SPEC-STAGE, @@CAPS-SUMS OK,
@@SPEC-ARM-BEGIN, @@SPEC-BEGIN). Then the CAMPAIGN kit's own step text from its bitstream-3 boot (t3kN -> capshN,
t3aN -> capsaN; the plans are equal position by position). Then @@SPEC-END, the hash block (the kit's SHA256SUMS) and
@@SPEC-DONE.

Relabelling. The arm tokens in the campaign's @@PERF-BEGIN/END lines are mapped by POSITION, T1 -> Caps, T3 -> MapLife,
T2 -> ObjLife. So the stub's step sequence equals the Caps kit's steps.tsv exactly (checked here). The data under each
label are the campaign arm's. That is right for a parser and chain dry run, and wrong for any measurement, so the
stubs carry a STUB marker line and are never results. The kit's DONE line @@PERF-T3K1-DONE becomes @@PERF-CAPSH1-DONE.
"""
import re, sys
from pathlib import Path
WORK = __import__('os').environ.get('CAPSTONE_WORK', '/mnt/hdd/' + __import__('os').environ.get('USER', '') + '/capstone')

P = Path(WORK + '/perf')
OUT = P / 'T0a/board/dryrun/stubs'
FRAME = Path(WORK + '/spec/board/out/519.lbm_r-test/guest/board.txt')
HOST = P / 'T0a/board/hosttest/pos.out'
CAMP = {'capsh1': 't3-run-b3-e7af48457-b1/main6-t3k1', 'capsh2': 't3-run-b3-e7af48457-b2/main6-t3k2',
        'capsh3': 't3-run-b3-e7af48457-b1/main6-t3k3', 'capsa1': 't3-run-b3-e7af48457-b1/main5-t3a1',
        'capsa2': 't3-run-b3-e7af48457-b2/main5-t3a2', 'capsa3': 't3-run-b3-e7af48457-b1/main5-t3a3',
        'capsa4': 't3-run-b3-e7af48457-b2/main5-t3a4'}
KITDIR = {k: P / 'T0a/kits' / ('heap15' if k.startswith('capsh') else 'other88') / k for k in CAMP}
POS = {'T1': 'Caps', 'T3': 'MapLife', 'T2': 'ObjLife'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = FRAME.read_text(errors='replace').replace('\r', '')
    i0 = frame.index('@@STEP-BEGIN spec\n')
    i1 = frame.index('@@STEP-END spec ')
    head, tail = frame[:i0], frame[frame.index('\n', i1) + 1:]
    host = HOST.read_text()
    pre = host[:host.index('@@SPEC-BEGIN 0 run perfkit 0\n') + len('@@SPEC-BEGIN 0 run perfkit 0\n')]
    bad = 0
    for k, camp in CAMP.items():
        b = (Path(WORK + '') / camp / 'guest/board.txt').read_text(errors='replace').replace('\r', '')
        s0 = b.index('@@STEP-BEGIN pilot\n') + len('@@STEP-BEGIN pilot\n')
        s1 = b.index('@@STEP-END pilot ')
        body, n = re.subn(r'^(@@PERF-(?:BEGIN|END) \d+ \S+ )(T1|T3|T2)( )', lambda m: m[1] + POS[m[2]] + m[3], b[s0:s1],
                          flags=re.M)
        camp_kit = re.search(r'main\d-(t3[ka]\d)', camp)[1]
        body, nd = re.subn(rf'^@@PERF-{camp_kit.upper()}-DONE$', f'@@PERF-{k.upper()}-DONE', body, flags=re.M)
        assert nd == 1, (k, nd)
        sums = (KITDIR[k] / 'SHA256SUMS').read_text()
        pre_k = pre.replace('capsh1', k).replace('run-capsh1.sh a9cf7a545a94fabd',
                                                  f"run-{k}.sh {__import__('hashlib').sha256((KITDIR[k] / 'stage' / f'run-{k}.sh').read_bytes()).hexdigest()[:16]}")
        pre_k = re.sub(r'@@CAPS-SUMS OK \d+ files', f'@@CAPS-SUMS OK {len(sums.splitlines())} files', pre_k)
        spec = (f'@@STEP-BEGIN spec\n@@DRYRUN-STUB {k}: frame 519.lbm_r-test board 1; body {camp} relabelled T1->Caps '
                f'T3->MapLife T2->ObjLife BY POSITION (not a result)\n' + pre_k + body +
                '@@SPEC-END 0 run perfkit 0 rc=0 up0=130.00 up1=1500.00 sys=1,1,1 nbench=1 :: bench cycles=1 instret=1 entries=1\n'
                '@@SPEC-HASH-BEGIN perfkit\n' + sums + '@@SPEC-HASH-END perfkit\n@@SPEC-ARM-END perfkit\n'
                f'@@SPEC-DONE caps-{k} board\n@@STEP-END spec 0\n')
        (OUT / f'{k}.board.txt').write_text(head + spec + tail)
        # the stub's step sequence must be the Caps kit's plan, line for line
        got = re.findall(r'^@@PERF-BEGIN (\d+) (\S+) (\S+) (\d+) K(\d+)$', body, re.M)
        want = [tuple(l.split('\t')[:5]) for l in (KITDIR[k] / 'steps.tsv').read_text().splitlines()[1:]]
        same = got == want
        bad += not same
        print(f'{k}: {n} PERF lines relabelled, {len(got)} BEGIN, plan {"EQUALS" if same else "DIFFERS FROM"} steps.tsv '
              f'({len(want)}) -> {OUT / (k + ".board.txt")}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
