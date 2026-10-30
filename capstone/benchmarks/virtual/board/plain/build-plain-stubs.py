#!/usr/bin/env python3
"""Stub board transcripts (BOARDVM_STUB_TRANSCRIPT) for the Plain chain dry run, one per kit, built from REAL boards.
The Caps lane's build-stubs.py, adapted; every stub carries a @@DRYRUN-STUB line and is never a result.

Frame. A real SPEC-chain boot on board 1, bitstream 3 (spec/board/out/519.lbm_r-test/guest/board.txt): its rungs,
its control step (crc.K0 T1, the control spec-board.sh runs) and everything after its spec step.

Spec step. The wrapper's own lines (@@SPEC-STAGE, @@PLAIN-SUMS OK, @@SPEC-ARM-BEGIN, @@SPEC-BEGIN), the kit's
`ls | grep bench` lines, the probe block, then one block per line of the kit's steps.tsv:
- a T1 or T3 step takes the CAMPAIGN's bitstream-3 block of the same program, arm and process (main5-t3a*, main6-t3k*),
  with its @@PERF-BEGIN/END renumbered to this kit's step and K;
- a Plain step takes the campaign's T2 block of the same program and process, relabelled Plain, with the harness's
  `BENCH-INFO <id> source=user-csr ...` line and one `BENCH-KERN` line per BENCH line added (irq = ns / 10 ms), and an
  END line without a dmesg quote, as a Plain process prints. Plain ctl-empty reads instret 600 and irq 0; Plain
  ctl-loop reads instret 600 + 20,000,002 + irq x 4,300, and the stub probe's gap_instret median is 4,300, so the
  pilot gate's instrument check has a consistent input. These numbers are invented for the parser; none is a reading.
Then the kit's DONE line, @@SPEC-END, the hash block (the kit's SHA256SUMS) and @@SPEC-DONE.

Pilot negatives (pp1 only):
- neg-trapped: what the bitstream-3 monitor (csbi-batch d9da397, handle_exception) prints when a user rdcycle traps:
  `ILLX:0000E003`, `MTVL:C0002573`, `MEPC:...`, `INJS:00000002`, then the probe's `PLAINPROBE access=trapped
  at=rdcycle` and rc 2; every Plain process then prints the monitor lines again and `counters=absent` with zero counts.
- neg-wedge: the monitor lines and `INJL:` (the injection would land on the faulting pc), then nothing: the step never
  ends, the transcript stops (the driver's idle cut, BAKED_IDLE_S=200), and the step's text is only the last
  heartbeat's tail.
- neg-hold: the counters work, but Plain ctl-loop's excess is 2x the probe's gap_instret (the instruments disagree).
"""
import base64, hashlib, os, re, sys
from pathlib import Path

WORK = Path(os.environ.get('CAPSTONE_WORK', '/mnt/hdd/' + os.environ.get('USER', '') + '/capstone'))
T0B = WORK / 'perf/T0b'
OUT = T0B / 'board/dryrun/stubs'
FRAME = WORK / 'spec/board/out/519.lbm_r-test/guest/board.txt'
CAMP = [WORK / f't3-run-b3-e7af48457-b{b}/{r}/guest/board.txt' for b, r in
        [(1, 'main5-t3a1'), (2, 'main5-t3a2'), (1, 'main5-t3a3'), (2, 'main5-t3a4'),
         (1, 'main6-t3k1'), (2, 'main6-t3k2'), (1, 'main6-t3k3')]]
KITS = [('pp1', T0B / 'kits-pilot/pp1')] + [(f'p{i}', T0B / f'kits/p{i}') for i in range(1, 7)]
KINS = 4300
MON_TRAP = 'ILLX:0000E003\nMTVL:C0002573\nMEPC:000105A8\nINJS:00000002\n'


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


def plain_block(inner, prog, trapped=False, hold=False):
    out = []
    sid = re.search(r'BENCH-INFO (\S+) counters=', inner)[1]
    for line in inner.splitlines():
        if line.startswith('BENCH-INFO ') and ' counters=' in line:
            if trapped:
                out.append(MON_TRAP.rstrip('\n'))
            out.append(f'BENCH-INFO {sid} source=user-csr rdcycle_rdinstret={"trapped" if trapped else "ok"} proc_interrupts=ok')
            out.append(line.replace('counters=present', 'counters=absent') if trapped else line)
            continue
        m = re.match(r'BENCH (\S+) it=(\S+) cycles=(\d+) instret=(\d+) entries=(\d+) ns=(-?\d+) check=(\S+)(.*)$', line)
        if m:
            irq = max(0, round(int(m[6]) / 1e7))
            ins = int(m[4])
            if prog == 'ctl-empty':
                irq, ins = 0, 600
            elif prog == 'ctl-loop':
                ins = 600 + 20_000_002 + irq * KINS * (2 if hold else 1)
            cyc = int(m[3])
            if trapped:
                cyc = ins = irq = 0
            out.append(f'BENCH {m[1]} it={m[2]} cycles={cyc} instret={ins} entries={irq} ns={m[6]} check={m[7]}{m[8]}')
            out.append(f'BENCH-KERN {m[1]} it={m[2]} timer={irq} irq={irq} minflt=0 majflt=0 nvcsw=0 nivcsw=0')
            continue
        out.append(line)
    return '\n'.join(out) + '\n'


PROBE_OK = ('@@PLAINPROBE-BEGIN\nPLAINPROBE access=ok cycle_advances=1 instret_advances=1\n'
            'PLAINPROBE spin cycles=50000020 instret=33000000 ns=2000001000 cycles_per_us=25.000 loop_iters=8000000 '
            'gaps=200 proc_interrupts_delta=201\n'
            f'PLAINPROBE gap_cycles min=5000 p10=6000 median=9000 p90=12000 max=40000 mean=9500\n'
            f'PLAINPROBE gap_instret min=4000 p10=4200 median={KINS} p90=4500 max=9000 mean=4400\n'
            'PLAINPROBE fault pages=256 minflt_delta=257 cycles_min=7000 cycles_median=8000 cycles_p90=9000 '
            'instret_min=3200 instret_median=3276 instret_p90=3400\nPLAINPROBE done\n@@PLAINPROBE-END rc=0\n')
PROBE_TRAP = '@@PLAINPROBE-BEGIN\n' + MON_TRAP + 'PLAINPROBE access=trapped at=rdcycle\n@@PLAINPROBE-END rc=2\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = FRAME.read_text(errors='replace').replace('\r', '')
    i0 = frame.index('@@STEP-BEGIN spec\n')
    i1 = frame.index('@@STEP-END spec ')
    head, tail = frame[:i0], frame[frame.index('\n', i1) + 1:]
    blocks = campaign_blocks()
    bad = 0
    variants = [(k, kit, None) for k, kit in KITS] + [('pp1', KITS[0][1], v) for v in ('neg-trapped', 'neg-wedge', 'neg-hold')]
    for k, kit, neg in variants:
        steps = [l.split('\t') for l in (kit / 'steps.tsv').read_text().splitlines()[1:]]
        sums = (kit / 'SHA256SUMS').read_text()
        run_sha = hashlib.sha256((kit / 'stage' / f'run-{k}.sh').read_bytes()).hexdigest()[:16]
        body = ['bench_cycles', 'bench_entries', 'bench_instret']
        body.append((PROBE_TRAP if neg in ('neg-trapped', 'neg-wedge') else PROBE_OK).rstrip('\n'))
        missing = []
        for n, prog, arm, proc, K, args in steps:
            src = blocks.get((prog, 'T2' if arm == 'Plain' else arm, proc))
            if src is None:
                missing.append((prog, arm, proc)); continue
            inner, end = src
            body.append(f'@@PERF-BEGIN {n} {prog} {arm} {proc} K{K}')
            if arm == 'Plain':
                body.append(plain_block(inner, prog, trapped=neg == 'neg-trapped', hold=neg == 'neg-hold').rstrip('\n'))
                body.append(f'@@PERF-END {n} {prog} {arm} {proc} K{K} rc=0 stderr_lines=0')
            else:
                body.append(inner.rstrip('\n'))
                body.append(f'@@PERF-END {n} {prog} {arm} {proc} K{K} {end}')
        body.append(f'@@PERF-{k.upper()}-DONE')
        text = '\n'.join(body) + '\n'
        pre = (f'@@SPEC-STAGE plain-{k} board arms=perfkit commands=1 validations=0\n'
               f'@@PLAIN-SUMS OK {len(sums.splitlines())} files verified on the board; run-{k}.sh {run_sha}\n'
               '@@SPEC-ARM-BEGIN perfkit\n@@SPEC-BEGIN 0 run perfkit 0\n')
        post = ('@@SPEC-END 0 run perfkit 0 rc=0 up0=130.00 up1=1500.00 sys=1,1,1 nbench=1 :: bench cycles=1 instret=1 entries=1\n'
                '@@SPEC-HASH-BEGIN perfkit\n' + sums + '@@SPEC-HASH-END perfkit\n@@SPEC-ARM-END perfkit\n'
                f'@@SPEC-DONE plain-{k} board\n')
        mark = f'@@DRYRUN-STUB {k}{" " + neg if neg else ""}: frame 519.lbm_r-test board 1; Plain blocks are campaign T2 blocks relabelled (not a result)\n'
        if neg == 'neg-wedge':
            partial = pre + '\n'.join(body[:4]) + '\n'
            hb = base64.b64encode(partial[-240:].encode()).decode()
            spec = (f'@@STEP-BEGIN spec\n{mark}@@HB 1 out={len(partial)}B/{partial.count(chr(10))}L tail={hb}\n'
                    + MON_TRAP.replace('INJS:00000002', 'INJL:00000002\nMEPC:000105A8'))
            stub = head + spec          # nothing after: the core spins in M, the driver's idle limit ends the session
        else:
            spec = '@@STEP-BEGIN spec\n' + mark + pre + text + post + '@@STEP-END spec 0\n'
            stub = head + spec + tail
        name = f'{k}.board.txt' if not neg else f'{k}-{neg}.board.txt'
        (OUT / name).write_text(stub)
        got = re.findall(r'^@@PERF-BEGIN (\d+) (\S+) (\S+) (\d+) K(\d+)$', text, re.M)
        want = [tuple(s[:5]) for s in steps]
        ok = got == want and not missing
        bad += not ok and neg is None
        print(f'{name}: {len(got)} steps, plan {"EQUALS" if got == want else "DIFFERS FROM"} steps.tsv ({len(want)}), '
              f'missing campaign blocks {len(missing)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
