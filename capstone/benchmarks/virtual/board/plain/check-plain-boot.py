#!/usr/bin/env python3
"""Completeness and validity gate for ONE Plain kit boot run through the SPEC chain (run-stage.py's out dir), and, with
--pilot, the PILOT GATE that decides whether the main kits may run at all. The Caps lane's check-caps-boot.py, adapted.

  check-plain-boot.py OUT_DIR KIT_DIR [--runs runs.tsv] [--pilot]

OUT_DIR is chain-board's <root>/perf-plain-<kit>; KIT_DIR is perf/T0b/kits/p<k> or perf/T0b/kits-pilot/pp1.
parse-board.py alone cannot see a boot that ended early (it reads the processes that are there and exits 0), so C5
compares the parsed processes with the kit's plan.

VOID (exit 1) unless every one holds:
  C1 run-stage's log says `RUN control PASSED` (the known-good crc.K0 T1 control exited 0, no fault);
  C2 the step text has `@@PLAIN-SUMS OK`, and its SPEC-HASH block equals the kit's SHA256SUMS line for line;
  C3 the kit's DONE line (@@PERF-<KIT>-DONE) and @@SPEC-DONE plain-<kit> are present;
  C4 parse-board.py exits 0 on the step text (every process: END, rc 0, W+I BENCH lines, counters=present, no
     VERIFY-FAILED, failed=0, and one BENCH-KERN line per BENCH line in every Plain process);
  C5 the parsed processes are EXACTLY the kit's steps.tsv: same (program, arm, proc) set, same count;
  C6 no @@STEP-BOUND, [runner] TIMEOUT, kernel oops or OOM in the step text.

PILOT GATE (--pilot; checked even when the boot is VOID, and printed first). The main kits run only on exit 0.
  STOP, exit 3: the user-mode counters are not usable on this board.
    P1 the monitor's illegal-instruction report is absent from the WHOLE transcript (guest/board.txt): no `ILLX:`,
       `INJS:` or `INJL:` line (the bitstream-3 monitor prints `ILLX:0000E003`, `MTVL:<instruction word>`, `MEPC:`
       and then `INJS:00000002` when it injects the trap into Linux, or `INJL:` and spins when it cannot). None of
       the 22 passing campaign and SPEC-chain transcripts on bitstream 3 carries one.
    P2 the probe ran and read the counters: `@@PLAINPROBE-BEGIN`, `PLAINPROBE access=ok cycle_advances=1
       instret_advances=1`, `PLAINPROBE done` and `@@PLAINPROBE-END rc=0` in the step text.
  HOLD, exit 4: the counters work, but the two instruments do not agree (the STATUS first-boot rule).
    P3 the spin line: cycles_per_us in [24.5, 25.5] (the 25 MHz core against CLOCK_MONOTONIC); gaps per second
       (gaps / (cycles / 25e6)) in [90, 110] (HZ=100); |gaps - proc_interrupts_delta| <= 3.
    P4 the Plain ctl-loop iterations: k_ins = sum(instret - median Plain ctl-empty instret - 20,000,002) / sum(irq)
       within ±15 % of the probe's gap_instret median, and every iteration's own ratio within ±10 % of k_ins.
Exit 0 = VALID (and, with --pilot, gate PASS); 1 VOID; 2 no data; 3 pilot STOP; 4 pilot HOLD.
"""
import argparse, csv, os, re, statistics as st, subprocess, sys
from pathlib import Path

WORK = Path(os.environ.get('CAPSTONE_WORK', '/mnt/hdd/' + os.environ.get('USER', '') + '/capstone'))
PB = Path(os.environ.get('PLAIN_PARSE_BOARD', str(WORK / 'wt-t0b/capstone/benchmarks/virtual/parse-board.py')))
N_LOOP = 10_000_000


def pilot_gate(text, board, runs, kern):
    """-> (exit code, [lines])"""
    out, stop, hold = [], [], []
    mon = re.findall(r'(ILLX|INJS|INJL|MTVL|MEPC):([0-9A-F]{8})', board)
    if any(t in ('ILLX', 'INJS', 'INJL') for t, _ in mon):
        stop.append('P1 the monitor reported an illegal instruction: ' + ' '.join(f'{t}:{v}' for t, v in mon[:8]))
    probe = re.search(r'^@@PLAINPROBE-BEGIN$(.*?)^@@PLAINPROBE-END rc=(\d+)$', text, re.M | re.S)
    if not probe or 'PLAINPROBE access=ok cycle_advances=1 instret_advances=1' not in probe[1] \
            or 'PLAINPROBE done' not in probe[1] or probe[2] != '0':
        got = probe[1].strip().splitlines()[:2] if probe else ['no @@PLAINPROBE-BEGIN/END pair']
        stop.append(f'P2 the probe did not read the counters ({" | ".join(got)[:160]})')
    if stop:
        return 3, ['PILOT STOP (the user-mode counters are not usable): ' + s for s in stop]
    body = probe[1]
    sp = re.search(r'PLAINPROBE spin cycles=(\d+) instret=\d+ ns=\d+ cycles_per_us=([\d.]+) loop_iters=\d+ gaps=(\d+) '
                   r'proc_interrupts_delta=(\d+)', body)
    gi = re.search(r'PLAINPROBE gap_instret min=\d+ p10=\d+ median=(\d+)', body)
    if not sp or not gi:
        hold.append('P3 no spin or gap_instret line in the probe')
    else:
        cyc, cpu, gaps, pid = int(sp[1]), float(sp[2]), int(sp[3]), int(sp[4])
        gps = gaps / (cyc / 25e6) if cyc else 0
        out.append(f'probe: cycles_per_us={cpu} gaps={gaps} ({gps:.1f}/s) proc_interrupts_delta={pid} '
                   f'gap_instret_median={gi[1]}')
        if not 24.5 <= cpu <= 25.5:
            hold.append(f'P3 cycles_per_us {cpu} outside [24.5, 25.5]')
        if not 90 <= gps <= 110:
            hold.append(f'P3 gaps per second {gps:.1f} outside [90, 110]')
        if abs(gaps - pid) > 3:
            hold.append(f'P3 gaps {gaps} vs proc_interrupts_delta {pid}: the detector and the kernel disagree')
    rows = [r for r in runs if r['arm'] == 'Plain' and not r['it'].startswith('w')]
    empty = [int(r['instret']) for r in rows if r['program'] == 'ctl-empty']
    loop = [r for r in rows if r['program'] == 'ctl-loop']
    if not empty or not loop or gi is None:
        hold.append('P4 no Plain ctl-empty/ctl-loop rows or no gap_instret: the instruments cannot be compared')
    else:
        e = st.median(empty)
        ex, irq = [], []
        for r in loop:
            k = kern.get((r['step'], r['it']))
            if k is None:
                hold.append(f'P4 ctl-loop step {r["step"]} it {r["it"]} has no BENCH-KERN row'); continue
            ex.append(int(r['instret']) - e - (2 * N_LOOP + 2)); irq.append(int(k['irq']))
        if sum(irq) <= 0:
            hold.append('P4 Plain ctl-loop saw no interrupts')
        else:
            k_ins = sum(ex) / sum(irq)
            per = [x / i for x, i in zip(ex, irq) if i]
            out.append(f'ctl-loop: k_ins={k_ins:.1f} instr/irq over {len(irq)} iterations (per-iteration '
                       f'{min(per):.1f}..{max(per):.1f}), probe gap_instret median {gi[1]}')
            if not 0.85 * int(gi[1]) <= k_ins <= 1.15 * int(gi[1]):
                hold.append(f'P4 k_ins {k_ins:.1f} not within 15 % of the probe gap_instret median {gi[1]}')
            if any(abs(p - k_ins) > 0.10 * k_ins for p in per):
                hold.append(f'P4 a ctl-loop iteration\'s excess/irq leaves ±10 % of k_ins ({min(per):.1f}..{max(per):.1f})')
    if hold:
        return 4, out + ['PILOT HOLD (the instruments disagree; the main kits wait for an explanation): ' + h for h in hold]
    return 0, out + ['PILOT PASS: counters readable, no monitor ILLX, the two instruments agree']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path); ap.add_argument('kit', type=Path); ap.add_argument('--runs', type=Path)
    ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()
    k = a.kit.name
    spec, log, board = a.out / 'spec.txt', Path(str(a.out) + '.log'), a.out / 'guest' / 'board.txt'
    if not board.exists():
        print(f'NO DATA: {board} missing'); return 2
    btext = board.read_text(errors='replace').replace('\r', '')
    text = spec.read_text(errors='replace') if spec.exists() else ''
    text = re.sub(r'@@HB \d+ out=\S* tail=[A-Za-z0-9+/=]*\n', '', text)
    bad = []
    if not spec.exists():
        bad.append('C0 no spec.txt (the spec step never ended: a wedge or an idle cut)')
    lg = log.read_text(errors='replace') if log.exists() else ''
    if 'RUN control PASSED' not in lg:
        bad.append(f'C1 control not PASSED ({log})')
    sums = (a.kit / 'SHA256SUMS').read_text().splitlines()
    m = re.search(r'^@@SPEC-HASH-BEGIN perfkit\n(.*?)^@@SPEC-HASH-END perfkit$', text, re.M | re.S)
    if '@@PLAIN-SUMS OK' not in text or not m or m.group(1).splitlines() != sums:
        bad.append('C2 board-side kit hashes absent or not equal to SHA256SUMS')
    if not re.search(rf'^@@PERF-{k.upper()}-DONE$', text, re.M) or not re.search(rf'^@@SPEC-DONE plain-{k} ', text, re.M):
        bad.append(f'C3 @@PERF-{k.upper()}-DONE or @@SPEC-DONE absent')
    runs = a.runs or (a.out / f'runs-{k}.tsv')
    rows, kern = [], {}
    if spec.exists():
        p = subprocess.run([sys.executable, str(PB), str(spec), '--board', '1', '--boot', a.out.name, '--out', str(runs)],
                           capture_output=True, text=True)
        if p.returncode:
            bad.append(f'C4 parse-board rc={p.returncode}: {(p.stdout + p.stderr).strip()[-300:]}')
        if runs.exists():
            rows = list(csv.DictReader(open(runs), delimiter='\t'))
        kp = Path(str(runs) + '.kern.tsv')
        if kp.exists():
            kern = {(r['step'], r['it']): r for r in csv.DictReader(open(kp), delimiter='\t')}
    want = [(r['program'], r['arm'], r['proc']) for r in csv.DictReader(open(a.kit / 'steps.tsv'), delimiter='\t')]
    got = {(r['program'], r['arm'], r['proc']) for r in rows}
    if got != set(want) or len(want) != len(set(want)):
        bad.append(f'C5 parsed processes {len(got)} != plan {len(want)} '
                   f'(missing {len(set(want) - got)}, extra {len(got - set(want))})')
    if re.search(r'@@STEP-BOUND|\[runner\] TIMEOUT|Oops|Unable to handle|BUG:|Out of memory|oom-kill', text):
        bad.append('C6 a bound, timeout or kernel oops/OOM in the step text')
    gate = 0
    if a.pilot:
        gate, lines = pilot_gate(text, btext, rows, kern)
        for l in lines:
            print(l)
    print(f'{k}: {"VALID" if not bad else "VOID"}: {len(got)} of {len(want)} processes parsed')
    for b in bad:
        print('  ' + b)
    if gate:
        return gate
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
