#!/usr/bin/env python3
"""The Plain arm's kernel-time correction, per boot, from that boot's own controls.

  plain-correct.py <runs.tsv> <runs.tsv.kern.tsv> --out <corrected.tsv> --probe <transcript of the same boot>
                   [--subtract-empty] [--fault-cycles C --fault-instret I] [--report <file>] [--ols-cycles]

The Plain arm (a native RISC-V process) counts with user-mode rdcycle/rdinstret, so its readings INCLUDE every kernel
entry inside the timed region: timer ticks (HZ=100), other interrupts and page faults. The Capstone arms count only
inside CSRUNV windows, and their escapes (a quantum, ~2e6 cycles on bitstream 3, not every tick) leave Linux's work
outside. This script estimates, per boot, what one interrupt adds to a Plain reading and removes it:

  k_ins  = sum over the boot's Plain ctl-loop timed iterations of (instret - median Plain ctl-empty instret - (2N+2))
           divided by the sum of their interrupt counts (irq= in BENCH-KERN). ctl/loop's user instructions are exactly
           2N+2 above ctl/empty's, so the excess is kernel instructions.
  k_cyc  = the probe's median gap (PLAINPROBE gap_cycles median= in the same boot's transcript): ~100 gaps per
           second, each one interrupt seen between two back-to-back rdcycle reads. The least-squares slope of ctl-loop
           cycles on irq is printed beside it with its standard error, as a cross-check only: ctl-loop iterations see
           ~100 interrupts +-1, so one interrupt of leverage under thousands of cycles of cache noise leaves the slope
           nearly unidentified. --ols-cycles uses the slope instead (no probe), and says so.
  The probe's gap_instret median is printed beside k_ins: two instruments for the same handler.

Plain rows are then rewritten: instret -= irq*k_ins, cycles -= irq*k_cyc (and minflt*fault costs when given). Both
raw and corrected readings are reported by running perf-analyze.py on the input and on this output. The correction
removes the tick handler's own cycles and the refill a TINY working set pays; the refill a real program pays after a
tick stays in, so the corrected Plain reading is still an upper bound on interrupt-free cycles, and T/Plain from it a
LOWER bound on the cost of Capstone; T/Plain from the raw reading is lower still. Read the two as a bracket.

Page faults: each fault in a Plain window costs the probe's `PLAINPROBE fault ... cycles_median= ...
instret_median=` (a first-touch minor fault, same boot), applied automatically with --probe; --fault-cycles and
--fault-instret override it. Plain's heap programs fault 60-470 times per iteration inside the window (QEMU).

--raw writes the rows with NO correction, only the control rename below, so that perf-analyze.py can report the raw
readings of several boots together (it refuses a program seen in two boots, and every kit carries the controls).

--subtract-empty also subtracts, per boot and arm, the median ctl-empty cycles and instret (each arm's own bracket
cost: ~0.3e6 cycles in the Capstone arms, a few hundred in Plain) from every row of that arm.

Control rows (ctl-empty, ctl-loop) are renamed <name>@<board>/<boot>, because every kit carries them and
perf-analyze.py refuses a program seen in two boots.

Exit 1 when a boot with Plain rows lacks Plain ctl-loop or ctl-empty rows, when a Plain row has no BENCH-KERN
counterpart, or when k_ins is not positive: "no calibration" is an error, never a zero correction.
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict

N_LOOP = 10_000_000


def die(msg):
    print(f'plain-correct: {msg}', file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs')
    ap.add_argument('kern')
    ap.add_argument('--out', required=True)
    ap.add_argument('--subtract-empty', action='store_true')
    ap.add_argument('--fault-cycles', type=float, default=None)
    ap.add_argument('--fault-instret', type=float, default=None)
    ap.add_argument('--report')
    ap.add_argument('--probe', help='the boot transcript holding its PLAINPROBE lines (one boot per invocation)')
    ap.add_argument('--ols-cycles', action='store_true', help='k_cyc from the ctl-loop slope (no probe); flagged')
    ap.add_argument('--raw', action='store_true', help='no correction: only rename the controls per boot')
    ap.add_argument('--k-ins-median', action='store_true',
                    help='Plain addendum 1 rule 2: k_ins = median over ctl-loop iterations of excess/irq (run on '
                         'plain-exclude.py output), with the 15 %% probe and 10 %% per-iteration checks printed')
    ap.add_argument('--k-cyc', type=float, default=None,
                    help='k_cyc given, not read from this boot\'s probe (FLAGGED in the report): for a boot whose '
                         'probe was itself disturbed')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.runs), delimiter='\t'))
    if not rows:
        die(f'no rows in {a.runs}')
    fields = list(rows[0].keys())
    kern = {}
    for k in csv.DictReader(open(a.kern), delimiter='\t'):
        kern[(k['board'], k['boot'], k['step'], k['it'])] = k
    def key(r):
        return (r['board'], r['boot'], r['step'], r['it'])
    boots = defaultdict(list)
    for r in rows:
        boots[(r['board'], r['boot'])].append(r)
    probe = None
    if a.raw:
        if a.probe or a.subtract_empty or a.fault_cycles is not None or a.ols_cycles:
            die('--raw takes no correction options')
        with open(a.out, 'w') as f:
            f.write('\t'.join(fields) + '\n')
            for r in rows:
                r = dict(r)
                if r['program'] in ('ctl-empty', 'ctl-loop'):
                    r['program'] = f'{r["program"]}@{r["board"]}/{r["boot"]}'
                f.write('\t'.join(r[x] for x in fields) + '\n')
        print(f'plain-correct: --raw: {len(rows)} rows, controls renamed per boot, values untouched')
        return
    if a.probe:
        import re
        text = open(a.probe, errors='replace').read()
        if 'PLAINPROBE access=ok' not in text:
            die(f'{a.probe}: no "PLAINPROBE access=ok" line: the user counters were not shown to work in this boot')
        gc = re.search(r'PLAINPROBE gap_cycles min=\d+ p10=\d+ median=(\d+)', text)
        gi = re.search(r'PLAINPROBE gap_instret min=\d+ p10=\d+ median=(\d+)', text)
        if not gc or not gi:
            die(f'{a.probe}: no PLAINPROBE gap_cycles/gap_instret lines')
        probe = (int(gc[1]), int(gi[1]))
        fl = re.search(r'PLAINPROBE fault pages=\d+ minflt_delta=\d+ cycles_min=\d+ cycles_median=(\d+) cycles_p90=\d+ '
                       r'instret_min=\d+ instret_median=(\d+)', text)
        if not fl:
            die(f'{a.probe}: no PLAINPROBE fault line: the per-fault cost is not measured in this boot')
        if a.fault_cycles is None:
            a.fault_cycles = float(fl[1])
        if a.fault_instret is None:
            a.fault_instret = float(fl[2])
        if len([b for b, rs in boots.items() if any(r['arm'] == 'Plain' for r in rs)]) != 1:
            die('--probe applies to one boot; give one boot of rows per invocation')
    elif not a.ols_cycles:
        die('no --probe: the per-interrupt cycle cost needs the probe of the same boot (or --ols-cycles, flagged)')
    report, cal = [], {}
    for b, rs in sorted(boots.items()):
        plain = [r for r in rs if r['arm'] == 'Plain']
        if not plain:
            continue
        for r in plain:
            if key(r) not in kern:
                die(f'boot {b}: Plain {r["program"]} step {r["step"]} it {r["it"]} has no BENCH-KERN row')
        timed = lambda prog, arm: [r for r in rs if r['program'] == prog and r['arm'] == arm and not r['it'].startswith('w')]
        empty, loop = timed('ctl-empty', 'Plain'), timed('ctl-loop', 'Plain')
        if not empty or not loop:
            die(f'boot {b}: no Plain ctl-empty ({len(empty)}) or ctl-loop ({len(loop)}) timed rows: no calibration')
        e_ins = st.median(int(r['instret']) for r in empty)
        e_cyc = st.median(int(r['cycles']) for r in empty)
        irq = [int(kern[key(r)]['irq']) for r in loop]
        excess = [int(r['instret']) - e_ins - (2 * N_LOOP + 2) for r in loop]
        if sum(irq) <= 0:
            die(f'boot {b}: Plain ctl-loop saw no interrupts; the per-tick cost cannot be estimated')
        per = [e / i for e, i in zip(excess, irq) if i]
        k_ins = st.median(per) if a.k_ins_median else sum(excess) / sum(irq)
        if k_ins <= 0:
            die(f'boot {b}: k_ins={k_ins:.1f} is not positive (excess {excess}, irq {irq})')
        cyc = [int(r['cycles']) for r in loop]
        mx, my = st.mean(irq), st.mean(cyc)
        sxx = sum((x - mx) ** 2 for x in irq)
        if sxx == 0:
            die(f'boot {b}: every Plain ctl-loop iteration saw {irq[0]} interrupts; the slope is not identified')
        slope = sum((x - mx) * (y - my) for x, y in zip(irq, cyc)) / sxx
        resid = [y - my - slope * (x - mx) for x, y in zip(irq, cyc)]
        se = (sum(r * r for r in resid) / max(1, len(cyc) - 2) / sxx) ** 0.5
        k_cyc = a.k_cyc if a.k_cyc is not None else probe[0] if probe else slope
        p4 = ''
        if a.k_ins_median:
            off = max(abs(x / k_ins - 1) for x in per)
            ok = (probe is not None and abs(k_ins / probe[1] - 1) <= 0.15) and off <= 0.10
            p4 = (f'; P4 (addendum 1): median k_ins {k_ins:.1f} vs probe {probe[1] if probe else "-"}'
                  f', worst iteration {off * 100:.1f} % off: {"PASS" if ok else "FAIL"}')
        cal[b] = (k_ins, k_cyc)
        report.append(f'boot {b[0]}/{b[1]}: k_ins={k_ins:.1f} instr/irq from ctl-loop (per-iteration {min(per):.1f}..{max(per):.1f})'
                      + (f', probe gap_instret median {probe[1]}' if probe else '')
                      + f'; k_cyc={k_cyc:.1f} cycles/irq from {"--k-cyc (FLAGGED: not this boot\'s probe)" if a.k_cyc is not None else "the probe gap median" if probe else "the OLS slope (FLAGGED: --ols-cycles)"}'
                      f'; cross-check OLS slope {slope:.1f} +- {se:.1f} (SE, {len(loop)} iterations, irq {min(irq)}..{max(irq)})'
                      f'; Plain ctl-empty median cycles={e_cyc:.0f} instret={e_ins:.0f}'
                      + (f'; per fault {a.fault_cycles:.0f} cycles, {a.fault_instret:.0f} instret' if a.fault_cycles is not None else '')
                      + p4)
    if not cal:
        die('no boot has Plain rows')
    empties = {}
    if a.subtract_empty:
        for b, rs in boots.items():
            for arm in {r['arm'] for r in rs}:
                e = [r for r in rs if r['program'] == 'ctl-empty' and r['arm'] == arm and not r['it'].startswith('w')]
                if not e:
                    die(f'boot {b} arm {arm}: no ctl-empty rows to subtract')
                empties[(b, arm)] = (st.median(int(r['cycles']) for r in e), st.median(int(r['instret']) for r in e))
    out = []
    for r in rows:
        r = dict(r)
        b = (r['board'], r['boot'])
        if r['arm'] == 'Plain' and b in cal:
            k = kern[key(r)]
            k_ins, k_cyc = cal[b]
            irq, flt = int(k['irq']), int(k['minflt']) + int(k['majflt'])
            c = int(r['cycles']) - irq * k_cyc - (flt * a.fault_cycles if a.fault_cycles is not None else 0)
            i = int(r['instret']) - irq * k_ins - (flt * a.fault_instret if a.fault_instret is not None else 0)
            r['cycles'], r['instret'] = str(max(1, round(c))), str(max(1, round(i)))
        if a.subtract_empty and r['program'] not in ('ctl-empty',):
            ec, ei = empties[(b, r['arm'])]
            r['cycles'] = str(max(1, round(int(r['cycles']) - ec)))
            r['instret'] = str(max(1, round(int(r['instret']) - ei)))
        if r['program'] in ('ctl-empty', 'ctl-loop'):
            r['program'] = f'{r["program"]}@{b[0]}/{b[1]}'
        out.append(r)
    with open(a.out, 'w') as f:
        f.write('\t'.join(fields) + '\n')
        for r in out:
            f.write('\t'.join(r[x] for x in fields) + '\n')
    faults = sum(int(k['minflt']) + int(k['majflt']) for k in kern.values() if not k['it'].startswith('w'))
    if faults and a.fault_cycles is None:
        report.append(f'WARNING: {faults} page faults in Plain timed iterations are NOT corrected (no --fault-cycles); '
                      'take the per-fault cost from the probe line PLAINPROBE fault')
    text = '\n'.join(report) + '\n'
    print(text, end='')
    if a.report:
        open(a.report, 'w').write(text)


if __name__ == '__main__':
    main()
