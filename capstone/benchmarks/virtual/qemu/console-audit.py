#!/usr/bin/env python3
"""Console audit of a run-qemu.py boot: every line each process put on the console besides the
harness's own.

  console-audit.py <run dir> [--tsv out.tsv]

Per (set, image) block of guest/serial.log, lines are classed as
  harness   BENCH-INFO, BENCH, BENCH-DONE, BENCH-ERROR, ALLOC (ALLOC only in counting builds)
  launcher  capstone-vexec's own stderr (CAPSTONE_VM_*, capstone-exec:, [CAPSTONE] fault lines)
  kernel    printk lines that reached the console ("[   12.345] ...")
  program   anything else: the benchmark's own stdout/stderr
The runner's markers (RUN-BEGIN/END, WALL, the DMESG-BEGIN..END dump) are not counted. A line is
attributed to an iteration window when it falls between that iteration's BENCH-INFO/previous BENCH
line and its BENCH line, i.e. it was printed before the iteration's report; the harness prints
nothing inside a bracket, so any such line was printed by the program or the launcher during an
iteration (inside or around its timed region).
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

HARNESS = re.compile(r'^(BENCH-INFO|BENCH-DONE|BENCH-ERROR|BENCH|ALLOC) ')
LAUNCHER = re.compile(r'^(CAPSTONE_VM_|capstone-exec|capstone-vexec|\[CAPSTONE\])')
KERNEL = re.compile(r'^\[\s*\d+\.\d+\] ')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--tsv', type=Path)
    a = p.parse_args()
    log = (a.run / 'guest' / 'serial.log').read_text(errors='replace').replace('\r', '')
    rows = []
    cur = None
    in_dmesg = False
    for line in log.splitlines():
        m = re.match(r'RUN-BEGIN (\S+) (\S+)$', line)
        if m:
            cur = dict(set=m.group(1), image=m.group(2), counts=Counter(), during=Counter(), examples={},
                       started=False, ended=False)
            continue
        if cur is None:
            continue
        if line.startswith('RUN-END '):
            rows.append(cur)
            cur = None
            continue
        if line == 'DMESG-BEGIN':
            in_dmesg = True
            continue
        if line == 'DMESG-END':
            in_dmesg = False
            continue
        if in_dmesg or line.startswith('WALL ') or not line.strip():
            continue
        if HARNESS.match(line):
            if line.startswith('BENCH-INFO'):
                cur['started'] = True
            if line.startswith('BENCH-DONE'):
                cur['ended'] = True
            continue
        cls = 'launcher' if LAUNCHER.match(line) else 'kernel' if KERNEL.match(line) else 'program'
        cur['counts'][cls] += 1
        if cur['started'] and not cur['ended']:
            cur['during'][cls] += 1
        key = re.sub(r'0x[0-9a-f]+|\d+', 'N', line)[:120]
        cur['examples'].setdefault((cls, key), line[:200])
    out = ['\t'.join(['set', 'image', 'program_lines', 'program_lines_during_iterations', 'launcher_lines',
                      'launcher_lines_during_iterations', 'kernel_lines', 'distinct_non_harness_lines'])]
    for r in rows:
        ex = ' || '.join(f'{c}: {v}' for (c, _), v in list(r['examples'].items())[:6])
        out.append('\t'.join(map(str, [r['set'], r['image'], r['counts']['program'], r['during']['program'],
                                       r['counts']['launcher'], r['during']['launcher'], r['counts']['kernel'], ex])))
    if a.tsv:
        a.tsv.write_text('\n'.join(out) + '\n')
    prog = [r for r in rows if r['counts']['program']]
    print(f'{len(rows)} process runs; {len(prog)} print program lines besides the harness')
    for r in prog:
        print(f"  {r['set']} {r['image']}: {r['counts']['program']} program lines "
              f"({r['during']['program']} during iterations)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
