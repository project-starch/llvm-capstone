#!/usr/bin/env python3
"""kernel-faults.py RUN_DIR: the steps of a run in which the kernel refused an access through a
program's reference, so the system call returned EFAULT or a short count and the program went on.

trap-vexec.sh prints one `capstone-exec: kernel fault cause= address= nr= image=` line per such access,
from the kernel's `sublet: kernel fault ...` line. A step with such a line is "blocked in the kernel":
it is counted here apart from caught and missed, and the judges' verdicts for it are left as they are,
so the scoring can be decided on the list. Writes RUN_DIR/kernel-faults.json and prints a summary
per part of the run (the directory that holds each serial log).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BEGIN, END = '@@STEP-BEGIN ', '@@STEP-END '
LINE = 'capstone-exec: kernel fault '


def main():
    run = Path(sys.argv[1])
    found = defaultdict(list)
    for serial in sorted(run.rglob('serial.log')):
        text = serial.read_text(errors='replace').replace('\r\n', '\n')
        part = serial.relative_to(run).parts[0:-1]
        part = '/'.join(p for p in part if not re.fullmatch(r'batch\d+|work|guest(\.boot-\d+)?|boot-\d+', p))
        for m in re.finditer(rf'^{re.escape(BEGIN)}(\S+)\n(.*?)(?:{re.escape(END)}\1 (\d+)$|\Z)', text, re.S | re.M):
            lines = [line.strip() for line in m.group(2).splitlines() if LINE in line]
            if lines:
                found[part].append({'serial': str(serial.relative_to(run)), 'step': m.group(1),
                                    'status': m.group(3), 'kernel_faults': lines})
    (run / 'kernel-faults.json').write_text(json.dumps(found, indent=1) + '\n')
    total = sum(len(v) for v in found.values())
    print(f'blocked in the kernel: {total} steps')
    for part, steps in sorted(found.items()):
        print(f'  {part}: {len(steps)} ({", ".join(s["step"] for s in steps[:6])}{" ..." if len(steps) > 6 else ""})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
