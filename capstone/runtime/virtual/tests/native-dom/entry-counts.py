#!/usr/bin/env python3
"""Count actual rd=x0 CSRUNV entries in matched, complete profile windows.

The reported step function's self cost excludes callees and hardware-hidden
state transfer. Module cost also includes setup/teardown; neither is an
isolated hardware context-switch latency.
"""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
for arm in ('plain', 'launcher', 'native'):
    directory = root / ('measure-' + arm)
    sections = {}
    for line in (directory / 'module-sections.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].startswith('.'):
            sections[fields[0]] = int(fields[1], 16)
    low = sections['.text']
    high = min(address for address in sections.values() if address > low)
    windows = []
    for i in range(6):
        record = json.loads((directory / 'invocations' / f'{i:04d}.json').read_text())
        assert record['rc'] == 0 and record['detail']['window'] == i
        output = (directory / 'invocations' / f'{i:04d}.out').read_text()
        assert 'bad=0' in output and f'n={10 if i % 2 == 0 else 100} ' in output
        profile = json.loads((directory / 'profile' / f'window-{i}.json').read_text())
        entries = total = 0
        for line in (directory / 'capprof' / f'window-{i}.tsv').read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            pc, length, bits, attempts, _ = line.split()
            attempts = int(attempts)
            total += attempts
            # custom-2, funct3=1, funct7=0x24, rd=x0: excludes discard/collect.
            if (low <= int(pc, 16) < high and int(length) == 4 and
                    int(bits, 16) & 0xfe007fff == 0x4800105b):
                entries += attempts
        assert total == profile['total_attempts'] == sum(profile['classes'].values())
        assert total == sum(row[2] for row in profile['all_functions'])
        step = sum(n for layer, name, n in profile['all_functions']
                   if layer == 'module' and name.startswith('cv_step_locked'))
        windows.append((entries, profile['classes'].get('module', 0), step))
    per_process = [tuple((b - a) / 90 for a, b in zip(windows[i], windows[i+1]))
                   for i in (0, 2, 4)]
    entries, module, step = (statistics.mean(row[j] for row in per_process) for j in range(3))
    assert (entries == 0) if arm == 'plain' else (entries > 0 and step > 0)
    print(json.dumps(dict(arm=arm, entries_per_process=entries,
        module_attempts_per_entry=module / entries if entries else None,
        step_self_attempts_per_entry=step / entries if entries else None,
        windows=windows)))
