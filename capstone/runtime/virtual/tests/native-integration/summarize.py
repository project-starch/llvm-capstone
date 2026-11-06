#!/usr/bin/env python3
"""System-wide attempted instructions; optional exact kernel/module text ranges."""
import argparse
import pathlib
import re
import struct

p = argparse.ArgumentParser()
p.add_argument('output', type=pathlib.Path)
p.add_argument('--kernel-map', type=pathlib.Path)
a = p.parse_args()
root = a.output
arms = [line.split() for line in (root/'arms.tsv').read_text().splitlines()]
windows = list((root/'capprof').glob('window-*.tsv'))
assert arms and len(windows) == len(arms), 'incomplete or unexpected windows'
assert not list((root/'capprof').glob('*.invalid')), 'invalid profiling window'
sections = []
if a.kernel_map:
    syms = {row.split()[2]: int(row.split()[0], 16) for row in a.kernel_map.read_text().splitlines() if len(row.split()) >= 3}
    kernel = (syms['_stext'], syms['_etext'])
    data = (root/'stage/capstone_vm.ko').read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01'
    shoff = struct.unpack_from('<Q', data, 0x28)[0]
    entsize, count, strindex = struct.unpack_from('<HHH', data, 0x3a)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, shoff+i*entsize) for i in range(count)]
    strings = headers[strindex][4]
    for h in headers:
        if h[2] & 4:  # SHF_EXECINSTR; no guessed module address range
            start = strings+h[0]
            sections.append((data[start:data.index(b'\0', start)].decode(), h[5]))
print('arm\tdirection\tbytes\trep\tattempts\tcalls\tattempts_per_call'+('\tkernel_per_call\tmodule_per_call\tother_per_call' if a.kernel_map else ''))
for row in arms:
    index, arm, size, rep = row[:4]
    direction = row[4] if len(row) == 5 else 'read'
    log = (root/f'{index}.log').read_text()
    calls = 1 if size == '65536' else 100
    assert f'BUDGET bytes={size} calls={calls} failures=0' in log, log
    mapped = []
    if a.kernel_map:
        bases = {n: int(v, 16) for n, v in re.findall(r'^MODULE_SECTION:([^:]+):(0x[0-9a-f]+)$', log, re.M)}
        assert '.text' in bases
        mapped = [(bases[n], bases[n]+length) for n, length in sections if n in bases]
    attempts = k = m = other = 0
    for line in (root/'capprof'/f'window-{index}.tsv').read_text().splitlines():
        if line.startswith('#'): continue
        fields = line.split()
        pc, n = int(fields[0], 16), int(fields[3])
        attempts += n
        if a.kernel_map:
            if kernel[0] <= pc < kernel[1]: k += n
            elif any(lo <= pc < hi for lo, hi in mapped): m += n
            else: other += n  # user, launcher, firmware, or unresolved PCs
    assert not a.kernel_map or attempts == k+m+other
    print(f'{arm}\t{direction}\t{size}\t{rep}\t{attempts}\t{calls}\t{attempts/calls:.2f}'+(f'\t{k/calls:.2f}\t{m/calls:.2f}\t{other/calls:.2f}' if a.kernel_map else ''))
