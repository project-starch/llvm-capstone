#!/usr/bin/env python3
"""Check launch-vector limits on the host, including an overlapping-vector control."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--cc', default='cc')
a = p.parse_args()
runtime = Path(__file__).resolve().parents[2]
a.work.mkdir(parents=True, exist_ok=False)
common = [a.cc, '-std=c11', '-D_GNU_SOURCE', '-Wall', '-Wextra', '-Werror',
          '-I' + str(runtime / 'include'),
          str(runtime / 'tests/application/launch-layout-test.c'),
          str(runtime / 'common/launch.c')]
modes = ('max-args', 'max-env', 'mixed', 'bad-argc', 'bad-envc', 'bad-offset', 'truncated')
for enabled in (0, 1):
    binary = a.work.resolve() / ('launch-' + str(enabled))
    subprocess.run(common + [f'-DCAPSTONE_RUNTIME_COMPACT_LAUNCH={enabled}',
                            '-o', str(binary)], check=True)
    for mode in modes:
        subprocess.run([str(binary), mode], check=True)

source = (runtime / 'domain/application.c').read_text()
assert source.count('arguments + h.argc + 1') == 1
mutant = a.work.resolve() / 'application-overlap.c'
mutant.write_text(source.replace('arguments + h.argc + 1', 'arguments + h.argc')
                       .replace('arguments + launch.argc + 1', 'arguments + launch.argc'))
binary = a.work.resolve() / 'launch-overlap'
subprocess.run(common + ['-DCAPSTONE_RUNTIME_COMPACT_LAUNCH=1',
                        '-DCAPSTONE_APPLICATION_SOURCE="' + str(mutant) + '"',
                        '-o', str(binary)], check=True)
with (a.work / 'overlap.log').open('w') as log:
    result = subprocess.run([str(binary), 'max-env'], stdout=log, stderr=subprocess.STDOUT)
assert result.returncode != 0, 'overlapping vectors escaped the oracle'
print('launch-layout overlapping-vector mutant rejected')
