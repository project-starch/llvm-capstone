#!/usr/bin/env python3
"""Check native exec of a delegated contract image in a provisioned guest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--image', required=True, help='Guest delegate-contract.dom path')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'host'))
    cli = [sys.executable, '-m', 'capstone_vm', '--state', str(args.state), 'exec']
    results = {}

    def check(name, command, status, stdout):
        result = subprocess.run(cli + command, env=env, capture_output=True, text=True, timeout=30)
        if result.returncode != status or result.stdout != stdout:
            raise RuntimeError(f'{name}: status={result.returncode}, stdout={result.stdout!r}, '
                               f'stderr={result.stderr!r}')
        results[name] = {'exit': result.returncode, 'stdout': result.stdout}
        print(f'{name}: PASS')

    check('native-elf', ['/bin/true'], 0, '')
    # BusyBox ash's exec -a reaches the kernel with an independent argv[0].
    # The existing child contract asserts that exact string and exits 7.
    check('original-argv0', ['sh', '-c', 'exec -a "custom argv zero" "$1" child',
                           'binfmt-test', args.image], 7, '')
    for mode, stdout in [('io', 'delegate-contract: io and spawn ok\n'),
                         ('exec', 'delegate-contract: exec ok\n'),
                         ('exec-closed', ''),
                         ('exec-error', 'delegate-contract: failed exec preserved task\n'),
                         ('lock', 'delegate-contract: record locks ok\n'),
                         ('usable-size', 'delegate-contract: usable size ok\n'),
                         ('buffer-bounds', 'delegate-contract: buffer bounds ok\n')]:
        check(mode, ['sh', '-c', 'exec "$@"', 'binfmt-test', args.image, mode, args.image],
              0, stdout)
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
