#!/usr/bin/env python3
"""Relink an existing ASan host Perl build with a test-only malloc wrapper.

The source tree and its commands.json must come from the same completed build.
No production object, licensed source, or native runtime is modified. The libc
allocation wrapper is linked into this standalone test executable only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-tree', required=True, type=Path)
    parser.add_argument('--commands', required=True, type=Path)
    parser.add_argument('--lib', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expect-vulnerable', action='store_true',
                        help='Require ASan UAF in both fault cases as a negative control')
    args = parser.parse_args()
    tree = args.source_tree.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parent
    commands = json.loads(args.commands.read_text())
    compile_command = commands['compile']['perlmain.o'][:]
    if '-fsanitize=address' not in compile_command or '-DCAPSTONE_PERL_PROVENANCE_TEST=1' not in compile_command:
        raise SystemExit('The host build must enable ASan and CAPSTONE_PERL_PROVENANCE_TEST')
    obj = out / 'allocation-failure.o'
    binary = out / 'allocation-failure'
    compile_command[compile_command.index('-o') + 1] = str(obj)
    compile_command[-1] = str(fixture / 'allocation-failure.c')
    compile_command.append('-g')
    link_command = [str(obj) if item == 'perlmain.o' else item
                    for item in commands['link']]
    if str(obj) not in link_command:
        raise SystemExit('Cannot identify perlmain.o in the recorded link command')
    link_command[link_command.index('-o') + 1] = str(binary)
    link_command.append('-Wl,--wrap=malloc')
    for name, command in [('compile', compile_command), ('link', link_command)]:
        result = subprocess.run(command, cwd=tree, capture_output=True, timeout=120)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise SystemExit(f'{name} failed: see {out}/{name}.stderr')
    env = os.environ.copy()
    env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
    results = {}
    for mode in ('catch', 'oom'):
        command = [str(binary), '-I' + str(args.lib.resolve()),
                   str(fixture / 'allocation-failure.pl'), mode]
        result = subprocess.run(command, cwd=out, capture_output=True,
                                timeout=60, env=env)
        (out / (mode + '.stdout')).write_bytes(result.stdout)
        (out / (mode + '.stderr')).write_bytes(result.stderr)
        triggered = b'ALLOCATION_FAULT ' + mode.encode() + b'\n' in result.stdout
        if args.expect_vulnerable:
            passed = triggered and result.returncode != 0 and b'ERROR: AddressSanitizer: heap-use-after-free' in result.stderr
            if mode == 'oom':
                passed = passed and b'cap_prov_free' in result.stderr and b'Out of memory!' in result.stderr
        elif mode == 'catch':
            passed = (triggered and result.returncode == 0 and not result.stderr
                      and result.stdout.count(b'ALLOCATION_FAULT catch\n') == 32
                      and b'ALLOCATION_CHECKS checks=128 failures=0\n' in result.stdout)
        else:
            passed = (triggered and result.returncode == 1
                      and result.stderr == b'Out of memory!\n')
        results[mode] = {'status': 'PASS' if passed else 'FAIL',
                         'returncode': result.returncode,
                         'stdout_sha256': sha(out / (mode + '.stdout')),
                         'stderr_sha256': sha(out / (mode + '.stderr')),
                         'stderr_bytes': len(result.stderr)}
    record = {
        'status': 'PASS' if all(r['status'] == 'PASS' for r in results.values()) else 'FAIL',
        'scope': 'host-only allocator-failure regression; no native tag or ISA qualification',
        'expect_vulnerable': args.expect_vulnerable,
        'fault_modes': {
            'catch': 'malloc wrapper throws test-only recoverable exception; 32 iterations, 128 checks',
            'oom': 'malloc wrapper returns NULL; real Perl safesysmalloc/croak_no_mem/my_exit cleanup'},
        'fixture_sha256': {name: sha(fixture / name) for name in
                           ('allocation-failure.c', 'allocation-failure.pl', 'run-allocation-failure.py')},
        'commands_sha256': sha(args.commands),
        'compiled_source_sha256': {name: sha(tree / name) for name in
                                   ('sv.c', 'sv.h', 'perl.h', 'mg.c', 'util.c', 'scope.c', 'spec_config.h')},
        'input_object_sha256': {name: sha(tree / name) for name in commands['link']
                                if name.endswith('.o') and name != 'perlmain.o'},
        'test_binary_sha256': sha(binary),
        'build_commands': {'compile': compile_command, 'link': link_command},
        'results': results,
    }
    (out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'results': results, 'summary': str(out / 'summary.json')}))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
