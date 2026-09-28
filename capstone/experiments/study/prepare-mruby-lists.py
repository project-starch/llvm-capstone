#!/usr/bin/env python3
"""Prepare the pinned upstream lists workload without vendoring its source.

The loop body and default work counts are unchanged. Added checks expose the
result of every iteration; phase markers bracket the loop without forcing GC.
Use the same generated file on all targets and an independent native reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = 'aae949f940157b0d4227095a9f81c5d441c92bc3768d2546d943b0b30c5ab673'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = args.source.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected mruby 4.0.0-rc2 benchmark/bm_so_lists.rb')
    source = original.decode()
    source = source.replace('i = 0\n', 'STDERR.syswrite("MEMPHASE before\\n")\ni = 0\n')
    source = source.replace('  result = test_lists()\n',
                            '  result = test_lists()\n  raise "list oracle" unless result == SIZE\n')
    source += '\nSTDERR.syswrite("MEMPHASE after\\n")\nputs "LISTS-OK #{i} #{result}"\n'
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'lists.rb').write_text(source)
    (args.out/'workload.json').write_text(json.dumps(dict(
        schema_version=1, suite='mruby-upstream', case='bm_so_lists.rb',
        source_sha256=SOURCE_SHA256,
        generated_sha256=hashlib.sha256(source.encode()).hexdigest(),
        parameters=dict(iterations=300, list_size=10000),
        adaptation='Unchanged work counts and body; per-iteration result check, two phase writes and final text output; no forced GC.',
        expected_stdout='LISTS-OK 300 10000\n',
        expected_phases=['startup', 'before', 'after', 'exit']), indent=2)+'\n')


if __name__ == '__main__':
    main()
