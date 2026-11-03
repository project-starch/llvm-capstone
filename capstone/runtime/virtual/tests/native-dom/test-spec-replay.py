#!/usr/bin/env python3
"""Incomplete census manifests must fail before starting a guest."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch

source = Path(__file__).with_name('spec-run.py')
with tempfile.TemporaryDirectory(prefix='native-replay-', dir=os.environ['CAPSTONE_TMP_ROOT']) as name:
    root = Path(name)
    census = root / 'prof-dev/runs/census4/cap-dev4/502.gcc_r/invocations'
    census.mkdir(parents=True)
    cases = [([], 'successful invocations and SPEC oracles'),
             ([{'route': 'guest', 'rc': 0}], 'successful invocations and SPEC oracles'),
             ([{'route': 'guest', 'status': 1}, {'route': 'host-spec-tool', 'status': 0}],
              'successful invocations and SPEC oracles'),
             ([{'route': 'guest', 'rc': 0, 'status': 1}, {'route': 'host-spec-tool', 'status': 0}],
              'successful invocations and SPEC oracles'),
             ([{'route': 'guest', 'rc': 0}, {'route': 'guest', 'rc': 0},
               {'route': 'host-spec-tool', 'rc': 0}], 'invocation count changed'),
             ([{'route': 'guest', 'status': 0}, {'route': 'guest', 'status': 0},
               {'route': 'host-spec-tool', 'status': 0}], 'invocation count changed')]
    for i, (records, reason) in enumerate(cases):
        for p in census.glob('*.json'):
            p.unlink()
        for j, record in enumerate(records):
            (census / f'{j:04d}.json').write_text(json.dumps(record))
        with patch.object(Path, 'home', return_value=root), \
             patch.dict(os.environ, LANE_ROOT=str(root), SPEC_RUN_ROOT=str(root / f'case-{i}')), \
             patch('sys.argv', [str(source), '502.gcc_r', 'native']), \
             patch('subprocess.run', side_effect=AssertionError('VM launched with invalid census')), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(source), run_name='__main__')
            except SystemExit as error:
                assert reason in str(error), error
            else:
                raise AssertionError('invalid census accepted')
        print('PASS replay rejects', reason)
