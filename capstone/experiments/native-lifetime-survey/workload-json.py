#!/usr/bin/env python3
"""Run upstream pyperformance JSON functions with a fixed work count.

The functions and datasets stay unchanged. This harness replaces pyperf's
time-based calibration and subprocess runner. It reports no benchmark score.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root, kind, iterations = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
spec = importlib.util.spec_from_file_location('upstream_json', root/f'json_{kind}.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if kind == 'loads':
    objects = (module.DICT, module.TUPLE, module.DICT_GROUP)
    encoded = tuple(json.dumps(x) for x in objects)
    for _ in range(iterations):
        module.bench_json_loads(encoded)
    # JSON turns tuples into lists. Re-encoding must preserve each input.
    for data in encoded:
        assert json.dumps(json.loads(data)) == data
    calls = iterations * 20 * len(encoded)
else:
    objects = tuple(getattr(module, name)[0] for name in module.CASES)
    data = tuple((getattr(module, name)[0], range(getattr(module, name)[1])) for name in module.CASES)
    for _ in range(iterations):
        module.bench_json_dumps(data)
    encoded = tuple(json.dumps(x) for x in objects)
    for obj, text in zip(objects, encoded):
        assert json.loads(text) == obj
    calls = iterations * sum(len(indices) for _, indices in data)
digest = hashlib.sha256('\n'.join(encoded).encode()).hexdigest()
print(json.dumps(dict(workload='pyperformance-json-'+kind, iterations=iterations,
                      calls=calls, payload_sha256=digest), sort_keys=True))
