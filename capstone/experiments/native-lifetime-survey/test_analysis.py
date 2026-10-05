#!/usr/bin/env python3
"""Negative controls against the checked-in campaign's acceptance rules."""
from copy import deepcopy
from pathlib import Path
import analyze

root=Path(__file__).resolve().parent/'results/native-arm64-20261005'
valid=analyze.read_export(root)
assert len(valid)==96

def rejected(records):
    try:
        analyze.validate(records)
    except ValueError:
        return
    raise AssertionError('invalid campaign accepted')

rejected(valid[:-1])
rejected(valid+[deepcopy(valid[0])])
wrong=deepcopy(valid)
wrong[0]['oracle']={'wrong':True}
rejected(wrong)
wrong=deepcopy(valid)
row=next(r for r in wrong if r['variant']=='observed')['measurement']['allocators'][0]
row['inside']+=1
rejected(wrong)
wrong=deepcopy(valid)
row=next(r for r in wrong if r['variant']=='observed')['measurement']['allocators'][0]
row['unknown_alloc']=1
rejected(wrong)
assert analyze.ratio(0,0) is None
print('PASS complete matrix, duplicate/missing cells, oracle mismatch, counter mismatch, unknown coverage, undefined ratio')
