#!/usr/bin/env python3
"""Exercise classifications, reset accounting and the fatal overlap control."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'
lib = WORK / 'lib'
lib.mkdir(parents=True, exist_ok=True)
subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', '-g', '-fPIC',
                '-shared', str(HERE/'observer.c'), '-pthread', '-o', str(lib/'libnativesurvey.so')], check=True)
with tempfile.TemporaryDirectory(prefix='observer-', dir=WORK) as name:
    work = Path(name)
    binary = work / 'fixture'
    subprocess.run(['cc', '-Wall', '-Wextra', '-Werror', '-O2', '-fno-builtin',
                    str(HERE/'test-observer.c'), f'-L{lib}', '-lnativesurvey',
                    f'-Wl,-rpath,{lib}', '-pthread', '-o', str(binary)], check=True)
    env = dict(os.environ, NS_OUT=str(work/'counts'))
    subprocess.run([str(binary)], env=env, check=True)
    report = json.loads(next(work.glob('counts.*.json')).read_text())
    rows = {r['family']: r for r in report['allocators']}
    r = rows['sqlite-lookaside']
    expected = dict(alloc=8, free=8, bulk_free=1, reuse=5, inside=3, outside=1,
                    unknown_reuse=1, unknown_alloc=2, cross_instance=1,
                    unknown_free=0, live=0, resize_inplace=1, resize_failed=1)
    for key, value in expected.items():
        assert r[key] == value, (key, r[key], value)
    assert r['gap_inside'][0] == 2 and sum(r['gap_inside']) == 2
    assert rows['wmem-simple']['unknown_alloc'] == 0
    assert rows['wmem-strict']['inside'] == 1
    control = subprocess.run([str(binary), 'overlap'], env=env, capture_output=True)
    assert control.returncode == 86, control
    assert b'overlaps a recorded live object' in control.stderr
    threaded=dict(env,NS_OUT=str(work/'threaded'))
    subprocess.run([str(binary),'threads'],env=threaded,check=True)
    report=json.loads(next(work.glob('threaded.*.json')).read_text())
    r=next(r for r in report['allocators'] if r['family']=='memcached-object-cache')
    assert r['alloc']==r['free']==4000 and r['inside']==3996
    assert r['unknown_alloc']==r['unknown_free']==r['live']==0
    print('PASS backing generations, bulk accounting, resize, unknown backing, cross-instance gaps, fatal overlap, threads')
