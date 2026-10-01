"""Link rawmutex-test.c into a domain with a CPython build's objects.

The test takes Programs/python.o's place in the interpreter's link, so it runs
against exactly the objects of that build (patch 0016's directed test).

usage: link-rawmutex.py APPLICATION_DIR CPY_ROOT OUT.dom
  APPLICATION_DIR  a common/application/build.py output (its commands.json)
  CPY_ROOT         the prepare root the interpreter was linked from"""
import json, subprocess, sys
from pathlib import Path

apps, root, out = map(Path, sys.argv[1:4])
link = json.loads((apps / 'commands.json').read_text())[2]
cc, src = link[0], root / 'src/Python-3.13.7'
obj = out.with_suffix('.o')
subprocess.run([cc, '-O1', '-DPy_BUILD_CORE', '-I', src / 'Include', '-I', src / 'Include/internal',
                '-I', root / 'build', '-c', Path(__file__).with_name('rawmutex-test.c'), '-o', obj],
               check=True)
objects = [w for w in link[1:] if w.endswith('.o') and not w.endswith('Programs/python.o')]
if len(objects) != len(link) - 4:
    sys.exit(f'expected the interpreter link with Programs/python.o, got {len(link)} words')
subprocess.run([cc, obj, *objects, '-o', out], check=True)
print(out)
