#!/usr/bin/env python3
"""Attribute native trampoline instructions from its assembled bytes and measured address."""
import os
import hashlib
from pathlib import Path
source = Path(os.environ['ORIGINAL_PROF_REPORT']).read_text()
if os.environ.get('DOM_SPEC_NATIVE') == '1':
    needle = '        images = [prog, *helpers, profrun]'
    assert source.count(needle) == 1
    source = source.replace(needle, '''        images = [prog, profrun]
        raw = Path(os.environ['DOM_TRAMPOLINE']).read_bytes()
        trampoline = int(Path(OUT, 'invocations', inv_id + '.addresses').read_text().split()[1])
        im = object.__new__(Image)
        im.name, im.path, im.slide = 'trampoline', None, trampoline
        im.segs, im.e_type = [(0, len(raw), raw)], 2
        im.syms, im.addrs = [(0, 'trampoline')], [0]
        im.sha = hashlib.sha256(raw).hexdigest()
        images.append(im)''')
exec(compile(source, os.environ['ORIGINAL_PROF_REPORT'], 'exec'))
