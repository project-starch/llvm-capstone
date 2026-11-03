#!/usr/bin/env python3
"""Extend the existing profiler with recorded mixed parent/child image placement.

ORIGINAL_PROF_REPORT names kit4/tools/prof-report.py. DOM_ARM selects plain,
launcher or native. All probes run outside the capprof windows. Byte collisions
remain in the upstream profiler's ambiguous bucket; they are never guessed.
"""
import os
import pathlib

path = pathlib.Path(os.environ['ORIGINAL_PROF_REPORT'])
source = path.read_text()
needle = '        images = [prog, profrun]'
assert source.count(needle) == 1
replacement = '''        images = [prog, profrun]
        arm = os.environ['DOM_ARM']
        if arm != 'plain':
            child = os.environ['DOM_CHILD']
            if arm == 'native':
                base, trampoline = [int(x) for x in open(os.path.join(OUT, 'invocations', inv_id + '.addresses')).read().split()]
            else:
                pilot = open(os.path.join(OUT, 'invocations', inv_id + '.pilot')).read()
                m = re.search(r'code=0x([0-9a-f]+)-', pilot)
                if not m: fail('missing launcher image placement probe')
                base = int(m.group(1), 16)
                images.append(Image('launcher', LAUNCHER))
            segs, _ = elf_segments(child)
            images.append(Image('domain', child, slide=base - segs[0][0]))
            if arm == 'native':
                import hashlib
                words = [0x00040513,0x00048593,0x00000613,0x01d00893,0x00000073,
                         0x10100293,0x00550c63,0xfea052e3,0xfff50513,0x05e00893,
                         0x00000073,0x0000006f,0x00058513,0x00060593,0x00068613,
                         0x0e200893,0x00000073,0x00050613,0x00040513,0x00090593,
                         0x01d00893,0x00000073,0xfa9ff06f]
                raw = struct.pack('<' + 'I' * len(words), *words)
                im = object.__new__(Image)
                im.name, im.path, im.slide = 'trampoline', None, trampoline
                im.segs, im.e_type = [(0, len(raw), raw)], 2
                im.syms, im.addrs = [(0, 'trampoline')], [0]
                im.sha = hashlib.sha256(raw).hexdigest()
                images.append(im)'''
source = source.replace(needle, replacement)
exec(compile(source, str(path), 'exec'))
