#!/usr/bin/env python3
"""Record the allocator's layout region for the two arms' SDKs (the layout control).

  heap-layout.py --sdk <sdk-T1/sdk> --sdk <sdk-T2/sdk> --out heap-layout.json

Every image links heap.c's object LAST in .text and in .bss, inside a region of fixed size that is
the same in both arms (build.py's layout-K<k>.ld): so every other input section, the benchmark's,
libc's and all of .rodata, sits at the same address in both arms, and only heap.c's own code moves.
The text region also holds the K = 0..3 allocator-alignment shifts (K x 64 bytes).

This script reads heap.c's object out of each SDK's libapplication-runtime.a, checks that its only
allocated sections are .text.* and .bss.* (any other would sit outside the region and move what
follows it), and writes the region sizes and the objects' hashes. build.py refuses an SDK whose
heap.c object is not one of the recorded ones, so a stale record cannot be used silently.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

K_STEP = 64
K_MAX = 3


def archive_member(path, member):
    data = Path(path).read_bytes()
    assert data[:8] == b'!<arch>\n', path
    pos, names, found = 8, b'', []
    while pos + 60 <= len(data):
        hdr = data[pos:pos + 60]
        size = int(hdr[48:58].decode().strip())
        body = data[pos + 60:pos + 60 + size]
        name = hdr[:16].decode().strip()
        pos += 60 + size + (size & 1)
        if name in ('/', '/SYM64/'):
            continue
        if name == '//':
            names = body
            continue
        if name.startswith('/') and name[1:].isdigit():
            off = int(name[1:])
            name = names[off:names.index(b'\n', off)].decode()
        if name.rstrip('/') == member:
            found.append(body)
    if len(found) != 1:
        sys.exit(f'{path}: {len(found)} members named {member}, expected exactly one')
    return found[0]


def sections(obj_bytes, readelf):
    with tempfile.NamedTemporaryFile(suffix='.o') as f:
        f.write(obj_bytes)
        f.flush()
        out = subprocess.run([readelf, '-S', '--wide', f.name], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith('['):
            continue
        rest = line.split(']', 1)[1].split()
        if len(rest) < 7 or not rest[0].startswith('.'):
            continue
        name, typ, size = rest[0], rest[1], int(rest[4], 16)
        flags = rest[6] if len(rest) >= 10 else ''
        align = int(rest[-1])
        rows.append(dict(name=name, type=typ, size=size, flags=flags, align=align))
    return rows


def packed(secs):
    off = 0
    for s in secs:
        off = (off + s['align'] - 1) // s['align'] * s['align'] + s['size']
    return off


def ceil(x, a):
    return (x + a - 1) // a * a


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sdk', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    readelf = os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), 'llvm-readelf')
    rec = dict(k_step=K_STEP, k_max=K_MAX, objects={})
    texts, bsses = [], []
    for sdk in a.sdk:
        cfg = json.loads((sdk / 'sdk.json').read_text())
        obj = archive_member(cfg['runtime'], 'heap.c.obj')
        secs = sections(obj, readelf)
        alloc = [s for s in secs if 'A' in s['flags']]
        istext = lambda n: n == '.text' or n.startswith('.text.')
        isbss = lambda n: n == '.bss' or n.startswith('.bss.')
        other = [s['name'] for s in alloc if not (istext(s['name']) or isbss(s['name']))]
        if other:
            sys.exit(f'{sdk}: heap.c.obj has allocated sections outside .text.*/.bss.*: {other}')
        text = packed([s for s in alloc if istext(s['name'])])
        bss = packed([s for s in alloc if isbss(s['name'])])
        rec['objects'][hashlib.sha256(obj).hexdigest()] = dict(
            sdk=str(sdk.resolve()), heap_lifetimes=cfg.get('heap_lifetimes'), heap_scrub_free=cfg.get('heap_scrub_free'),
            text_bytes=text, bss_bytes=bss)
        texts.append(text)
        bsses.append(bss)
    rec['text_region'] = ceil(max(texts), K_STEP) + K_MAX * K_STEP
    rec['bss_region'] = ceil(max(bsses), K_STEP)
    a.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
