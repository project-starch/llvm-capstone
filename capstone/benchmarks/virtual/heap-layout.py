#!/usr/bin/env python3
"""Record fixed allocator text/bss regions for benchmark layout controls.

Minimal mallocng includes malloc-mapping.c and libc's malloc archive members.
Other allocated constants must be relocation-free and identical in both arms.
Historical heap.c SDKs remain readable, but cannot be mixed with the new port.
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


def archive_members(path):
    data = Path(path).read_bytes()
    assert data[:8] == b'!<arch>\n', path
    pos, names, found = 8, b'', {}
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
        found[name.rstrip('/')] = body
    return found

def archive_member(path, member):
    members = archive_members(path)
    if member not in members: sys.exit(f'{path}: no member named {member}')
    return members[member]

def allocator_members(cfg):
    runtime = archive_members(cfg['runtime'])
    if 'heap.c.obj' in runtime:
        return {'heap.c.obj': runtime['heap.c.obj']}
    members = {'malloc-mapping.c.obj': runtime['malloc-mapping.c.obj']}
    members.update({n:b for n,b in archive_members(cfg['libc']).items() if n.startswith('src_malloc_') and not n.startswith(('src_malloc_lite_malloc.', 'src_malloc_oldmalloc_'))})
    return members

def allocator_digest(cfg):
    members = allocator_members(cfg)
    if list(members) == ['heap.c.obj']:
        return hashlib.sha256(members['heap.c.obj']).hexdigest()
    return hashlib.sha256(b''.join(n.encode()+b'\0'+b for n,b in sorted(members.items()))).hexdigest()


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
        rows.append(dict(name=name, type=typ, size=size, flags=flags, align=align, offset=int(rest[3],16)))
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
    common_other = None
    for sdk in a.sdk:
        cfg = json.loads((sdk / 'sdk.json').read_text())
        members = allocator_members(cfg)
        patterns = ['*libapplication-runtime.a:heap.c.obj'] if list(members)==['heap.c.obj'] else [
            '*libapplication-runtime.a:malloc-mapping.c.obj'] + [
                '*lib*.a:'+n for n in sorted(members) if n.startswith('src_malloc_')]
        if 'members' in rec and rec['members'] != patterns:
            sys.exit('Do not mix the retired heap and minimal mallocng in one layout control')
        rec['members'] = patterns
        alloc = []
        other = {}
        for name,obj in sorted(members.items()):
            for sec in sections(obj,readelf):
                if 'A' not in sec['flags']: continue
                if sec['name'].startswith(('.text','.bss')): alloc.append(sec)
                else:
                    # Constants outside the shifted regions must be byte-identical
                    # in both arms. No address-bearing data is admitted here.
                    if sec['name'] not in ('.rodata',) and not sec['name'].startswith('.rodata.'):
                        if sec['size']: sys.exit(f'{sdk}: {name}: unexpected allocated {sec["name"]}')
                    if any(x['name'] in ('.rela'+sec['name'], '.rel'+sec['name']) for x in sections(obj,readelf)):
                        sys.exit(f'{sdk}: relocated allocator constants need a separate layout region')
                    other[name+':'+sec['name']] = hashlib.sha256(obj[sec['offset']:sec['offset']+sec['size']]).hexdigest()
        if common_other is not None and other != common_other:
            sys.exit('Allocator constants outside the layout regions differ between SDKs')
        common_other = other
        # Include an alignment allowance for each section; archive extraction
        # order need not be the same as input object order at the final link.
        text = sum(x['size']+x['align']-1 for x in alloc if x['name'].startswith('.text'))
        bss = sum(x['size']+x['align']-1 for x in alloc if x['name'].startswith('.bss'))
        rec['objects'][allocator_digest(cfg)] = dict(
            sdk=str(sdk.resolve()), heap_lifetimes=cfg.get('heap_lifetimes'),
            heap_scrub_free=cfg.get('heap_scrub_free'), heap_object_bounds=cfg.get('heap_object_bounds',True),
            text_bytes=text, bss_bytes=bss)
        texts.append(text)
        bsses.append(bss)
    rec['shared_allocator_constants'] = common_other
    rec['text_region'] = ceil(max(texts), K_STEP) + K_MAX * K_STEP
    rec['bss_region'] = ceil(max(bsses), K_STEP)
    a.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
