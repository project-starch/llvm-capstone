#!/usr/bin/env python3
"""Check every mallocng file against verified upstream plus the reviewed port patches.

Also compare the unchanged allocation/release policy functions directly to upstream.
Representation changes (32-byte group header, capability-preserving arithmetic,
representable alignment and lifetime hooks) are explicit in patch 0009.
"""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ARCHIVE_SHA256 = 'a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4'

def tokens(text):
    return re.findall(r'\w+|[^\s]', re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S))

def body(text, name):
    m = re.search(r'\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{', text)
    if not m: raise ValueError('missing function: '+name)
    depth = 0
    for i in range(m.end()-1, len(text)):
        depth += (text[i]=='{') - (text[i]=='}')
        if not depth: return text[m.end()-1:i+1]
    raise ValueError('unterminated function: '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tree', type=Path); p.add_argument('archive', type=Path)
    a=p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:
        p.error('archive does not match musl 1.2.5')
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(a.archive) as t: t.extractall(d, filter='data')
        root=Path(d)/'musl-1.2.5'; rel=Path('src/malloc/mallocng')
        upstream={f.name:f.read_text() for f in (root/rel).iterdir() if f.is_file()}
        for name in ('0009-virtual-mallocng-capabilities.patch', '0010-virtual-mallocng-meta-area-at-once.patch',
                     '0011-virtual-mallocng-grow-mmapped-chunks-in-place.patch'):
            with (Path(__file__).parent/'musl-patches'/name).open() as patch:
                subprocess.run(['patch','-s','-p1','--no-backup-if-mismatch','--fuzz=0','-d',str(root)],stdin=patch,check=True)
        checks={str(rel/f.name): (a.tree/rel/f.name).is_file() and
                f.read_bytes()==(a.tree/rel/f.name).read_bytes()
                for f in (root/rel).iterdir() if f.is_file()}
        checks['file population']={f.name for f in (a.tree/rel).iterdir() if f.is_file() and f.suffix in ('.c','.h')}==set(upstream)
        for file,names in {'malloc.c':('try_avail','alloc_slot'),
                           'free.c':('okay_to_free','nontrivial_free'),
                           'meta.h':('size_overflows','step_seq','record_seq','account_bounce','decay_bounces','is_bouncing')}.items():
            current=(a.tree/rel/file).read_text()
            for name in names: checks[name]=tokens(body(upstream[file],name))==tokens(body(current,name).replace('GRP_SIZE','UNIT'))
        for name,ok in checks.items(): print(('PASS ' if ok else 'FAIL ')+name)
        return 0 if all(checks.values()) else 1
if __name__=='__main__': raise SystemExit(main())
