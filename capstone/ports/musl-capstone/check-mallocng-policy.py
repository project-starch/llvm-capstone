#!/usr/bin/env python3
"""Check the mallocng decision code against the verified musl 1.2.5 archive.

This is a focused source guard, not a proof of equivalent runtime behavior.
Representation hooks are qualified separately with application tests.
"""
import argparse
import hashlib
from pathlib import Path
import re
import tarfile

ARCHIVE_SHA256 = 'a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4'


def tokens(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return re.findall(r'\w+|[^\s]', text)


def body(text, name):
    match = re.search(r'\b' + re.escape(name) + r'\s*\([^;{}]*\)\s*\{', text)
    if not match:
        raise ValueError('missing function: ' + name)
    start = match.end()-1
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if not depth:
            return text[start:i+1]
    raise ValueError('unterminated function: ' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tree', type=Path)
    p.add_argument('archive', type=Path)
    a = p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        p.error('archive does not match musl 1.2.5')
    prefix = 'musl-1.2.5/src/malloc/mallocng/'
    with tarfile.open(a.archive) as tar:
        original = {n: tar.extractfile(prefix+n).read().decode()
                    for n in ('malloc.c', 'free.c', 'realloc.c', 'meta.h')}
    current = {n: (a.tree/'src/malloc/mallocng'/n).read_text() for n in original}
    checks = {}

    def same(label, old, new):
        checks[label] = tokens(old) == tokens(new)

    for name in ('UNIT', 'IB', 'MMAP_THRESHOLD'):
        pattern = r'^#define ' + name + r'\s+(.*)$'
        old = re.findall(pattern, original['meta.h'], re.M)
        new = re.findall(pattern, current['meta.h'], re.M)
        checks[name] = len(old) == len(new) == 1 and old == new
    for name in ('size_classes', 'small_cnt_tab', 'med_cnt_tab'):
        pattern = r'\b' + name + r'\[.*?\]\s*=\s*(\{.*?\});'
        same(name, re.search(pattern, original['malloc.c'], re.S)[1],
             re.search(pattern, current['malloc.c'], re.S)[1])
    functions = {'meta.h': ('size_to_class', 'size_overflows', 'step_seq',
                            'record_seq', 'account_bounce', 'decay_bounces', 'is_bouncing'),
                 'malloc.c': ('try_avail', 'alloc_slot'),
                 'free.c': ('okay_to_free', 'nontrivial_free')}
    for file, names in functions.items():
        for name in names:
            same(name, body(original[file], name), body(current[file], name))
    old = body(original['malloc.c'], 'alloc_group').split('\t\tp = mmap(', 1)[0]
    new = body(current['malloc.c'], 'alloc_group').split('#ifdef CAPSTONE_MUSL_MALLOC', 1)[0]
    same('group size/count/mmap decisions', old, new)
    for file, begin, end, label in (
        ('malloc.c', '\tsc = size_to_class(n);', '\nsuccess:', 'slot/class selection'),
        ('realloc.c', '\tnew = malloc(n);', '\n}', 'moving realloc sequence'),
    ):
        same(label, original[file].split(begin, 1)[1].split(end, 1)[0],
             current[file].split(begin, 1)[1].split(end, 1)[0])
    for name in ('n <= avail_size', 'g->sizeclass>=48'):
        pattern = r'if \((' + re.escape(name) + r'.*?)\) \{'
        same('realloc ' + name, re.search(pattern, original['realloc.c'], re.S)[1],
             re.search(pattern, current['realloc.c'], re.S)[1])
    for name, passed in checks.items():
        print(('PASS ' if passed else 'FAIL ') + name)
    return 0 if checks and all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
