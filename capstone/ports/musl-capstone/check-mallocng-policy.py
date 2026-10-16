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


# The ONE narrowing of this guard, authorised by the project lead for option (iii).
#
# Every large class's stride must be a multiple of its own bounds granule, or the capability the
# allocator derives over an object is not exactly representable: the compressed encoder truncates
# an unaligned base DOWNWARD, so the object would gain authority below itself. Seven strides are
# 16 x odd and so guarantee only 16-byte alignment while their capacities need 32 to 128. Raising
# those seven entries by 1 to 4 units each is the whole change; `size_to_class` is untouched, and
# the rounding that drives requests into the right class lives in the runtime's own malloc wrapper,
# outside musl.
#
# This is deliberately NOT a relaxation to "the table may differ". It pins the seven indices AND
# their values, so an eighth edit, a different value, or a reverted entry all still FAIL. The guard
# itself is the external collaborator's work; only this exception is ours.
ALLOWED_SIZE_CLASS_EDITS = {36: 1170, 38: 1638, 39: 2048, 41: 2732, 43: 4096, 45: 5464, 47: 8192}


# The single authorised edit inside alloc_group's size/count/mmap decision region, pinned as literal
# text on BOTH sides. alloc_group's cnt==1 branch mapped exactly pageround(n+IB+UNIT), which leaves
# no room for the granule-sized group header the capability representation needs; the one slot then
# ended past the mapping and malloc returned NULL for 95.5% of single-slot sizes from n=16365 up.
# The mapping SIZE therefore has to change. The policy -- which class, which count, whether this
# branch is taken at all -- does not, and is why the condition below is still evaluated on upstream's
# own `req`.
#
# Like the size-class exception, this is NOT a relaxation to "the region may differ": the old and new
# texts are both exact, so a different expression, a second edit elsewhere in the region, or a revert
# all still FAIL. Negative-tested in test-mallocng-policy-narrowing.py.
ALLOWED_GROUP_SIZE_EDIT_OLD = '\t\tif (!nosmall && cnt<=7) {\n\t\t\treq += IB + UNIT;\n\t\t\treq += -req & (pagesize-1);\n\t\t\tif (req<size+UNIT || (req>=4*pagesize && 2*cnt>usage)) {\n\t\t\t\tcnt = 1;\n\t\t\t\tneeded = req;\n'
ALLOWED_GROUP_SIZE_EDIT_NEW = "\t\tif (!nosmall && cnt<=7) {\n\t\t\tsize_t want = req;\n\t\t\treq += IB + UNIT;\n\t\t\treq += -req & (pagesize-1);\n\t\t\tif (req<size+UNIT || (req>=4*pagesize && 2*cnt>usage)) {\n\t\t\t\tcnt = 1;\n#ifdef CAPSTONE_MUSL_MALLOC\n\t\t\t\t/* The path selection above stays on upstream's req; only the SIZE\n\t\t\t\t * changes, to leave room for the granule-sized group header that\n\t\t\t\t * this one slot would otherwise be pushed past the mapping by. */\n\t\t\t\tneeded = cap_single_slot_bytes(want, pagesize);\n\t\t\t\tif (!needed) { free_meta(m); return 0; }\n#else\n\t\t\t\tneeded = req;\n#endif\n"


def normalise_group_region(current_body):
    """Substitute the ONE authorised edit back to upstream's text, so the comparison that follows
    is against upstream. Returns the text unchanged if the authorised edit is not present verbatim,
    which makes the comparison fail -- the behaviour we want for any unauthorised change."""
    if ALLOWED_GROUP_SIZE_EDIT_NEW not in current_body:
        return current_body
    return current_body.replace(ALLOWED_GROUP_SIZE_EDIT_NEW, ALLOWED_GROUP_SIZE_EDIT_OLD, 1)


def size_class_values(text):
    pattern = r'\bsize_classes\[.*?\]\s*=\s*\{(.*?)\};'
    body = re.search(pattern, text, re.S)[1]
    return [int(v) for v in re.findall(r'\d+', body)]


def size_classes_ok(original, current, allowed=None):
    """Upstream's table except exactly the authorised entries, at exactly their authorised values."""
    allowed = ALLOWED_SIZE_CLASS_EDITS if allowed is None else allowed
    old, new = size_class_values(original), size_class_values(current)
    if len(old) != len(new):
        return False
    differ = {i: new[i] for i in range(len(old)) if old[i] != new[i]}
    return differ == allowed


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
    for name in ('small_cnt_tab', 'med_cnt_tab'):
        pattern = r'\b' + name + r'\[.*?\]\s*=\s*(\{.*?\});'
        same(name, re.search(pattern, original['malloc.c'], re.S)[1],
             re.search(pattern, current['malloc.c'], re.S)[1])
    checks['size_classes'] = size_classes_ok(original['malloc.c'], current['malloc.c'])
    functions = {'meta.h': ('size_to_class', 'size_overflows', 'step_seq',
                            'record_seq', 'account_bounce', 'decay_bounces', 'is_bouncing'),
                 'malloc.c': ('try_avail', 'alloc_slot'),
                 'free.c': ('okay_to_free', 'nontrivial_free')}
    for file, names in functions.items():
        for name in names:
            same(name, body(original[file], name), body(current[file], name))
    old = body(original['malloc.c'], 'alloc_group').split('\t\tp = mmap(', 1)[0]
    # Normalise the authorised size edit FIRST, so the only '#ifdef CAPSTONE_MUSL_MALLOC' left in
    # the region is the mapping call the split below has always truncated at.
    new = normalise_group_region(body(current['malloc.c'], 'alloc_group'))
    new = new.split('#ifdef CAPSTONE_MUSL_MALLOC', 1)[0]
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
