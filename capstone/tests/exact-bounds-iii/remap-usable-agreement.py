#!/usr/bin/env python3
"""Pin remap-usable-test.c's copy of the predicate to heap.c's real one.

remap-usable-test.c cannot include heap.c -- that is capability code -- so it reimplements
__capstone_malloc_remap_usable, exactly the duplication granule-agree-test.c exists to guard for
the granule. That copy drifted once: heap.c was tightened to check the length ACTUALLY mapped
(capstone_repr_ok(base, bytes)) while the test kept capstone_repr_round(bytes), so the test
carried on passing without testing the predicate it is named for.

This compares the two return expressions textually. Textual rather than behavioural because a
behavioural comparison would need the capability toolchain, and the failure mode here is edit
drift, which text catches.

  python3 remap-usable-agreement.py <repo root>
"""
import argparse, re, sys
from pathlib import Path

def body(text, name):
    m = re.search(r'\b' + re.escape(name) + r'\s*\([^;{}]*\)\s*\{', text)
    if not m:
        # Exit 2, not 1. A wrong function name must NOT read as a drift finding: the first version
        # of this script named the test's function `remap_usable` (it is `usable`) and exited 1,
        # so its negative control "fired" for the wrong reason and proved nothing.
        print("CANNOT CHECK: missing function %s -- this is not a drift result" % name)
        raise SystemExit(2)
    i = m.end() - 1
    d = 0
    for k in range(i, len(text)):
        d += (text[k] == '{') - (text[k] == '}')
        if not d:
            return text[i:k+1]
    raise SystemExit("unterminated: " + name)

def returns(src):
    """The return expressions, comments and whitespace stripped."""
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return [re.sub(r'\s+', '', r) for r in re.findall(r'return\s+([^;]+);', src)]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parents[3])
    a = ap.parse_args()
    heap = (a.root / 'capstone/runtime/virtual/heap.c').read_text()
    test = (a.root / 'capstone/tests/exact-bounds-iii/remap-usable-test.c').read_text()
    real = returns(body(heap, '__capstone_malloc_remap_usable'))
    copy = returns(body(test, 'usable'))
    print("  heap.c: %s" % real)
    print("  test  : %s" % copy)
    if real != copy:
        print("  DISAGREE -- the test is no longer testing heap.c's predicate")
        return 1
    print("  agree")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
