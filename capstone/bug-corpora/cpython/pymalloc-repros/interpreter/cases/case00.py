#!/usr/bin/env python3
"""Python-level trigger for case 00 / gh-143543.

Re-entrant use-after-free in ``itertools.groupby``.  A user-defined ``__eq__``
re-enters the iterator and advances it while groupby is comparing the key
through a borrowed pointer; advancing drops the key's last reference and frees
its pymalloc block, and the comparison then reads through the stale pointer.

Reduced from CPython's own regression test for the fix,
``Lib/test/test_itertools.py::test_groupby_reentrant_eq_does_not_crash``
(commented "must pass with address sanitizer").  Standalone: no unittest, no
test.support.  On the pinned v3.13.7 interpreter this drives the defect; a
``--without-pymalloc`` ASan build reports the heap-use-after-free.
"""
import itertools


def main():
    class Key:
        def __init__(self, do_advance):
            self.do_advance = do_advance

        def __eq__(self, other):
            if self.do_advance:
                self.do_advance = False
                next(g)                 # re-enters groupby, frees the key
                return NotImplemented
            return False

    def keys():
        yield Key(True)
        yield Key(False)

    g = itertools.groupby([None, None], keys().send)
    next(g)
    next(g)                             # stale read through the freed key


if __name__ == "__main__":
    main()
    print("case00 gh-143543: sequence completed")
