#!/usr/bin/env python3
"""Python-level trigger for case 12 / gh-143004.

Use-after-free in ``collections.Counter.update()`` (``Modules/_collectionsmodule.c``).
``oldval`` is borrowed from the mapping and passed to ``PyNumber_Add``; a user
``__add__`` can run arbitrary Python that mutates or clears the dict, freeing
``oldval`` while the C code still holds the borrowed pointer.  The container is
emptied and kept (unlike case 04, which abandons it), and the stale pointer is a
value rather than a key.

Reduced from CPython's own regression test,
``Lib/test/test_collections.py::test_update_reentrant_add_clears_counter``.
Standalone.  A ``--without-pymalloc`` ASan build reports the heap-use-after-free
on pinned v3.13.7.
"""
from collections import Counter


def main():
    c = Counter()
    key = object()

    class Evil(int):
        def __add__(self, other):
            c.clear()                   # frees the borrowed oldval
            return NotImplemented

    c[key] = Evil()
    c.update([key])                     # PyNumber_Add reads freed oldval
    assert c[key] == 1


if __name__ == "__main__":
    main()
    print("case12 gh-143004: sequence completed")
