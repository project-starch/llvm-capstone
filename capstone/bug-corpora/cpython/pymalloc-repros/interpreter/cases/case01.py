#!/usr/bin/env python3
"""Python-level trigger for case 01 / gh-146613.

Re-entrant use-after-free in ``itertools._grouper`` (the child iterator of
``groupby``).  ``_grouper_next`` compares the grouper's target key with the
parent's current key, both borrowed.  A user ``__eq__`` re-enters and advances
the parent, replacing ``gbo->currkey`` and freeing the object the comparison
still holds a borrowed pointer to.

Reduced from CPython's own regression test,
``Lib/test/test_itertools.py::test_grouper_reentrant_eq_does_not_crash``.
Standalone.  Drives the defect on pinned v3.13.7; a ``--without-pymalloc`` ASan
build reports the heap-use-after-free.
"""
import itertools


def main():
    grouper_iter = None

    class Key:
        __hash__ = None

        def __init__(self, do_advance):
            self.do_advance = do_advance

        def __eq__(self, other):
            nonlocal grouper_iter
            if self.do_advance:
                self.do_advance = False
                if grouper_iter is not None:
                    try:
                        next(grouper_iter)      # advances parent, frees currkey
                    except StopIteration:
                        pass
                return NotImplemented
            return True

    def keyfunc(element):
        if element == 0:
            return Key(do_advance=True)
        return Key(do_advance=False)

    g = itertools.groupby(range(4), keyfunc)
    key, grouper_iter = next(g)
    items = list(grouper_iter)              # stale read through the freed key
    assert len(items) == 1, items


if __name__ == "__main__":
    main()
    print("case01 gh-146613: sequence completed")
