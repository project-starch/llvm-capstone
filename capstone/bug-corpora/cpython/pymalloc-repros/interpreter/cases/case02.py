#!/usr/bin/env python3
"""Python-level trigger for case 02 / gh-142829.

Use-after-free in ``Context.__eq__`` via a re-entrant ``ContextVar.set``.
``_PyHamt_Eq`` (Python/hamt.c) walks one hash-array-mapped trie while looking
keys up in the other, holding no reference to the node it stands in.  A user
``__eq__`` (or ``__hash__``) runs ``ContextVar.set``, which rebuilds the map and
can free the node; the walk keeps an interior pointer into that freed block.

Reduced from CPython's own regression tests,
``Lib/test/test_context.py::test_context_eq_reentrant_contextvar_set`` and
``...set_in_hash``.  Standalone.  Drives the defect on pinned v3.13.7; a
``--without-pymalloc`` ASan build reports the heap-use-after-free.
"""
import contextvars


def eq_variant():
    var = contextvars.ContextVar("v")
    ctx1 = contextvars.Context()
    ctx2 = contextvars.Context()

    class ReentrantEq:
        def __eq__(self, other):
            ctx1.run(lambda: var.set(object()))   # rebuilds ctx1's map
            return True

    ctx1.run(var.set, ReentrantEq())
    ctx2.run(var.set, object())
    ctx1 == ctx2                                   # walk reads freed node


def hash_variant():
    var = contextvars.ContextVar("v")
    ctx1 = contextvars.Context()
    ctx2 = contextvars.Context()

    class ReentrantHash:
        def __hash__(self):
            ctx1.run(lambda: var.set(object()))
            return 0

        def __eq__(self, other):
            return isinstance(other, ReentrantHash)

    ctx1.run(var.set, ReentrantHash())
    ctx2.run(var.set, ReentrantHash())
    ctx1 == ctx2


if __name__ == "__main__":
    eq_variant()
    hash_variant()
    print("case02 gh-142829: sequence completed")
