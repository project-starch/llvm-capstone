#!/usr/bin/env python3
"""Python-level trigger for case 14 / gh-146011.

Use-after-free in ``_decimal``'s signal dict (``Modules/_decimal/_decimal.c``).
A context's ``flags`` (a ``SignalDict``) holds ``traps->flags``, a borrowed
interior pointer into the context object's storage.  Deleting the context frees
that storage without clearing the signal dict's pointer, and the signal dict can
survive; a later ``repr()`` of it reads through the dangling interior pointer.
The gap is unbounded -- it waits for a ``repr`` that may never come.

Reduced from CPython's own regression test,
``Lib/test/test_decimal.py::CContextFlags.test_signaldict_repr``.  Uses the C
``_decimal``.  Standalone.  On pinned v3.13.7 ``repr`` reads the freed context;
a ``--without-pymalloc`` ASan build reports the heap-use-after-free.  (The fixed
build instead raises ``ValueError: invalid signal dict``.)
"""
import _decimal as C


def main():
    ctx = C.Context(prec=7)
    mapping = ctx.flags                 # SignalDict aliasing ctx's storage
    del ctx                             # frees the context; mapping survives
    try:
        repr(mapping)                   # reads the dangling interior pointer
    except ValueError:
        pass                            # fixed build: "invalid signal dict"


if __name__ == "__main__":
    main()
    print("case14 gh-146011: sequence completed")
