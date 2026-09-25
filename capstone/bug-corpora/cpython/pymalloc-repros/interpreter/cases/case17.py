#!/usr/bin/env python3
"""Python-level trigger for case 17 / gh-151416.

Use-after-free in ``os.spawnv`` / ``os.spawnve`` via ``__fspath__``
(``Modules/posixmodule.c``).  The C spawn path walks *argv* with a getitem
function pointer, converting each borrowed item with ``fsconvert_strdup``; an
item's ``__fspath__`` can mutate the list and release its reference, freeing the
item the converter still uses.  The fix takes a strong reference to the item.
This is the sibling of case 16 (``_posixsubprocess.fork_exec``), fixed months
apart.

Reduced from CPython's own regression test,
``Lib/test/test_os.py::test_spawnv_arg_conversion_errors`` (which upstream marks
``@requires_native_spawnv``).

PLATFORM NOTE: the C ``posix.spawnv`` is exposed only on systems with a native
``spawnv()``.  On Linux ``os.spawnv`` is the pure-Python ``fork()`` + ``exec*()``
fallback in ``os.py``, so the C conversion path this defect lives in is NOT
reached, and the upstream test is skipped there.  On such platforms this case's
demonstrable driver is the C model (``case.c``); the ``fork_exec`` sibling
(case 16) exercises the same borrowed-argv shape natively on Linux.
"""
import os
import sys
import types


def native_spawnv():
    return isinstance(getattr(os, "spawnv", None), types.BuiltinFunctionType)


def main():
    if not native_spawnv():
        raise SystemExit(
            "trigger-17 gh-151416: os.spawnv is the pure-Python fallback on "
            "this platform (%s); the C posixmodule conversion path is "
            "unreachable here -- see case.c" % sys.platform)

    # A non-path argv item is a TypeError...
    try:
        os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, 123])
    except TypeError:
        pass
    # ...but a __fspath__ that mutates the argv list frees the borrowed item
    # while the converter still holds it (the defect this case is about).
    argv = [sys.executable, None]

    class EvilPath:
        def __fspath__(self):
            argv[1] = sys.executable      # drop the list's ref to this item
            return 12345                  # invalid type -> converter reads it

    argv[1] = EvilPath()
    try:
        os.spawnv(os.P_NOWAIT, sys.executable, argv)
    except Exception:
        pass


if __name__ == "__main__":
    main()
    print("trigger-17 gh-151416: reached")
