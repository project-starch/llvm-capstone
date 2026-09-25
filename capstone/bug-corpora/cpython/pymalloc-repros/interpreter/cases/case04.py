#!/usr/bin/env python3
"""Python-level trigger for case 04 / gh-145244.

Use-after-free on a borrowed dict key in the JSON encoder (``Modules/_json.c``).
``key`` is borrowed from ``PyDict_Next`` and handed on with no ``Py_INCREF``; a
``default`` callback that clears the dict frees the key, and the error path then
formats it with ``_PyErr_FormatNote("%R", key)`` on freed memory.

This fix was never back-ported to 3.13.  Reduced from the fix's own regression
test on ``main``,
``Lib/test/test_json/test_dump.py::test_default_clears_dict_key_uaf``.
Standalone.

REACHABILITY: verified against the pinned v3.13.7 ASan build, this trigger
completes cleanly -- it does NOT fault.  In the pinned source
``encoder_encode_key_value`` does ``keystr = Py_NewRef(key)`` for a string key,
so the borrowed key is pinned across the callback, and with ``check_circular``
off the borrowed value is not re-read after the callback either.  The defect the
corpus models (``case.c``) is the pre-fix ``main`` code, where those references
were absent.  This upstream test is kept as the faithful specimen for the fix;
on the pinned interpreter the case's demonstrable driver is the C model.
"""
import json


def main():
    class Evil:
        pass

    class AlsoEvil:
        pass

    # Non-interned string key, so clearing the dict can actually free it.
    key = "A" * 100
    target = {key: Evil()}
    del key

    def evil_default(obj):
        if isinstance(obj, Evil):
            target.clear()             # bulk free; borrowed key now dangles
            return AlsoEvil()
        raise TypeError("not serializable")

    try:
        json.dumps(target, default=evil_default, check_circular=False)
    except TypeError:
        pass                           # expected; UAF is on this error path


if __name__ == "__main__":
    main()
    print("case04 gh-145244: sequence completed")
