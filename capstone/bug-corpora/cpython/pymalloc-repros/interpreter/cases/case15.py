#!/usr/bin/env python3
"""Python-level trigger for case 15 / gh-149449.

Use-after-free in ``_PyUnicode_GetNameCAPI`` (``Modules/unicodedata.c``).  The
``_PyUnicode_Name_CAPI`` struct is a bare ``PyMem_Malloc`` block (not a
``PyObject``); it exercises the ``PYMEM_DOMAIN_MEM`` -> pymalloc path.  Other
code (the ``\\N{...}`` escape handler, the ``namereplace`` error handler) caches
the CAPI pointer.  When ``unicodedata`` leaves ``sys.modules`` and its capsule
is garbage-collected, the destructor frees the struct while the cached pointer
still refers to it; the next use reads freed memory.

Reduced from CPython's own regression test,
``Lib/test/test_unicodedata.py::test_unicodedata_unload_reload`` (its child
process body).  Standalone: run this script in a fresh interpreter.  A
``--without-pymalloc`` ASan build reports the heap-use-after-free.
"""
import gc
import sys


def main():
    # Populate the cached _ucnhash_CAPI pointer via the namereplace handler and
    # the \N{...} escape compiler.
    assert "\N{GRINNING FACE}".encode("ascii", errors="namereplace") \
        == b"\\N{GRINNING FACE}"
    compile(r"x = '\N{LATIN CAPITAL LETTER A}'", "<x>", "exec")

    # Drop unicodedata and collect: the capsule destructor frees the CAPI block.
    del sys.modules["unicodedata"]
    gc.collect()

    # These reach through the cached (now dangling) pointer.
    assert "\N{WINKING FACE}".encode("ascii", errors="namereplace") \
        == b"\\N{WINKING FACE}"
    compile(r"x = '\N{LATIN CAPITAL LETTER B}'", "<x>", "exec")


if __name__ == "__main__":
    main()
    print("case15 gh-149449: sequence completed")
