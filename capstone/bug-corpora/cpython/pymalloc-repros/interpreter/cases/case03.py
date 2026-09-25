#!/usr/bin/env python3
"""Python-level trigger for case 03 / gh-142831.

Use-after-free in the C JSON encoder during re-entrant mutation
(``Modules/_json.c``).  Every item the encoder walks is borrowed; encoding an
item can run arbitrary Python (a ``default`` callback, a subclass ``items()``)
that clears the container, freeing an item the encoder still reaches through a
pointer sitting in the live storage array.

Reduced from CPython's own regression tests,
``Lib/test/test_json/test_speedups.py::test_mutate_dict_items_during_encode``
and ``test_mutate_list_during_encode``.  Uses only the public ``json`` C
accelerator (``_json.make_encoder`` via ``json.encoder.c_make_encoder``); no
_testcapi.  Drives the defect on pinned v3.13.7; a ``--without-pymalloc`` ASan
build reports the heap-use-after-free.
"""
import gc

import json.encoder

c_make_encoder = json.encoder.c_make_encoder
c_encode_basestring = json.encoder.c_encode_basestring


def gc_collect():
    for _ in range(3):
        gc.collect()


def mutate_dict_items():
    items = None

    class BadDict(dict):
        def items(self):
            nonlocal items
            items = [("boom", object())]
            return items

    def encode_str(obj):
        nonlocal items
        if items is not None:
            items.clear()          # frees the borrowed item mid-iteration
            items = None
            gc_collect()
        return '"x"'

    encoder = c_make_encoder(
        None, lambda o: "null",
        encode_str, None,
        ": ", ", ", False,
        False, True,
    )
    encoder(BadDict(real=1), 0)    # stale read through the live items list


def mutate_list():
    call_count = 0
    lst = [object() for _ in range(10)]

    def default(obj):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            lst.clear()            # frees a still-referenced list item
            gc_collect()
        return None

    encoder = c_make_encoder(
        None, default,
        c_encode_basestring, None,
        ": ", ", ", False,
        False, True,
    )
    encoder(lst, 0)


if __name__ == "__main__":
    mutate_dict_items()
    mutate_list()
    print("case03 gh-142831: sequence completed")
