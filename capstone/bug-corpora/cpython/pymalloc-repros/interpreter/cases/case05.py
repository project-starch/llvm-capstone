#!/usr/bin/env python3
"""Python-level trigger for case 05 / gh-148660.

Use-after-free in ``OrderedDict.copy()`` on re-entrant mutation
(``Objects/odictobject.c``).  ``_odict_FOREACH`` advances by reading
``node->next`` out of the node it just processed.  Copying can run arbitrary
Python (a key ``__eq__``/``__hash__``, a subclass ``__getitem__``) that clears
the source and frees the node, so the walk loads a link pointer from a freed
block and follows it -- a plausible wrong answer, not just a crash.

Reduced from CPython's own regression tests,
``Lib/test/test_ordered_dict.py::test_issue148660_copy_clear_in_key_eq`` and
``...copy_clear_in_subclass_getitem``.  Uses the C ``OrderedDict``
(``collections.OrderedDict``).  Standalone.  On pinned v3.13.7 the walk reads
the freed node; a ``--without-pymalloc`` ASan build reports it.
"""
from collections import OrderedDict


def copy_clear_in_key_eq():
    armed = False
    calls = 0

    class Key:
        def __hash__(self):
            return 1

        def __eq__(self, other):
            nonlocal calls
            if armed:
                calls += 1
                if calls == 2:
                    od.clear()            # frees the node the walk stands on
            return self is other

    od = OrderedDict()
    od[Key()] = "v1"
    od[Key()] = "v2"
    armed = True
    try:
        od.copy()                         # stale link load and follow
    except RuntimeError:
        pass                              # "OrderedDict mutated during iteration"


def copy_clear_in_subclass_getitem():
    class OD(OrderedDict):
        def __getitem__(self, key):
            od.clear()
            return "v"

    od = OD([(1, "v1"), (2, "v2")])
    try:
        od.copy()
    except RuntimeError:
        pass


if __name__ == "__main__":
    copy_clear_in_key_eq()
    copy_clear_in_subclass_getitem()
    print("case05 gh-148660: sequence completed")
