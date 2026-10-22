#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

trigger.py is upstream's reproducer and prints no marker at the defective
access, so a run that ends without a fault cannot say whether the access
happened at all -- the runner records only "the trigger began". This probe
prints a marker on either side of the access, so reachability is measured
rather than inferred. It does not replace trigger.py.

The defect (gh-140551): X.__eq__ calls d.clear() while d is mid-insert, so the
float key 0.0 ends up in a keys object still flagged DK_IS_UNICODE. The
decisive access is repr(d) -> dict_repr -> _PyDict_Next ->
unicode_get_hash(key) at Objects/dictobject.c:440, which reads
_PyASCIIObject_CAST(key)->hash -- offset 24 of a 24-byte PyFloatObject.

Host ASan on the pinned 3.13.7 build (PYTHONMALLOC=malloc,
ASAN_OPTIONS=detect_leaks=0) reports, from exactly this sequence:
    heap-buffer-overflow, READ of size 8
    "0 bytes to the right of 24-byte region", allocated by PyFloat_FromDouble

Reading the output:
    CONFUSED  the dict reached the state the defect needs (len == 1)
    AT-ACCESS the next statement is the defective read
    PASSED    the read completed and this arm allowed it
A fault between AT-ACCESS and PASSED is the defective read being rejected.
No AT-ACCESS means the defect was never reached and the row is NO-READING,
not MISSED.
"""
import os


class X:
    def __hash__(self):
        return -1

    def __eq__(self, other):
        d.clear()
        return False


d = {}
d[X()] = 3
d[X()] = 4              # __eq__ fires here and clears d under the insert
d[0.0] = 6              # the float lands in a keys object flagged unicode

os.write(1, ("PROBE-140551 CONFUSED len=%d\n" % len(d)).encode())
os.write(1, b"PROBE-140551 AT-ACCESS\n")
s = repr(d)
os.write(1, ("PROBE-140551 PASSED repr=%r\n" % (s,)).encode())
