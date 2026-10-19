#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

trigger.py marks completion but not the access: its final print means the
dangling CAPI was not caught, and nothing is printed at the moment it is used.
That is why the base arm's reading needed a negative control to be attributed
at all. This prints a marker on either side of the first use.

The defect (gh-149449, fixed by 60d843777c2c): `_PyUnicode_Name_CAPI` is a
16-byte PyMem_Malloc at Modules/unicodedata.c:1465, and the interpreter caches
a pointer to it in `_ucnhash_CAPI`. Dropping the module frees the struct and
leaves the cached pointer dangling, and the next `\\N{...}` escape LOADS a
function pointer out of the freed struct and CALLS it.

Two steps, and the arms differ in which one they stop:
  pools    rejects the LOAD, cause=25 INVALID_CAP on `ldc a4, 0x0(s2)`
  base     allows the load, and the MMU traps the CALL, cause=12
  CheriBSD allows the load, and the call fails the permission check, si_code=5

Reading the output:
    PRIMED     the cache is populated and the module is dropped
    AT-ACCESS  the next statement loads from the freed struct and calls it
    PASSED     the call returned and this arm allowed it
A fault between AT-ACCESS and PASSED is that use. No AT-ACCESS means the defect
was never reached and the row is NO-READING, not MISSED.
"""
import gc
import os
import sys

assert '\N{GRINNING FACE}'.encode('ascii', errors='namereplace') == b'\\N{GRINNING FACE}'
compile(r"x = '\\N{LATIN CAPITAL LETTER A}'", '<x>', 'exec')
del sys.modules['unicodedata']
gc.collect()
os.write(1, b"PROBE-149449 PRIMED cache populated, module dropped\n")
os.write(1, b"PROBE-149449 AT-ACCESS\n")
out = '\N{WINKING FACE}'.encode('ascii', errors='namereplace')
os.write(1, ("PROBE-149449 PASSED first use returned %r\n" % (out,)).encode())
compile(r"x = '\\N{LATIN CAPITAL LETTER B}'", '<x>', 'exec')
os.write(1, b"PROBE-149449 PASSED second use returned\n")
