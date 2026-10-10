#!/usr/bin/env python3
"""Trigger for gh149449 -- dropping unicodedata leaves _ucnhash_CAPI dangling.

Upstream fix: 60d843777c2c
Test file at the fix: Lib/test/test_unicodedata.py
Test: UnicodeMiscTest.test_unicodedata_unload_reload (kept here as provenance)

The upstream test hands this body to a CHILD interpreter through
script_helper.assert_python_ok, which isolates the crash from the test runner.
That is a problem for the Capstone arms and not for the defect: the application
heap size is compiled into the binary (CAPSTONE_APPLICATION_ARENA_BYTES), so a
child wants a second heap of the same size, and on the sublet arm the domain
answered "cannot allocate application heap: No space left on device". The parent
then reported a plain test failure, which reads exactly like a silent arm.

So the body runs in-process here. It was checked to be the same defect, not a
near one: on the pinned 3.13.7 host build it gives SIGSEGV, and under ASan the
same signature the child gives -- SEGV with pc equal to the faulting address,
because the fault is an indirect CALL through the dangling pointer rather than a
load or a store. Case 07 departs from its upstream test for the same reason.

The struct comes from PyMem_Malloc(sizeof(_PyUnicode_Name_CAPI)) = 16 bytes at
Modules/unicodedata.c:1465, which was measured to land in a pymalloc pool.
"""
import gc, sys

assert '\N{GRINNING FACE}'.encode('ascii', errors='namereplace') == b'\\N{GRINNING FACE}'
compile(r"x = '\\N{LATIN CAPITAL LETTER A}'", '<x>', 'exec')
del sys.modules['unicodedata']
gc.collect()
# The dangling _ucnhash_CAPI is called from here on. On an arm that catches it,
# control never reaches the print below.
assert '\N{WINKING FACE}'.encode('ascii', errors='namereplace') == b'\\N{WINKING FACE}'
compile(r"x = '\\N{LATIN CAPITAL LETTER B}'", '<x>', 'exec')
print("gh149449 trigger ran to completion; the stale CAPI was not caught")
