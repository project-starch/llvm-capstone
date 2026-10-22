#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

trigger.py runs upstream's test method, which passes whether or not the
defective read happens -- the test only asserts that the final decode raises
UnicodeDecodeError, and it does either way. So a silent arm cannot be read as
"the defect was not caught" until the access is marked. This marks it.

The defect (gh-157325, fixed by af1be50e4cfd) is an order-of-checks bug in
Modules/cjkcodecs/_codecs_cn.c:

    if (c == '~') {
        unsigned char c2 = INBYTE2;   /* reads inbuf[1] ... */
        REQUIRE_INBUF(2);             /* ... before checking there are 2 */

The one-byte buffer comes from the SECOND call below:
decoder.decode(b'', True) has size == 0 and pendingsize == 1, so
multibytecodec.c:1183 does PyMem_Malloc(1), copies the pending '~' into it,
and hz_decode then reads offset 1 of a 1-byte block.

Host ASan on the pinned build reports, from exactly this sequence:
    heap-buffer-overflow, READ of size 1
    "0 bytes to the right of 1-byte region", allocated by PyMem_Malloc

It also prints the interpreter's allocator names, because the whole question
on the pools arm is whether that 1-byte PyMem_Malloc went through pymalloc and
so through patch 0014's per-block narrowing.
"""
import os

try:
    import _testinternalcapi
    os.write(1, ("PROBE-157325 allocators=%r\n"
                 % (_testinternalcapi.pymem_getallocatorsname(),)).encode())
except Exception as exc:                                  # noqa: BLE001
    os.write(1, ("PROBE-157325 allocators unavailable: %r\n" % (exc,)).encode())

import codecs

decoder = codecs.getincrementaldecoder("hz")()
os.write(1, ("PROBE-157325 pending-set first=%r\n"
             % (decoder.decode(b"~"),)).encode())
os.write(1, b"PROBE-157325 AT-ACCESS\n")
try:
    decoder.decode(b"", True)
except UnicodeDecodeError:
    os.write(1, b"PROBE-157325 PASSED raised UnicodeDecodeError\n")
else:
    os.write(1, b"PROBE-157325 PASSED no exception\n")
