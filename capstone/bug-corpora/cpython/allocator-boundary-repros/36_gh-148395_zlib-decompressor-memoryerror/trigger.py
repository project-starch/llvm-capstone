#!/usr/bin/env python3
"""Trigger for gh-148395 / CVE-2026-6100 -- the body of upstream's regression test.

Upstream fix: c3cf71c336 ([3.13] back-port of 8fc66aef6d); it adds no test. The regression test is
upstream gh-154713 (commit 5bcd67189b), Lib/test/test_zlib.py
test_decompress_memoryerror_no_dangling_input. Its body is copied unchanged; only two things differ:
the two skip decorators are dropped, and the allocation-failure injector is imported as whichever of
CPython's _testcapi (the host oracle build) or the capstone image's _capstone_nomemory is present --
the same pyfailmalloc set_nomemory/remove_mem_hooks either way.
"""
import zlib
try:
    import _testcapi as nomem
except ImportError:
    import _capstone_nomemory as nomem

data = zlib.compress(b'x' * 4096)
for start in range(0, 40):
    d = zlib._ZlibDecompressor()
    try:
        nomem.set_nomemory(start, start + 1)
        try:
            d.decompress(bytearray(data), max_length=0)
        except MemoryError:
            pass
    finally:
        nomem.remove_mem_hooks()
    try:
        out = d.decompress(bytearray(data))
    except (MemoryError, ValueError, EOFError, OSError, zlib.error):
        continue
    assert isinstance(out, bytes)
    assert b'\x00' * 64 not in out
print("TRIGGER ran all starts")
