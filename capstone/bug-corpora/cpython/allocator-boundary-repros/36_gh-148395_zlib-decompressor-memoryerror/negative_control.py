#!/usr/bin/env python3
"""Strong negative control for case 36: the same decompressor traffic without the failure injection.

trigger.py forces a MemoryError inside each decompress() with set_nomemory, which is what leaves the
stale input pointer the defect reads on the next call. Here set_nomemory is never installed, so every
allocation succeeds, no decompress() takes its error path, and the second decompress() reads a live
buffer. The same forty _ZlibDecompressor objects and the same two decompress() calls each still run.
A fault here does not depend on the defect.
"""
import zlib

data = zlib.compress(b'x' * 4096)
for start in range(0, 40):
    d = zlib._ZlibDecompressor()
    try:
        d.decompress(bytearray(data), max_length=0)
    except (MemoryError, ValueError, EOFError, OSError, zlib.error):
        pass
    try:
        out = d.decompress(bytearray(data))
    except (MemoryError, ValueError, EOFError, OSError, zlib.error):
        continue
    assert isinstance(out, bytes)
print("NEGATIVE-CONTROL no defect performed")
