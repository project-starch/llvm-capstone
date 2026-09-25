#!/usr/bin/env python3
"""Python-level trigger for case 07 / gh-148395 (CVE-2026-6100).

Dangling input pointer in the decompressors (``Modules/zlibmodule.c``,
``_bz2module.c``, ``_lzmamodule.c``).  Each keeps a stream whose ``next_in``
points into the caller's input buffer while ``decompress()`` runs.  On the error
path the fix adds ``next_in = NULL``; without it, after an error ``next_in`` is
left pointing into the released caller buffer, and the next ``decompress()``
call reads or writes through the stale pointer.

Reduced from CPython's own regression test,
``Lib/test/test_zlib.py::test_decompress_memoryerror_no_dangling_input`` (and the
bz2/lzma twins).  The zlib arm is used, per the corpus's module constraints.

REACHABILITY: in the pinned v3.13.7, the only ``goto error`` that leaves
``next_in`` set (the tail-copy ``PyMem_Malloc`` failure in ``decompress()``) is
reachable ONLY by making that allocation fail -- the data-error path already
clears ``next_in``.  Upstream forces the failure with
``_testcapi.set_nomemory``.  The guest interpreter is built
``--disable-test-modules``, so ``_testcapi`` is absent and this exact path
cannot be driven from Python there; the demonstrable driver on the guest is the
C model (``case.c``).  On a build that ships ``_testcapi`` this script runs the
faithful upstream trigger and a ``--without-pymalloc`` ASan build reports the
heap-use-after-free.
"""
import zlib


def main():
    if not hasattr(zlib, "_ZlibDecompressor"):
        raise SystemExit("trigger-07 gh-148395: zlib._ZlibDecompressor missing")
    try:
        import _testcapi
    except ImportError:
        raise SystemExit(
            "trigger-07 gh-148395: _testcapi unavailable (--disable-test-modules); "
            "the buggy error path is only reachable via MemoryError injection -- "
            "see case.c")

    data = zlib.compress(b"x" * 4096)
    for start in range(0, 40):
        d = zlib._ZlibDecompressor()
        try:
            _testcapi.set_nomemory(start, start + 1)
            try:
                d.decompress(bytearray(data), max_length=0)
            except MemoryError:
                pass
        finally:
            _testcapi.remove_mem_hooks()
        try:
            out = d.decompress(bytearray(data))   # resumes from stale next_in
        except (MemoryError, ValueError, EOFError, OSError, zlib.error):
            continue
        assert isinstance(out, bytes)
        assert b"\x00" * 64 not in out


if __name__ == "__main__":
    main()
    print("trigger-07 gh-148395: reached")
