#!/usr/bin/env python3
"""Python-level trigger for case 10 / gh-142560.

Use-after-free in ``bytearray`` search-like methods (``Objects/bytearrayobject.c``).
``find``/``count``/``index``/``rfind``/``rindex`` (and ``in``, ``split``,
``rsplit``) cache ``PyByteArray_AS_STRING(self)`` and then convert the search
argument, which can run user code (``__buffer__`` / ``__index__``) that clears
or resizes the bytearray -- moving or freeing the storage the cached pointer
still refers to.  The fix bumps ``ob_exports`` around the operation.

Reduced from CPython's own regression test,
``Lib/test/test_bytes.py::test_search_methods_reentrancy_raises_buffererror``.
Standalone.  Storage kept small (< 512 bytes) so it stays pymalloc memory; a
``--without-pymalloc`` ASan build reports the heap-use-after-free on pinned
v3.13.7 (where the BufferError guard is absent).
"""


def main():
    class Evil:
        def __init__(self, ba):
            self.ba = ba

        def __buffer__(self, flags):
            self.ba.clear()             # frees / moves the cached storage
            return memoryview(self.ba)

        def __release_buffer__(self, view):
            view.release()

        def __index__(self):
            self.ba.clear()
            return ord("A")

    def make_case():
        ba = bytearray(b"A")
        return ba, Evil(ba)

    for name in ("find", "count", "index", "rindex", "rfind"):
        ba, evil = make_case()
        try:
            getattr(ba, name)(evil)     # cached storage read after clear
        except BufferError:
            pass                        # fixed build raises; pinned build faults

    ba, evil = make_case()
    try:
        evil in ba
    except BufferError:
        pass
    for name in ("split", "rsplit"):
        ba, evil = make_case()
        try:
            getattr(ba, name)(evil)
        except BufferError:
            pass


if __name__ == "__main__":
    main()
    print("case10 gh-142560: sequence completed")
