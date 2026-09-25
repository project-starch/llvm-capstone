#!/usr/bin/env python3
"""Python-level trigger for case 19 / gh-153539.

Use-after-free in ``TextIOWrapper.tell()`` with a re-entrant decoder
(``Modules/_io/textio.c``).  ``tell()`` re-decodes the residual snapshot bytes
and calls the decoder's ``getstate()``, which is user code; a re-entrant
``seek()`` there drops the snapshot ``next_input`` (a borrowed bytes object)
that ``tell()`` is still using.  C-only: the ``_pyio`` fallback binds
``next_input`` as a strong local and cannot crash.

Reduced from CPython's own regression test,
``Lib/test/test_io.py::test_reentrant_seek_during_tell``.  Uses the C ``_io``
classes.  Snapshot pinned small (< 512 bytes), so it stays pymalloc memory; a
``--without-pymalloc`` ASan build reports the heap-use-after-free on pinned
v3.13.7.
"""
import codecs
import io


def main():
    wrapper = None
    armed = False

    class ReentrantDecoder(codecs.IncrementalDecoder):
        def decode(self, input, final=False):
            return bytes(input).decode("latin-1")

        def getstate(self):
            nonlocal armed
            if wrapper is not None and armed:
                armed = False
                wrapper.seek(0)          # re-entrant seek drops the snapshot
            return (b"", 0)

        def setstate(self, state):
            pass

    def search(name):
        if name != "reentrant_tell_test":
            return None
        return codecs.CodecInfo(
            name=name,
            encode=lambda s, e="strict": (s.encode("latin-1"), len(s)),
            decode=lambda b, e="strict": (bytes(b).decode("latin-1"), len(b)),
            incrementaldecoder=ReentrantDecoder,
        )

    codecs.register(search)
    try:
        raw = io.BytesIO(b"abcdefghijklmnop" * 8)
        wrapper = io.TextIOWrapper(io.BufferedReader(raw),
                                   encoding="reentrant_tell_test", newline="")
        wrapper._CHUNK_SIZE = 8
        wrapper.read(5)                  # leaves residual bytes in the snapshot
        armed = True
        assert isinstance(wrapper.tell(), int)   # stale read of freed snapshot
        wrapper.seek(0)
        assert isinstance(wrapper.tell(), int)
    finally:
        codecs.unregister(search)


if __name__ == "__main__":
    main()
    print("case19 gh-153539: sequence completed")
