#!/usr/bin/env python3
"""Trigger for gh-153539 -- the body of upstream's regression test.

Upstream fix: 081e0beae2 ([3.13] back-port of fa0ec86ab6)
Test at the fix: Lib/test/test_io.py, CTextIOWrapperTest.test_reentrant_seek_during_tell
The body is copied unchanged except that self.BytesIO/BufferedReader/TextIOWrapper are the C io
classes the CTextIOWrapperTest binds, and addCleanup/asserts are explicit.
"""
import codecs
import io

wrapper = None
armed = False


class ReentrantDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return bytes(input).decode("latin-1")

    def getstate(self):
        global armed
        if wrapper is not None and armed:
            armed = False
            wrapper.seek(0)
        return (b"", 0)

    def setstate(self, state):
        pass


def search(name):
    if name != "reentrant_tell_test":
        return None
    return codecs.CodecInfo(
        name=name,
        encode=lambda s, e='strict': (s.encode("latin-1"), len(s)),
        decode=lambda b, e='strict': (bytes(b).decode("latin-1"), len(b)),
        incrementaldecoder=ReentrantDecoder)


codecs.register(search)
raw = io.BytesIO(b"abcdefghijklmnop" * 8)
wrapper = io.TextIOWrapper(io.BufferedReader(raw), encoding="reentrant_tell_test", newline="")
wrapper._CHUNK_SIZE = 8
wrapper.read(5)
armed = True
print("TRIGGER tell", wrapper.tell())
wrapper.seek(0)
print("TRIGGER tell", wrapper.tell())
codecs.unregister(search)
