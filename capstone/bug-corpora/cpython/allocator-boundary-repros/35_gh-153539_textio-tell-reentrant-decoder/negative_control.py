#!/usr/bin/env python3
"""Strong negative control for case 35: the same reads and tells, with getstate not re-entering.

The decoder's getstate() still runs inside tell() -- the snapshot still holds residual bytes, so
tell() still re-decodes -- but it never calls wrapper.seek(0), so the snapshot tell() is using is
not released under it. The same seek(0) is then done after tell() returns. A fault here does not
depend on the defect.
"""
import codecs
import io

wrapper = None


class ReentrantDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return bytes(input).decode("latin-1")

    def getstate(self):
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
pos = wrapper.tell()
wrapper.seek(0)
pos2 = wrapper.tell()
codecs.unregister(search)
if not (isinstance(pos, int) and isinstance(pos2, int)):
    print("NEGATIVE-CONTROL SELFTEST-FAILED tell did not return ints")
    raise SystemExit(3)
print("NEGATIVE-CONTROL no defect performed")
