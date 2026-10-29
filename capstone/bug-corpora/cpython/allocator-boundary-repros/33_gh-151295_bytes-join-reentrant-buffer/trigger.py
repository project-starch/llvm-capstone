#!/usr/bin/env python3
"""Trigger for gh-151295 -- the body of upstream's regression test.

Upstream fix: 5c17ae6c1c ([3.13] back-port of 84a322aa15)
Test at the fix: Lib/test/test_bytes.py, BaseBytesTest.test_join_concurrent_buffer_mutation,
run for both type2test classes the test file instantiates (bytes and bytearray).
The body is copied unchanged except that subTest/assert calls are explicit checks.
"""


def make_seq(mutate):
    # Item is only referenced from the list slot, so mutate() frees it.
    class Item:
        def __buffer__(self, flags):
            mutate(seq)
            return memoryview(b'x')
    seq = [b'a', Item(), b'c']
    return seq


for type2test in (bytes, bytearray):
    for sep in (type2test(b''), type2test(b'::')):
        # Changing the list length is reported as a RuntimeError.
        seq = make_seq(lambda seq: seq.clear())
        try:
            sep.join(seq)
        except RuntimeError:
            pass

        # The list length is unchanged, so the size-change recheck
        # cannot fire: only keeping the item alive avoids the crash.
        def replace(seq):
            seq[1] = b'z'
        seq = make_seq(replace)
        print("TRIGGER", type2test.__name__, len(sep), sep.join(seq))
