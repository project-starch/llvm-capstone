#!/usr/bin/env python3
"""Python-level trigger for case 06 / gh-151295.

Use-after-free in ``bytes.join()`` / ``bytearray.join()`` via a re-entrant
``__buffer__`` (``Objects/stringlib/join.h``).  ``join`` fills an array of
``Py_buffer`` from the sequence's borrowed items; acquiring an item's buffer
runs its ``__buffer__``, which can mutate the sequence and drop the item's last
reference, freeing the payload that ``buffers[i].buf`` already points into.

Reduced from CPython's own regression test,
``Lib/test/test_bytes.py::test_join_concurrent_buffer_mutation``.  The item is
kept only in one list slot, so replacing that slot (length unchanged, so the
size-change recheck cannot fire) frees it -- only holding a strong reference
avoids the crash.  Payload pinned small (< 512 bytes), so it stays pymalloc
memory; a ``--without-pymalloc`` ASan build reports the heap-use-after-free.
"""


def make_seq(mutate):
    class Item:
        def __buffer__(self, flags):
            mutate(seq)                 # frees the borrowed item mid-join
            return memoryview(b"x" * 48)

    seq = [b"a", Item(), b"c"]
    return seq


def run(type2test):
    for sep in (type2test(b""), type2test(b"::")):
        # Length change is caught as RuntimeError (control, not a UAF).
        seq = make_seq(lambda s: s.clear())
        try:
            sep.join(seq)
        except RuntimeError:
            pass

        # Length unchanged: the recheck cannot fire, so the freed payload is
        # read unless the item is kept alive -- this is the defect.
        def replace(s):
            s[1] = b"z"

        seq = make_seq(replace)
        sep.join(seq)                   # stale read through buffers[i].buf


if __name__ == "__main__":
    run(bytes)
    run(bytearray)
    print("case06 gh-151295: sequence completed")
