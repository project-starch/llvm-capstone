#!/usr/bin/env python3
"""Strong negative control for case 33: the same joins, with every Item kept alive.

Each Item is also held in `keep`, so the mutation inside __buffer__ (the clear, and the slot
replacement) no longer drops the item's last reference. Everything else -- the same sequences, the
same mutations, the same joins -- is as in trigger.py. A fault here does not depend on the defect.
"""
keep = []


def make_seq(mutate):
    class Item:
        def __buffer__(self, flags):
            mutate(seq)
            return memoryview(b'x')
    item = Item()
    keep.append(item)
    seq = [b'a', item, b'c']
    return seq


for type2test in (bytes, bytearray):
    for sep in (type2test(b''), type2test(b'::')):
        seq = make_seq(lambda seq: seq.clear())
        try:
            sep.join(seq)
        except RuntimeError:
            pass

        def replace(seq):
            seq[1] = b'z'
        seq = make_seq(replace)
        got = sep.join(seq)
        if got != sep.join([b'a', b'x', b'c']):
            print("NEGATIVE-CONTROL SELFTEST-FAILED join returned", got)
            raise SystemExit(3)
print("NEGATIVE-CONTROL no defect performed")
