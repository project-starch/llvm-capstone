"""Negative control for case 06: the side effect removed, the dict traffic kept.

The defect is that __eq__, called while the dict is inserting, clears the dict
under the insertion, so a float key ends up in a keys object that is still
flagged DK_IS_UNICODE and unicode_get_hash() then reads past it.

Here __eq__ still runs, still costs a call during the insertion, and still
returns False -- it just touches a different dict, so nothing is cleared under
the insertion in progress and the keys object's kind stays consistent with what
it holds.

Everything else is identical: the same two colliding X objects with the same
hash, the same float key added afterwards, and the same repr() of the result,
which is the statement that performs the defective read in the trigger.
"""
import os

other = {"a": 1}


class Benign:
    def __hash__(self):
        return -1

    def __eq__(self, other_obj):
        # The defect clears `d` here, under the insertion walking it. This
        # clears an unrelated dict, so the call happens and the structure
        # being inserted into does not change.
        other.clear()
        other["a"] = 1
        return False


d = {}
d[Benign()] = 3
d[Benign()] = 4
d[0.0] = 6
s = repr(d)
os.write(1, ("NEGATIVE-CONTROL no defect performed len=%d repr=%r\n"
             % (len(d), s)).encode())
