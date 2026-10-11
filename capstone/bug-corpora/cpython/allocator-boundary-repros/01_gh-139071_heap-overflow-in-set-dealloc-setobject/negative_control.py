"""Negative control for case 01: the reentrancy removed, the set traffic kept.

The defect is that __eq__, called while the set is comparing, re-enters
_asyncio._register_task and mutates the set under the comparison. Here __eq__
still runs, still costs a call during the comparison, and still returns False --
it just registers an object that is already live and not the one being compared,
so no entry is added or moved under the iteration in progress.

Everything else is identical: the same two dummy entries, the same del, the same
two CorrupTrigger objects, the same number of _register_task calls.
"""
import _asyncio

class dummy(object):
    def __hash__(self):
        return 0

keepalive = dummy()
_asyncio._register_task(keepalive)

class Benign():
    def __hash__(self):
        return 0
    def __eq__(self, other):
        try:
            # The defect registers `self` here, from inside the comparison that
            # is walking the set. This registers an object already in it, so the
            # call happens and the structure does not change under the walk.
            _asyncio._register_task(keepalive)
        except:
            pass
        return False

a = [dummy(), dummy()]
_asyncio._register_task(a[0])
_asyncio._register_task(a[1])
del a
print(_asyncio._scheduled_tasks)
a = [Benign() for i in range(2)]
_asyncio._register_task(a[0])
_asyncio._register_task(a[1])
print("NEGATIVE-CONTROL no defect performed")
