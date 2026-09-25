# gh-112127: atexit.unregister() holds a borrowed callback across a compare
# whose user __eq__ re-enters unregister and mutates the callback array.
# atexit_unregister loops over state->callbacks, comparing cb->func with the
# target via PyObject_RichCompareBool; the target's __eq__ unregisters the very
# callback being compared, freeing its struct and shifting the array, and the
# loop then deletes through the stale index. The backport pins the callback;
# 3.13.7 (our pin) corrupts the callback array (crash in atexit_delete_cb).
import atexit

class Evil:
    __hash__ = None
    def __eq__(self, other):
        atexit.unregister(other)   # frees the callback struct being compared
        return True

def a(): pass
def b(): pass
def c(): pass

for f in (a, b, c, a, b, c):
    atexit.register(f)
atexit.unregister(Evil())          # re-entrant delete corrupts the array
print("trigger-08: reached")
