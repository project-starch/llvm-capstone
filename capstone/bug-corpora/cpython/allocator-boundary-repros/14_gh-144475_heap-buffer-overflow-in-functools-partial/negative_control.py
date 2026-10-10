"""Negative control for case 14: __setstate__ aimed at a different partial.

The defect is that __repr__, called while the partial's args tuple is being
walked, calls __setstate__ on THAT partial and frees the tuple under the walk.
Here __repr__ still fires, still builds the same new state, still calls
__setstate__ and still collects -- on a second partial that nothing is iterating.
The allocation and free traffic is the same; the object being freed is not the
one in use.
"""
import gc
from functools import partial

g_other = None

class EvilObject:
    def __init__(self, name, is_trigger=False):
        self.name = name
        self.is_trigger = is_trigger
        self.triggered = False

    def __repr__(self):
        global g_other
        if self.is_trigger and not self.triggered and g_other is not None:
            self.triggered = True
            new_state = (lambda x: x, ("replaced",), {}, None)
            # The defect calls this on the partial being walked. g_other is a
            # second one, so the same tuple is freed with nobody reading it.
            g_other.__setstate__(new_state)
            gc.collect()
        return f"EvilObject({self.name})"

evil1 = EvilObject("trigger", is_trigger=True)
evil2 = EvilObject("victim1")
evil3 = EvilObject("victim2")
evil4 = EvilObject("victim3")
evil5 = EvilObject("victim4")

p = partial(lambda: None, evil1, evil2, evil3, evil4, evil5)
g_other = partial(lambda: None, EvilObject("x"), EvilObject("y"))

del evil1, evil2, evil3, evil4, evil5

repr(p)
print("NEGATIVE-CONTROL no defect performed")
