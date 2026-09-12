# MPY-S32 / upstream 18617, OPEN, present at the pin.
# array_construct sizes the buffer from a user __len__; array_extend_impl then fills
# it from __iter__ without re-checking that bound. Report 1, yield 1000.
# On the host at the pin this is SIGSEGV (rc 139), verified before this ran.
class Sneaky:
    def __len__(self):
        return 1
    def __iter__(self):
        for _ in range(1000):
            yield 65

b = bytearray(Sneaky())
print("S32", len(b), b[0])
