# MPY-S31 arm B: which guard fires first, MicroPython's or the hardware's?
# One alloca of n_state*16 fits easily in this domain's ~800 KB stack, and the heap
# is far too small to compile an n large enough to overflow it in one go. Recursion
# accumulates them instead. But this port sets MICROPY_STACK_CHECK with a 4096-byte
# margin against MPY_CSTACK_MAX, so the expectation is that mp_cstack_check raises
# RuntimeError LONG before the stack capability's bound is reached.
#
# If that is what happens it is a real finding and not a failed test: the port's own
# software guard masks the defect, so this row cannot demonstrate the hardware's
# stack bound however it is arranged, and the corpus should say so instead of
# leaving MPY-S31 open as if one more attempt would settle it.
# Runs LAST because it is the arm that might not return.
depth = 0
big = None
usedN = 0
for n in (3000, 2000, 1000, 500):
    try:
        big = eval("lambda f: max(" + "0," * n + "f(f))")
        usedN = n
        break
    except MemoryError:
        pass
print("compiled N", usedN)
ballast = []
try:
    while True:
        ballast.append(bytearray(1024))
except MemoryError:
    pass
print("filled")

def rec(f):
    global depth
    depth += 1
    return big(f)

try:
    rec(rec)
    print("returned", depth)
except MemoryError:
    print("MemoryError at depth", depth)
except RuntimeError:
    print("RuntimeError at depth", depth)      # MicroPython's own stack check
except Exception as e:
    print("exc", type(e).__name__, "at depth", depth)
print("doneB")
