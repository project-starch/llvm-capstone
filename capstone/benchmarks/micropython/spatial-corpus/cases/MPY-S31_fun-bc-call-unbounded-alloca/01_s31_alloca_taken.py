# MPY-S31 arm A: WAS the alloca fallback taken at all?
# The previous shape ended in "call MemoryError" and could not say whether that came
# from the VM-state allocation (so no fallback) or from building the result list
# AFTER a successful alloca (so the fallback ran, untrapped). max() over N arguments
# gives the same large n_state and returns a SMALL INT, allocating nothing on the
# way out, so the two cases separate:
#   "called ok 0"      -> the fallback ran and nothing trapped it
#   "call MemoryError" -> the VM-state allocation raised instead of falling back
big = None
usedN = 0
for n in (8000, 5000, 3000, 2000, 1000):
    try:
        big = eval("lambda: max(" + "0," * n + "0)")
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
try:
    print("called ok", big())
except MemoryError:
    print("call MemoryError")
except Exception as e:
    print("call exc", type(e).__name__)
print("doneA")
