# MPY-S05 / upstream 15271, fix reverted on the pinned tree.
# Pre-fix array_append sets self->free = 8 BEFORE the m_renew that justifies it, so
# a failed grow leaves the array claiming eight slots it does not own and the next
# append writes past its buffer.
#
# Three earlier shapes failed, each for a reason worth keeping:
#   1-byte array, one ballast size  -> first == 0. MICROPY_BYTES_PER_GC_BLOCK is 32,
#     so a one-byte array grows to nine inside the block it already owns.
#   1-byte array, ballast down to 8 -> first == 0 for the same reason.
#   block-aligned candidates, ballast down to 8 -> uncaught MemoryError at the
#     enumerate(): the heap was filled so completely that no Python could run after.
# Hence: ONE large array, and COARSE ballast so ~2 KB of slack remains for the
# interpreter while still being far too little for a 4 KB block to be moved.
import gc

a = bytearray(b"x" * 4096)
# The victim is allocated IMMEDIATELY after a, before anything else, so it occupies
# the blocks the overflow will reach. Allocating it after the collect (as the fourth
# attempt did) put it elsewhere in the heap and the eight stray bytes landed in free
# space: untrapped, but with nothing to show they had hit anything live.
victim = bytearray(b"\x22" * 64)

ballast = []
try:
    while True:
        ballast.append(bytearray(2048))
except MemoryError:
    pass

# Allocation-free from here until the ballast is dropped: a small-int assignment
# and an exception that MicroPython has already preallocated.
first = 0
try:
    a.append(1)
except MemoryError:
    first = 1

del ballast
gc.collect()

hit = -1
if first:
    for _ in range(8):
        a.append(0xAA)          # eight writes past the end of a's buffer
    for i in range(64):
        if victim[i] != 0x22:
            hit = i
            break
print("S05", first, len(a), hit)
