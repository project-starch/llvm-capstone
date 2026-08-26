# MPY-T28 / upstream 322: gc_realloc returns NULL while the heap still has room.
# The original report grows one block with gc_realloc from C and sees it fail at
# 37000 bytes with 67072 free. bytearray.extend goes through m_renew -> gc_realloc,
# so the same question is askable from Python: how much is still free at the
# moment growth fails?
import gc
gc.collect()
total_before = gc.mem_free()
b = bytearray(100)
grew = 0
try:
    for i in range(2000):
        b.extend(bytearray(1000))
        grew = len(b)
except MemoryError:
    pass
free_after = gc.mem_free()
print("T28 grew_to", grew, "free_at_failure", free_after, "free_at_start", total_before)
