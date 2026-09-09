# MPY-T02 / CVE-2024-8947 and MPY-T05 / issue 13283, fixed by 4bed614e707c.
# The core case: extending a bytearray from ITSELF. array_extend sees
# self_in == arg_in, m_renew moves the buffer, and the argument's cached pointer
# is left dangling. Needs nothing but bytearray, which is why this is the only
# host-measured row that fits this domain.
b = bytearray(b"A" * 64)
b.extend(b)
ext = len(b)

# The fix names a second case, slice assignment from self. It is guarded by
# MICROPY_PY_ARRAY_SLICE_ASSIGN, which this profile does not enable, so report
# whether it was even reachable instead of failing the whole test on it.
try:
    c = bytearray(b"B" * 64)
    c[len(c):] = c
    sl = len(c)
except Exception:
    sl = -1

print("T02", ext, sl)
