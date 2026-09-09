# MPY-S01 / upstream 19314, the bytes/str path (py/objstr.c).
# len(seq) * n is computed in unchecked size_t: 8 * 2**61 == 2**64 == 0, so the
# allocation is EMPTY while the write that follows is 2**64 bytes long. Kept to
# small ints on purpose (2**61 < MP_SMALL_INT_MAX) so it triggers under
# MICROPY_LONGINT_IMPL_NONE, which is what this domain is built with.
# Present at the pin: stock MicroPython at 2e3304a segfaults on this line.
r = b"aaaaaaaa" * (2 ** 61)
print("S01b", len(r))
