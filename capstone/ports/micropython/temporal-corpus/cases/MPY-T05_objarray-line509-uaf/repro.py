# MPY-T05 / upstream 13283, and the same defect as CVE-2024-8947 (MPY-T02).
# Trigger taken from the fix commit 4bed614e707c, which names both cases:
# extending a bytearray from ITSELF, and assigning to a slice from itself. In
# both, m_renew moves the buffer and the argument's cached pointer is left
# dangling.
#
# Run this against the fix's PARENT, ce491ab0d1; at the pin it is already fixed.
b = bytearray(b"A" * 64)
b.extend(b)
print("extend  len", len(b), "head", bytes(b[:2]))

c = bytearray(b"B" * 64)
c[len(c):] = c
print("slice   len", len(c), "head", bytes(c[:2]))
