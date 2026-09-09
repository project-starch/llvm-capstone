# MPY-S32, bounded arm. The 1000-item form faults with cause 24 and tells us only
# that SOMETHING downstream used a non-capability word as a pointer. The upstream
# report claims the overflow itself goes through and the crash comes later, from a
# neighbouring object it corrupted. That is a claim about which of the two events
# the hardware stopped, and it is testable: overrun by a bounded amount that stays
# well inside the heap, plant a victim first, and look.
#
#   "S32b <n> <hit>" with hit >= 0  -> the out-of-bounds write COMPLETED untrapped
#                                      and reached the victim: the report is right,
#                                      and the fault in the other arm is downstream.
#   fault                           -> the write itself was stopped, and the report's
#                                      account of the mechanism does not hold here.
victim = bytearray(b"\x22" * 256)

class Sneaky:
    def __len__(self):
        return 1
    def __iter__(self):
        for _ in range(48):      # 47 bytes past a one-byte buffer, far inside the heap
            yield 0xAA

b = bytearray(Sneaky())

hit = -1
for i in range(len(victim)):
    if victim[i] != 0x22:
        hit = i
        break
print("S32b", len(b), hit)
