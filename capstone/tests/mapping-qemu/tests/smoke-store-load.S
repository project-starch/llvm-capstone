// EXPECT: 0
// A store and a load through a minted linear page capability round-trip.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    li t3, 0x1234abcd
    sw t3, 16(t2)
    lw t4, 16(t2)
    bne t3, t4, 1f
    PASS
1:  FAIL(1)
