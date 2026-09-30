// EXPECT: 0
// Control for perm-store-ro: the same store through an RW capability lands.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    TIGHTEN(t2, t2, PERM_RW)
    li t3, 0x77
    sw t3, 0(t2)
    lw t4, 0(t2)
    bne t3, t4, 1f
    PASS
1:  FAIL(1)
