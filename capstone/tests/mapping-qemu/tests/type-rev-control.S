// EXPECT: 0
// Control for type-rev-load: after MREV the linear capability still loads.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    li t5, 0x55
    sw t5, 0(t2)
    MREV(t3, t2)
    lw t4, 0(t2)
    bne t4, t5, 1f
    PASS
1:  FAIL(1)
