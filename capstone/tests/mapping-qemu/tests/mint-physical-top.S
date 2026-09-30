// EXPECT: 0
// Control for mint-above-physical: the last page below 2^56 is mintable, and
// its end reads back as exactly 2^56.
#include "capstone.h"
TEST_PROLOGUE
    li t1, 1
    slli t1, t1, 56
    li t0, 4096
    sub t0, t1, t0
    DEBUGGENCAP(t2, t0, t1)
    LCC(t3, t2, x4)
    bne t3, t1, 1f
    PASS
1:  FAIL(1)
