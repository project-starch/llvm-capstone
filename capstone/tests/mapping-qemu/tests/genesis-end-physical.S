// EXPECT: 0
// The genesis capability above the code ends at 2^56, not 2^63 (E2).
#include "capstone.h"
TEST_PROLOGUE
    LCC(t3, a1, x4)
    li t1, 1
    slli t1, t1, 56
    bne t3, t1, 1f
    PASS
1:  FAIL(1)
