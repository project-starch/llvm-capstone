// EXPECT: 0x5d
// The debug mint refuses a range reaching into the logical region (E2):
// illegal operand value (0x1d). Control: mint-physical-top.S.
#include "capstone.h"
TEST_PROLOGUE
    li t0, 1
    slli t0, t0, 57
    li t1, 4096
    add t1, t0, t1
    DEBUGGENCAP(t2, t0, t1)
    FAIL(1)
