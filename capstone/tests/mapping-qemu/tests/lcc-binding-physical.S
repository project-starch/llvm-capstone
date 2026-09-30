// EXPECT: 0
// lcc selector 8 reads the node's binding word; a physical capability, minted
// or split, reads 0.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    LCC(t3, t2, x8)
    bnez t3, 1f
    li t4, PAGE_A + 2048
    SPLIT(t5, t2, t4)
    LCC(t3, t5, x8)
    bnez t3, 1f
    MREV(t6, t2)
    LCC(t3, t6, x8)
    bnez t3, 1f
    PASS
1:  FAIL(1)
