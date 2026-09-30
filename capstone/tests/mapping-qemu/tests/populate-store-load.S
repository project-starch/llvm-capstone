// EXPECT: 0
// POPULATE backs the mapping's first page; the domain stores and loads
// through its delivered logical capability.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    li t0, 0x1234
    sw t0, 0(t3)
    lw t1, 0(t3)
    bne t0, t1, 1f
    li t0, 0x77
    li t1, 2048
    CINCOFFSET(t4, t3, t1)
    sb t0, 2047(t4)
    lbu t1, 2047(t4)
    bne t0, t1, 1f
    PASS
1:  FAIL(1)
_dom_end:
