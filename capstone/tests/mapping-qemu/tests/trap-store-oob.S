// EXPECT: 0x47
// A store beyond the capability's bounds traps with a store access fault (7).
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    li t3, 0x77
    li t4, 4096
    CINCOFFSET(t2, t2, t4)
    EXPECT_FAULT_AT(site)
site:
    sw t3, 0(t2)
    FAIL(1)
