// EXPECT: 0x5a
// A load through a revocation handle traps with unexpected capability type
// (0x1a): a handle carries no data authority. Control: type-rev-control.S.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    MREV(t3, t2)
    EXPECT_FAULT_AT(site)
site:
    lw t4, 0(t3)
    FAIL(1)
