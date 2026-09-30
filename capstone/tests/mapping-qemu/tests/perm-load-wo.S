// EXPECT: 0x5b
// A load through a write-only capability traps with insufficient permission.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    TIGHTEN(t2, t2, PERM_WO)
    lw t4, 0(t2)
    FAIL(1)
