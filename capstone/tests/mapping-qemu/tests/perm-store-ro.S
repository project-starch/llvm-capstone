// EXPECT: 0x5b
// A store through a read-only capability traps with insufficient permission
// (0x1b). Positive control: perm-store-rw.S stores through the same page RW.
#include "capstone.h"
TEST_PROLOGUE
    li t0, PAGE_A
    li t1, PAGE_A + 4096
    DEBUGGENCAP(t2, t0, t1)
    TIGHTEN(t2, t2, PERM_RO)
    li t3, 0x77
    EXPECT_FAULT_AT(site)
site:
    sw t3, 0(t2)
    FAIL(1)
