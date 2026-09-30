// EXPECT: 0x5d
// CREATE refuses a root capability that is not exactly one aligned page.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    li t0, PAGE_A
    li t1, PAGE_A + 8192
    DEBUGGENCAP(s1, t0, t1)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
