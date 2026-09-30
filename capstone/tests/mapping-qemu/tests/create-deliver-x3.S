// EXPECT: 0x5d
// CREATE refuses x3 as the delivery register: cscall writes it after
// delivery (x1 the sealed-return capability, x3 the fabricated gp), so the
// mapping capability would be lost. On 0efac317 it was accepted and lost.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 3)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
