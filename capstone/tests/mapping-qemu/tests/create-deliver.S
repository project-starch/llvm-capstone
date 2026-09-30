// EXPECT: 0
// CREATE reserves a mapping, returns a REV detach handle, consumes the root
// page, and delivers the mapping capability into the named register of the
// sealed context when the context is called: linear, bound (id 0, gen 1),
// based at 2^57 with the requested rights.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    LCC(t0, s2, x1)
    li t1, 2
    bne t0, t1, 1f
    LCC(t0, s1, x1)
    li t1, 7
    bne t0, t1, 1f
    li t3, 0
    CALL(s0, s0)
_dom:
    LCC(t0, t3, x1)
    bnez t0, 1f
    LCC(t0, t3, x8)
    li t1, 0x1000
    bne t0, t1, 1f
    LCC(t0, t3, x3)
    li t1, 1
    slli t1, t1, 57
    bne t0, t1, 1f
    LCC(t0, t3, x5)
    li t1, 6
    bne t0, t1, 1f
    PASS
1:  FAIL(1)
_dom_end:
