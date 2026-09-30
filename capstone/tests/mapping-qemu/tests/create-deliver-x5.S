// EXPECT: 0
// Control for the reserved delivery registers: x5 (t0) receives the mapping
// capability and holds it at the domain's first instruction.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 5)
    MAPCREATE(s2, s1, s0)
    li t0, 0
    CALL(s0, s0)
_dom:
    LCC(t1, t0, x1)
    bnez t1, 1f
    LCC(t1, t0, x8)
    li t2, 0x1000
    bne t1, t2, 1f
    PASS
1:  FAIL(1)
_dom_end:
