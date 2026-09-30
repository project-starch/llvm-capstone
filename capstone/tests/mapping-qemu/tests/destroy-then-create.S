// EXPECT: 0
// DESTROY frees the entry and its range: a new CREATE for the same id and
// range succeeds with the next generation, delivered to the same context.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    CALL(s0, s0)
    MAPDETACH(s5, s2)
    MAPDESTROY(s5)
    LCC(t0, s5, x1)
    li t1, 7
    bne t0, t1, 1f
    MINT_PAGE(s1, PAGE_B)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    CALL(s0, s0)
1:  FAIL(2)
_dom:
    DOM_RETURN(_dom_resume)
_dom_resume:
    LCC(t0, t3, x8)
    li t1, 0x2000
    bne t0, t1, 1f
    PASS
1:  FAIL(1)
_dom_end:
