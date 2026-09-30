// EXPECT: 0
// Three scattered frames back three consecutive logical pages; bounded
// pointer arithmetic crosses the page boundaries and each page keeps its own
// contents. The mapping capability is delinearized first: deriving a
// pointer from a linear capability moves it.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_I, 0, 20)
    POPULATE(s3, PAGE_E, 0x1000, 0)
    POPULATE(s3, PAGE_G, 0x2000, 0)
    CALL(s0, s0)
_dom:
    DELIN(t3)
    li t0, 0x1000
    CINCOFFSET(t4, t3, t0)
    CINCOFFSET(t5, t4, t0)
    li t0, 10
    sd t0, 8(t3)
    li t0, 20
    sd t0, 8(t4)
    li t0, 30
    sd t0, 8(t5)
    ld t1, 8(t3)
    li t0, 10
    bne t0, t1, 1f
    ld t1, 8(t4)
    li t0, 20
    bne t0, t1, 1f
    ld t1, 8(t5)
    li t0, 30
    bne t0, t1, 1f
    li t0, 0x1000 - 8
    CINCOFFSET(t6, t3, t0)
    ld t1, 16(t6)
    li t0, 20
    bne t0, t1, 1f
    PASS
1:  FAIL(1)
_dom_end:
