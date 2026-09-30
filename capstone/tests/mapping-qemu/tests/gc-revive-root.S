// EXPECT: 0x59
// A revoked root table stays dead after the collector recycled its node
// number and an unrelated MREV took it: the domain's parked pointer still
// faults (cause 25). The registry checks the node's binding, not only its
// number. Control: gc-revive-root-control.S.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    MREV(s7, s1)
    MINT_PAGE(s6, PAGE_H)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
    REVOKE(s7)
    SUPERVISOR_GC(t0)
    MREV(s8, s6)
    CALL(s0, s0)
    FAIL(3)
_dom:
    li t0, 4242
    sw t0, 0(t3)
    STC(x1, t3, 0x100)
    DOM_RETURN(_resume)
_resume:
    LDC(t3, x1, 0x100)
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(t3)
    FAIL(1)
_dom_end:
