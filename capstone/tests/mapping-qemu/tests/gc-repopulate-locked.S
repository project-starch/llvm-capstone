// EXPECT: 0x5a
// A PTE revoked from above stays locked after the node collector cleared its
// tag: POPULATE on the same page is still refused (cause 26). Locked does
// not depend on a tag or a node. Control: gc-repopulate-locked-control.S.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MINT_PAGE(s3, PAGE_E)
    MREV(s7, s3)
    li a0, 1
    slli a0, a0, 57
    li a5, 20
    MAPPOPULATE(s2, s3)
    REVOKE(s7)
    SUPERVISOR_GC(t0)
    MINT_PAGE(s3, PAGE_G)
    li a0, 1
    slli a0, a0, 57
    li a5, 0
    EXPECT_FAULT_AT(site)
site:
    MAPPOPULATE(s2, s3)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
