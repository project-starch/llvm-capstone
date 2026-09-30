// EXPECT: 0x5a
// REVOKE refuses the detach handle (cause 26): the only operations on it are
// DETACH, UNMAP and DESTROY, so a delinearized mapping can never come back
// readable through a plain revocation.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    EXPECT_FAULT_AT(site)
site:
    REVOKE(s2)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
