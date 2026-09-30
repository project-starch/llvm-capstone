// EXPECT: 0x5d
// CREATE refuses a root page outside RAM (0x90000000 with 256 MiB of RAM)
// and keeps the operand: an unbacked table cannot be cleared or written.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, 0x90000000)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
