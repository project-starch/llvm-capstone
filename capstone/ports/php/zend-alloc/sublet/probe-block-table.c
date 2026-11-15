/* Exercise the block table through the real PHP adapter: the block found again from the
 * payload after a re-issue, a realloc that fails, one that grows past the 16 KiB slack threshold,
 * and emalloc(0). Until 2026-11-15 this probe exercised the saved-parent prefix the table replaced. */
#include "zend-cap-mode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HDR 48UL
#define MODE (ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_REVOKE | ZEND_CAP_MODE_RETURN)
#define CHECK(c) do { if (!(c)) { printf("TABLE_FAIL line=%d\n", __LINE__); return 1; } } while (0)

int main(int argc, char **argv)
{
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    void *block, *p, *old, *fresh;
    unsigned long stats[8];
    zend_cap_set_mode(MODE);
    zend_cap_release(NULL);
    block = zend_cap_block(HDR + 56);
    CHECK(block);
    p = zend_cap_issue(block, HDR, 54);
    memset(p, 0x5a, 54);
    old = p;
    CHECK(zend_cap_reclaim(p, HDR) == block);
    p = zend_cap_issue(block, HDR, 55);
    ((volatile unsigned char *)p)[54] = 0x6b;
    CHECK(((unsigned char *)p)[54] == 0x6b);
    CHECK(zend_cap_header(p, HDR) == block);
    if (mode == 1) {
        puts("TABLE_FAULT_READY stale-free-after-reissue"); fflush(stdout);
        zend_cap_reclaim(old, HDR);
        return 1;
    }
    if (mode == 2) {
        puts("TABLE_FAULT_READY header-underread"); fflush(stdout);
        return ((volatile unsigned char *)p)[-1];
    }
    /* Allocation failure must leave both the old child and its block usable. */
    CHECK(zend_cap_reblock(block, (size_t)-1, HDR + 55) == NULL);
    CHECK(zend_cap_reblock(block, (size_t)-129, HDR + 55) == NULL);
    CHECK(((unsigned char *)p)[54] == 0x6b);
    CHECK(zend_cap_header(p, HDR) == block);
    fresh = zend_cap_reblock(block, HDR + 200000, HDR + 55);
    CHECK(fresh);
    p = zend_cap_issue(fresh, HDR, 200000);
    CHECK(((unsigned char *)p)[0] == 0x5a);
    CHECK(((unsigned char *)p)[54] == 0x6b);
    ((volatile unsigned char *)p)[199999] = 0x7c;
    CHECK(zend_cap_reclaim(p, HDR) == fresh);
    zend_cap_release(fresh);

    /* Cache reuse keeps a single block; each issue gets a fresh child. */
    block = zend_cap_block(HDR);
    CHECK(block);
    for (unsigned i = 0; i < 10000; ++i) {
        p = zend_cap_issue(block, HDR, i % 2); /* emalloc(0), emalloc(1) */
        *(volatile unsigned char *)p = 0x41;
        CHECK(zend_cap_reclaim(p, HDR) == block);
    }
    zend_cap_release(block);
    zend_cap_stats(stats);
    /* Two malloc calls: the realloc is no block of its own. */
    CHECK(stats[0] == 2 && stats[2] == 10003 && stats[3] == 10003 && stats[6] == 1);
    puts("TABLE_PASS cache-54-to-55 realloc-failure realloc-large zero-size reuse-10000");
    return 0;
}
